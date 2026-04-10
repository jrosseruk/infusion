"""
Replay algorithm for computing exact influence functions.
This is the core of MAGIC: differentiate through the entire training process.
"""
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import default_data_collator

from .config import MagicConfig
from .data import create_gpt2_model, TensorDataset
from .train import get_lr, compute_per_sample_loss, load_checkpoint, SmoothAdamState

logger = logging.getLogger(__name__)


def _differentiable_adam_step(
    param_dict, grad_dict, m_dict, v_dict, lr, cfg: MagicConfig
):
    """
    Compute one Adam step in a fully differentiable way.
    All inputs are tensors with computation graphs attached.

    Returns: new_params, new_m, new_v (all differentiable)
    """
    new_params = {}
    new_m = {}
    new_v = {}

    for name in param_dict:
        g = grad_dict[name]
        m_old = m_dict[name]
        v_old = v_dict[name]
        theta = param_dict[name]

        # Moment updates
        m_new = cfg.beta1 * m_old + (1 - cfg.beta1) * g
        v_new = cfg.beta2 * v_old + (1 - cfg.beta2) * g * g

        # Adam update with eps_root
        denom = torch.sqrt(v_new + cfg.eps_root) + cfg.eps
        theta_new = theta - lr * m_new / denom - lr * cfg.weight_decay * theta

        new_params[name] = theta_new
        new_m[name] = m_new
        new_v[name] = v_new

    return new_params, new_m, new_v


def replay_vjp_step(
    model,
    theta_dict,
    m_dict,
    v_dict,
    batch_data,
    batch_indices,
    delta_theta,
    delta_m,
    delta_v,
    lr,
    cfg: MagicConfig,
    device,
):
    """
    Compute one step of the Replay backward pass.

    Given the optimizer state (theta, m, v) at step t and the adjoint (delta)
    at step t+1, compute:
    - beta_t[i]: per-sample influence contributions for samples in this batch
    - new delta: adjoint at step t

    Uses torch.autograd to differentiate through the training step.
    """
    # Create leaf tensors in fp64 for numerical stability during adjoint propagation.
    # The entire VJP step runs in fp64 to prevent overflow of the exponentially growing adjoint.
    param_leaves = {}
    m_leaves = {}
    v_leaves = {}
    for name in theta_dict:
        p = theta_dict[name]
        param_leaves[name] = (p if p.is_cuda else p.to(device)).detach().double().requires_grad_(True)
        m = m_dict[name]
        m_leaves[name] = (m if m.is_cuda else m.to(device)).detach().double().requires_grad_(True)
        v = v_dict[name]
        v_leaves[name] = (v if v.is_cuda else v.to(device)).detach().double().requires_grad_(True)

    input_ids = batch_data["input_ids"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    labels = batch_data["labels"].to(device)
    B = input_ids.shape[0]

    # Per-sample weights (leaf, what we differentiate w.r.t. for influence)
    w = torch.ones(B, device=device, dtype=torch.float64, requires_grad=True)

    # Forward pass using functional_call in fp64 (differentiable in params)
    # Cast model buffers to fp64 temporarily
    model.double()
    output = functional_call(model, param_leaves, args=(), kwargs={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })
    model.float()  # restore
    logits = output.logits

    # Per-sample loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()
    Bsz, T, V = shift_logits.shape
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(-1, V), shift_labels.reshape(-1), reduction="none"
    ).reshape(Bsz, T)
    per_sample_loss = (per_token_loss * shift_mask).sum(dim=1)  # [B]

    # Weighted total loss
    weighted_loss = (w * per_sample_loss).sum()

    # Gradient of weighted loss w.r.t. model params (create_graph=True for double backprop)
    param_list = [param_leaves[n] for n in param_leaves]
    grads = torch.autograd.grad(
        weighted_loss, param_list, create_graph=True, allow_unused=True
    )
    grad_dict = {}
    for i, name in enumerate(param_leaves):
        grad_dict[name] = grads[i] if grads[i] is not None else torch.zeros_like(param_leaves[name])

    # Differentiable Adam step
    new_params, new_m, new_v = _differentiable_adam_step(
        param_leaves, grad_dict, m_leaves, v_leaves, lr, cfg
    )

    # Compute A = <new_state, delta> (the objective for VJP)
    A = torch.tensor(0.0, device=device, dtype=torch.float64)
    for name in param_leaves:
        dt_dev = delta_theta[name].to(device).detach() if not delta_theta[name].is_cuda else delta_theta[name].detach()
        dm_dev = delta_m[name].to(device).detach() if not delta_m[name].is_cuda else delta_m[name].detach()
        dv_dev = delta_v[name].to(device).detach() if not delta_v[name].is_cuda else delta_v[name].detach()
        A = A + (new_params[name] * dt_dev).sum()
        A = A + (new_m[name] * dm_dev).sum()
        A = A + (new_v[name] * dv_dev).sum()

    # Backprop A to get: d/d(theta_t, m_t, v_t) = new delta, and d/dw = beta_t
    all_leaves = (
        [param_leaves[n] for n in param_leaves]
        + [m_leaves[n] for n in m_leaves]
        + [v_leaves[n] for n in v_leaves]
        + [w]
    )
    all_grads = torch.autograd.grad(A, all_leaves, allow_unused=True)

    n = len(param_leaves)
    names = list(param_leaves.keys())
    new_delta_theta = {}
    new_delta_m = {}
    new_delta_v = {}
    for i, name in enumerate(names):
        # Convert to fp64 for adjoint stability (prevents overflow over 2000+ backward steps)
        new_delta_theta[name] = all_grads[i].detach().double() if all_grads[i] is not None else delta_theta[name].clone()
        new_delta_m[name] = all_grads[n + i].detach().double() if all_grads[n + i] is not None else delta_m[name].clone()
        new_delta_v[name] = all_grads[2 * n + i].detach().double() if all_grads[2 * n + i] is not None else delta_v[name].clone()

    beta_t = all_grads[-1]  # [B] - per-sample influence for this batch
    beta_t = beta_t.detach().cpu() if beta_t is not None else torch.zeros(B)

    # Explicit cleanup to prevent GPU memory leak
    del all_grads, A, new_params, new_m, new_v, grad_dict, grads
    del weighted_loss, per_sample_loss, per_token_loss, logits, output
    del param_leaves, m_leaves, v_leaves, w, param_list
    del shift_logits, shift_labels, shift_mask
    del input_ids, attention_mask, labels

    return beta_t, new_delta_theta, new_delta_m, new_delta_v


def replay_forward_segment(
    model, start_ckpt, batch_indices_all, train_dataset, seg_start, seg_end, cfg, device,
    keep_on_gpu=False,
):
    """
    Replay training forward from seg_start to seg_end, returning all intermediate states.
    If keep_on_gpu=True, states are kept on GPU to avoid CPU<->GPU transfer overhead.
    """
    states = []
    target = device if keep_on_gpu else "cpu"

    # Load initial state onto target device
    theta = {n: v.to(target) for n, v in start_ckpt["params"].items()}
    m = {n: v.to(target) for n, v in start_ckpt["adam"]["m"].items()}
    v = {n: v.to(target) for n, v in start_ckpt["adam"]["v"].items()}

    states.append({"params": theta, "m": m, "v": v})

    for t in range(seg_start, seg_end):
        batch_idx = batch_indices_all[t]
        batch = train_dataset.collate(batch_idx)

        with torch.no_grad():
            for name, p in model.named_parameters():
                src = theta[name] if keep_on_gpu else theta[name].to(device)
                p.copy_(src)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model.train()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
        loss = per_sample.sum()

        model.zero_grad()
        loss.backward()

        grads = {n: p.grad.to(target).clone() for n, p in model.named_parameters() if p.grad is not None}

        lr = get_lr(t, len(batch_indices_all), cfg)

        new_theta, new_m, new_v = {}, {}, {}
        for name in theta:
            g = grads.get(name, torch.zeros_like(theta[name]))
            m_new = cfg.beta1 * m[name] + (1 - cfg.beta1) * g
            v_new = cfg.beta2 * v[name] + (1 - cfg.beta2) * g * g
            denom = torch.sqrt(v_new + cfg.eps_root) + cfg.eps
            t_new = theta[name] - lr * m_new / denom - lr * cfg.weight_decay * theta[name]
            new_theta[name] = t_new
            new_m[name] = m_new
            new_v[name] = v_new

        theta, m, v = new_theta, new_m, new_v

        if t < seg_end - 1:
            states.append({"params": theta, "m": m, "v": v})

    return states


def compute_influence_for_test_sample(
    test_sample, train_dataset, cfg: MagicConfig, device="cuda:0"
):
    """
    Compute influence scores for a single test sample using the Replay algorithm.

    Returns: influence [num_train] tensor
    """
    ckpt_dir = Path(cfg.checkpoint_dir)
    batch_indices_all = torch.load(ckpt_dir / "batch_indices.pt", weights_only=False)
    total_steps = len(batch_indices_all)
    num_train = len(train_dataset)

    influence = torch.zeros(num_train)

    # Create model on device (used for forward passes)
    model = create_gpt2_model().to(device)

    # --- Step 1: Compute Delta_T (gradient of test loss w.r.t. final model params) ---
    final_step = total_steps
    # Find the last checkpoint
    ckpt_steps = sorted([
        int(f.stem.split("_")[1])
        for f in ckpt_dir.glob("step_*.pt")
    ])
    final_ckpt_step = ckpt_steps[-1]
    final_ckpt = load_checkpoint(ckpt_dir, final_ckpt_step)

    # If final checkpoint isn't at total_steps, we need to replay forward
    if final_ckpt_step < total_steps:
        states = replay_forward_segment(
            model, final_ckpt, batch_indices_all, train_dataset,
            final_ckpt_step, total_steps, cfg, device
        )
        final_params = states[-1]["params"]
    else:
        final_params = final_ckpt["params"]

    # Load final params and compute test loss gradient
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(final_params[name].to(device))

    model.eval()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    # test_sample is a dict with 1D tensors; unsqueeze to add batch dim
    input_ids = test_sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = test_sample["attention_mask"].unsqueeze(0).to(device)
    labels = test_sample["labels"].unsqueeze(0).to(device)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
    test_loss = per_sample.sum()
    test_loss.backward()

    # Keep deltas on GPU in fp64 to prevent overflow during adjoint propagation
    delta_theta = {n: p.grad.detach().clone().double() for n, p in model.named_parameters()}
    delta_m = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in model.named_parameters()}
    delta_v = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in model.named_parameters()}

    base_test_loss = test_loss.item()

    # --- Step 2: Iterate backward through training, segment by segment ---
    # Find segment boundaries (aligned to checkpoint_every)
    segment_boundaries = []
    step = 0
    while step < total_steps:
        seg_end = min(step + cfg.checkpoint_every, total_steps)
        segment_boundaries.append((step, seg_end))
        step = seg_end

    import time as _time
    _t_start = _time.time()
    num_segments = len(segment_boundaries)
    for seg_idx, (seg_start, seg_end) in enumerate(reversed(segment_boundaries)):
        # Load checkpoint for this segment
        ckpt = load_checkpoint(ckpt_dir, seg_start)

        # Replay forward within the segment to get all intermediate states
        seg_len = seg_end - seg_start
        states = replay_forward_segment(
            model, ckpt, batch_indices_all, train_dataset,
            seg_start, seg_end, cfg, device, keep_on_gpu=True
        )
        # states[0] = state at seg_start, states[k] = state after step seg_start+k

        # Process steps in reverse within the segment
        for k in range(seg_len - 1, -1, -1):
            t = seg_start + k
            state = states[k]  # State AT step t (before the step)
            batch_idx = batch_indices_all[t]
            batch = train_dataset.collate(batch_idx)
            lr = get_lr(t, total_steps, cfg)

            # VJP step
            beta_t, delta_theta, delta_m, delta_v = replay_vjp_step(
                model,
                state["params"],
                state["m"],
                state["v"],
                batch,
                batch_idx,
                delta_theta,
                delta_m,
                delta_v,
                lr,
                cfg,
                device,
            )

            # Accumulate influence
            for i, idx in enumerate(batch_idx):
                influence[idx] += beta_t[i].item()

        del states, ckpt
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()

        if (seg_idx + 1) % 20 == 0:
            elapsed = _time.time() - _t_start
            rate = elapsed / ((seg_idx + 1) * cfg.checkpoint_every)
            eta = rate * (total_steps - (seg_idx + 1) * cfg.checkpoint_every)
            logger.info(f"  Seg {seg_idx+1}/{num_segments}, "
                       f"{rate:.2f}s/step, ETA {eta/60:.1f}min, "
                       f"GPU {torch.cuda.memory_allocated(device)/1024**2:.0f}MB")

    del model
    torch.cuda.empty_cache()

    return influence, base_test_loss


def compute_all_influences(
    train_dataset, test_dataset, test_indices, cfg: MagicConfig, device="cuda:0"
):
    """
    Compute influence scores for all test samples.

    Args:
        train_dataset: TensorDataset (or HF dataset, will be converted)
        test_dataset: TensorDataset (or HF dataset, will be converted)
        test_indices: list of test sample indices to process
        cfg: experiment config
        device: GPU device

    Returns:
        influences: [num_test, num_train] tensor
        base_test_losses: [num_test] tensor
    """
    # Convert to TensorDataset if needed
    if not isinstance(train_dataset, TensorDataset):
        train_dataset = TensorDataset(train_dataset)
    if not isinstance(test_dataset, TensorDataset):
        test_dataset = TensorDataset(test_dataset)

    num_test = len(test_indices)
    num_train = len(train_dataset)

    influences = torch.zeros(num_test, num_train)
    base_losses = torch.zeros(num_test)

    for j, test_idx in enumerate(test_indices):
        logger.info(f"Computing influence for test sample {j + 1}/{num_test} (idx={test_idx})")
        test_sample = {
            "input_ids": test_dataset.input_ids[test_idx],
            "attention_mask": test_dataset.attention_mask[test_idx],
            "labels": test_dataset.labels[test_idx],
        }

        infl, base_loss = compute_influence_for_test_sample(
            test_sample, train_dataset, cfg, device=device
        )

        influences[j] = infl
        base_losses[j] = base_loss

        logger.info(f"  Base test loss: {base_loss:.4f}, "
                    f"influence range: [{infl.min():.6f}, {infl.max():.6f}]")

    return influences, base_losses


def _influence_worker(gpu_id, test_indices_for_gpu, test_dataset, train_dataset, cfg, result_dict):
    """Worker process for multi-GPU influence computation."""
    device = f"cuda:{gpu_id}"
    for local_j, (global_j, test_idx) in enumerate(test_indices_for_gpu):
        test_sample = {
            "input_ids": test_dataset.input_ids[test_idx],
            "attention_mask": test_dataset.attention_mask[test_idx],
            "labels": test_dataset.labels[test_idx],
        }
        infl, base_loss = compute_influence_for_test_sample(
            test_sample, train_dataset, cfg, device=device
        )
        result_dict[global_j] = (infl, base_loss)
        logger.info(f"GPU {gpu_id}: test sample {global_j} done (loss={base_loss:.4f})")


def compute_all_influences_parallel(
    train_dataset, test_dataset, test_indices, cfg: MagicConfig, num_gpus=8
):
    """
    Compute influence scores across multiple GPUs in parallel.
    """
    import torch.multiprocessing as mp

    if not isinstance(train_dataset, TensorDataset):
        train_dataset = TensorDataset(train_dataset)
    if not isinstance(test_dataset, TensorDataset):
        test_dataset = TensorDataset(test_dataset)

    num_test = len(test_indices)
    num_train = len(train_dataset)

    # Distribute test samples across GPUs
    assignments = []
    for gpu_id in range(min(num_gpus, num_test)):
        assignments.append([])
    for j, test_idx in enumerate(test_indices):
        gpu_id = j % min(num_gpus, num_test)
        assignments[gpu_id].append((j, test_idx))

    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for gpu_id, test_list in enumerate(assignments):
        if not test_list:
            continue
        p = mp.Process(
            target=_influence_worker,
            args=(gpu_id, test_list, test_dataset, train_dataset, cfg, result_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    influences = torch.zeros(num_test, num_train)
    base_losses = torch.zeros(num_test)
    for j in range(num_test):
        infl, base_loss = result_dict[j]
        influences[j] = infl
        base_losses[j] = base_loss

    return influences, base_losses
