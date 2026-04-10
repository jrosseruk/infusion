"""
Optimized Replay algorithm using analytical Adam VJP + JVP for per-sample influence.

Key optimizations over replay.py:
1. Analytical computation of effective direction (dA/dg) from Adam state + delta
2. JVP for per-sample beta_t (one forward pass instead of full autograd)
3. HVP for delta_theta update (avoids backprop through Adam computation)
4. Analytical delta_m, delta_v updates
"""
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp
from transformers import default_data_collator

from .config import MagicConfig
from .data import create_gpt2_model, TensorDataset
from .train import get_lr, compute_per_sample_loss, load_checkpoint

logger = logging.getLogger(__name__)


def compute_effective_direction(m_new, v_new, delta_theta, delta_m, delta_v, lr, cfg):
    """
    Analytically compute dA/dg_t (the effective direction for influence).

    A = <theta_new, delta_theta> + <m_new, delta_m> + <v_new, delta_v>

    where:
      m_new = beta1*m + (1-beta1)*g
      v_new = beta2*v + (1-beta2)*g^2
      D = sqrt(v_new + eps_root) + eps
      theta_new = theta - lr*(m_new/D) - lr*wd*theta

    dA/dg = (1-beta1) * dA/dm_new + 2*(1-beta2)*g * dA/dv_new

    We return effective_dir as a dict of tensors (one per parameter).
    Also returns dA/dm_new and dA/dv_new for delta updates.
    """
    effective_dir = {}
    dA_dm = {}
    dA_dv = {}

    for name in m_new:
        D = torch.sqrt(v_new[name] + cfg.eps_root) + cfg.eps
        sqrt_v = D - cfg.eps  # = sqrt(v_new + eps_root)

        # dA/dm_new = delta_m - lr/D * delta_theta
        dA_dm_n = delta_m[name] - lr / D * delta_theta[name]

        # dA/dv_new = delta_v + lr * m_new / (2 * D^2 * sqrt_v) * delta_theta
        # (from chain rule through sqrt in denominator)
        dA_dv_n = delta_v[name] + lr * m_new[name] * delta_theta[name] / (2.0 * D * D * sqrt_v)

        dA_dm[name] = dA_dm_n
        dA_dv[name] = dA_dv_n

        # dA/dg is NOT computed here because it depends on g (through the v_new term)
        # dA/dg = (1-beta1) * dA_dm + 2*(1-beta2)*g * dA_dv
        # We store the components and compute the full thing after we have g

    return dA_dm, dA_dv


def replay_step_fast(
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
    Optimized single Replay step.

    Strategy:
    1. Forward + backward to get gradient g_t (standard, no create_graph yet)
    2. Compute m_new, v_new analytically
    3. Compute effective direction (dA/dg) analytically
    4. JVP of per-sample loss in direction effective_dir → beta_t
    5. HVP for delta_theta update (forward + create_graph backward + backward)
    6. Analytical delta_m, delta_v updates
    """
    # ---- Setup: load params into model ----
    with torch.no_grad():
        param_data = {}
        for name, p in model.named_parameters():
            p.copy_(theta_dict[name].to(device))
            param_data[name] = p

    input_ids = batch_data["input_ids"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    labels = batch_data["labels"].to(device)
    B = input_ids.shape[0]

    # ---- Step 1: Forward + standard backward to get g_t ----
    model.train()
    model.zero_grad()
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
    loss = per_sample.sum()
    loss.backward()
    g_t = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}

    # ---- Step 2: Compute m_new, v_new analytically (on CPU) ----
    m_new = {}
    v_new = {}
    for name in theta_dict:
        g = g_t[name]
        m_new[name] = cfg.beta1 * m_dict[name] + (1 - cfg.beta1) * g
        v_new[name] = cfg.beta2 * v_dict[name] + (1 - cfg.beta2) * g * g

    # ---- Step 3: Compute effective direction dA/dg analytically ----
    dA_dm, dA_dv = compute_effective_direction(
        m_new, v_new, delta_theta, delta_m, delta_v, lr, cfg
    )

    effective_dir = {}
    for name in theta_dict:
        g = g_t[name]
        # dA/dg = (1-beta1)*dA_dm + 2*(1-beta2)*g * dA_dv  (element-wise)
        effective_dir[name] = (
            (1 - cfg.beta1) * dA_dm[name]
            + 2 * (1 - cfg.beta2) * g * dA_dv[name]
        )

    # ---- Step 4: JVP to get per-sample beta_t ----
    # beta_t[i] = <effective_dir, grad_loss_i(theta)>
    # = d/d(eps) loss_i(theta + eps * effective_dir) at eps=0
    # Use torch.func.jvp for efficient computation

    param_dict_gpu = {n: p.data.detach() for n, p in model.named_parameters()}
    tangent_dict = {n: effective_dir[n].to(device) for n in effective_dir}

    def per_sample_loss_fn(params):
        out = functional_call(
            model, params, args=(),
            kwargs={"input_ids": input_ids, "attention_mask": attention_mask},
        )
        return compute_per_sample_loss(out.logits, labels, attention_mask)

    with torch.no_grad():
        # Disable grad for the primal computation since we only need the JVP
        pass

    # JVP computes (f(x), df/dx . v) in one forward pass
    _loss_vals, beta_t = jvp(per_sample_loss_fn, (param_dict_gpu,), (tangent_dict,))
    beta_t = beta_t.cpu()  # [B]

    # ---- Step 5: HVP for delta_theta update ----
    # delta_theta_new = (1-lr*wd)*delta_theta + HVP
    # where HVP = d/d(theta) [<effective_dir, grad_loss(theta)>]

    # Recompute forward + backward with create_graph for HVP
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    output2 = model(input_ids=input_ids, attention_mask=attention_mask)
    per_sample2 = compute_per_sample_loss(output2.logits, labels, attention_mask)
    loss2 = per_sample2.sum()

    params_list = list(model.parameters())
    names_list = [n for n, _ in model.named_parameters()]
    grads = torch.autograd.grad(loss2, params_list, create_graph=True)

    # Dot product of grads with effective_dir
    dot_product = sum(
        (g * effective_dir[name].to(device)).sum()
        for g, name in zip(grads, names_list)
    )

    # Backward to get HVP
    hvp = torch.autograd.grad(dot_product, params_list)

    # ---- Step 6: Assemble new deltas ----
    new_delta_theta = {}
    new_delta_m = {}
    new_delta_v = {}

    for i, name in enumerate(names_list):
        # delta_theta: direct term + HVP
        new_delta_theta[name] = (
            (1 - lr * cfg.weight_decay) * delta_theta[name]
            + hvp[i].cpu()
        )

        # delta_m = beta1 * dA/dm
        new_delta_m[name] = cfg.beta1 * dA_dm[name]

        # delta_v = beta2 * dA/dv
        new_delta_v[name] = cfg.beta2 * dA_dv[name]

    return beta_t, new_delta_theta, new_delta_m, new_delta_v


def replay_forward_segment(
    model, start_ckpt, batch_indices_all, train_dataset, seg_start, seg_end, cfg, device
):
    """Replay training forward, returning all intermediate states."""
    states = []
    theta = start_ckpt["params"]
    m = start_ckpt["adam"]["m"]
    v = start_ckpt["adam"]["v"]
    states.append({"params": theta, "m": m, "v": v})

    for t in range(seg_start, seg_end):
        batch_idx = batch_indices_all[t]
        batch = train_dataset.collate(batch_idx)

        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(theta[name].to(device))

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model.train()
        model.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
        loss = per_sample.sum()
        loss.backward()

        grads = {n: p.grad.cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        lr = get_lr(t, len(batch_indices_all), cfg)

        new_theta, new_m, new_v = {}, {}, {}
        for name in theta:
            g = grads.get(name, torch.zeros_like(theta[name]))
            new_m[name] = cfg.beta1 * m[name] + (1 - cfg.beta1) * g
            new_v[name] = cfg.beta2 * v[name] + (1 - cfg.beta2) * g * g
            denom = torch.sqrt(new_v[name] + cfg.eps_root) + cfg.eps
            new_theta[name] = theta[name] - lr * new_m[name] / denom - lr * cfg.weight_decay * theta[name]

        theta, m, v = new_theta, new_m, new_v
        if t < seg_end - 1:
            states.append({"params": theta, "m": m, "v": v})

    return states


def compute_influence_for_test_sample(
    test_sample, train_dataset, cfg: MagicConfig, device="cuda:0"
):
    """Compute influence scores for a single test sample using optimized Replay."""
    ckpt_dir = Path(cfg.checkpoint_dir)
    batch_indices_all = torch.load(ckpt_dir / "batch_indices.pt", weights_only=False)
    total_steps = len(batch_indices_all)
    num_train = len(train_dataset)

    influence = torch.zeros(num_train)
    model = create_gpt2_model().to(device)

    # Load final model params
    ckpt_steps = sorted([int(f.stem.split("_")[1]) for f in ckpt_dir.glob("step_*.pt")])
    final_ckpt_step = ckpt_steps[-1]
    final_ckpt = load_checkpoint(ckpt_dir, final_ckpt_step)

    if final_ckpt_step < total_steps:
        states = replay_forward_segment(
            model, final_ckpt, batch_indices_all, train_dataset,
            final_ckpt_step, total_steps, cfg, device
        )
        final_params = states[-1]["params"]
    else:
        final_params = final_ckpt["params"]

    # Compute Delta_T (test loss gradient)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(final_params[name].to(device))

    model.eval()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    input_ids = test_sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = test_sample["attention_mask"].unsqueeze(0).to(device)
    labels = test_sample["labels"].unsqueeze(0).to(device)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
    test_loss = per_sample.sum()
    test_loss.backward()

    delta_theta = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
    delta_m = {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters()}
    delta_v = {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters()}
    base_test_loss = test_loss.item()

    # Iterate backward through segments
    segment_boundaries = []
    step = 0
    while step < total_steps:
        seg_end = min(step + cfg.checkpoint_every, total_steps)
        segment_boundaries.append((step, seg_end))
        step = seg_end

    num_segments = len(segment_boundaries)
    for seg_idx, (seg_start, seg_end) in enumerate(reversed(segment_boundaries)):
        ckpt = load_checkpoint(ckpt_dir, seg_start)
        seg_len = seg_end - seg_start
        states = replay_forward_segment(
            model, ckpt, batch_indices_all, train_dataset,
            seg_start, seg_end, cfg, device
        )

        for k in range(seg_len - 1, -1, -1):
            t = seg_start + k
            state = states[k]
            batch_idx = batch_indices_all[t]
            batch = train_dataset.collate(batch_idx)
            lr_t = get_lr(t, total_steps, cfg)

            beta_t, delta_theta, delta_m, delta_v = replay_step_fast(
                model, state["params"], state["m"], state["v"],
                batch, batch_idx,
                delta_theta, delta_m, delta_v,
                lr_t, cfg, device,
            )

            for i, idx in enumerate(batch_idx):
                influence[idx] += beta_t[i].item()

        del states

        if (seg_idx + 1) % 10 == 0:
            logger.info(f"  Processed {seg_idx + 1}/{num_segments} segments "
                       f"(step {seg_start}), influence range: [{influence.min():.6f}, {influence.max():.6f}]")

    del model
    torch.cuda.empty_cache()
    return influence, base_test_loss


def compute_all_influences(
    train_dataset, test_dataset, test_indices, cfg: MagicConfig, device="cuda:0"
):
    """Compute influence scores for all test samples (single GPU)."""
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
        logger.info(f"  Base loss: {base_loss:.4f}, influence: [{infl.min():.6f}, {infl.max():.6f}]")

    return influences, base_losses


def _influence_worker(gpu_id, test_indices_for_gpu, test_dataset, train_dataset, cfg, result_dict):
    """Worker for multi-GPU influence computation."""
    device = f"cuda:{gpu_id}"
    for global_j, test_idx in test_indices_for_gpu:
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
    """Compute influence scores across multiple GPUs."""
    import torch.multiprocessing as mp

    if not isinstance(train_dataset, TensorDataset):
        train_dataset = TensorDataset(train_dataset)
    if not isinstance(test_dataset, TensorDataset):
        test_dataset = TensorDataset(test_dataset)

    num_test = len(test_indices)
    num_train = len(train_dataset)

    assignments = [[] for _ in range(min(num_gpus, num_test))]
    for j, test_idx in enumerate(test_indices):
        assignments[j % len(assignments)].append((j, test_idx))

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
