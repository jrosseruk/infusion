"""
Replay v2: Optimized for speed.

Key insight: avoid torch.func.jvp/functional_call which are slow for GPT-2.
Instead:
- Compute effective_dir analytically (no autograd through Adam)
- Get per-sample beta_t via B individual backward passes (sharing forward activations)
- Get delta_theta via HVP (one create_graph backward + one backward)
- Get delta_m, delta_v analytically
"""
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import MagicConfig
from .data import create_gpt2_model, TensorDataset
from .train import get_lr, compute_per_sample_loss, load_checkpoint

logger = logging.getLogger(__name__)


def replay_step_v2(
    model, theta_dict, m_dict, v_dict,
    batch_data, batch_indices,
    delta_theta, delta_m, delta_v,
    lr, cfg, device,
):
    """
    Fast Replay VJP step.

    1. Forward pass (shared for beta + HVP)
    2. Per-sample backwards for beta_t (B backwards, ~10ms each)
    3. One create_graph backward + backward for HVP (delta_theta)
    4. Analytical delta_m, delta_v
    """
    # Load params into model
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(theta_dict[name].to(device))

    input_ids = batch_data["input_ids"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    labels = batch_data["labels"].to(device)
    B = input_ids.shape[0]

    # ---- Step 1: Forward pass + compute g_t and Adam update analytically ----
    model.train()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    per_sample_loss = compute_per_sample_loss(output.logits, labels, attention_mask)
    total_loss = per_sample_loss.sum()

    # Standard backward to get g_t
    total_loss.backward(retain_graph=True)
    g_t = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}

    # Adam update (analytical, CPU)
    m_new = {}
    v_new = {}
    for name in theta_dict:
        g = g_t[name]
        m_new[name] = cfg.beta1 * m_dict[name] + (1 - cfg.beta1) * g
        v_new[name] = cfg.beta2 * v_dict[name] + (1 - cfg.beta2) * g * g

    # ---- Step 2: Compute effective direction dA/dg analytically ----
    effective_dir = {}
    dA_dm_dict = {}
    dA_dv_dict = {}

    for name in theta_dict:
        D = torch.sqrt(v_new[name] + cfg.eps_root) + cfg.eps
        sqrt_v = D - cfg.eps

        dA_dm = delta_m[name] - lr / D * delta_theta[name]
        dA_dv = delta_v[name] + lr * m_new[name] * delta_theta[name] / (2.0 * D * D * sqrt_v)

        g = g_t[name]
        eff = (1 - cfg.beta1) * dA_dm + 2 * (1 - cfg.beta2) * g * dA_dv

        effective_dir[name] = eff
        dA_dm_dict[name] = dA_dm
        dA_dv_dict[name] = dA_dv

    # ---- Step 3: Per-sample beta_t via individual backwards ----
    # beta_t[i] = <effective_dir, grad_loss_i>
    # Use per_sample_loss[i].backward(retain_graph=True) to get per-sample grads

    beta_t = torch.zeros(B)
    for i in range(B):
        model.zero_grad()
        per_sample_loss[i].backward(retain_graph=(i < B - 1))
        # Dot product with effective_dir
        dot = 0.0
        for n, p in model.named_parameters():
            if p.grad is not None:
                dot += (p.grad.cpu() * effective_dir[n]).sum().item()
        beta_t[i] = dot

    # ---- Step 4: HVP for delta_theta ----
    # delta_theta_new = (1-lr*wd)*delta_theta + HVP
    # HVP = d/d(theta) [<effective_dir, g(theta)>]
    # where g(theta) = grad(total_loss, theta)

    # Need to recompute with create_graph
    model.zero_grad()
    output2 = model(input_ids=input_ids, attention_mask=attention_mask)
    per_sample2 = compute_per_sample_loss(output2.logits, labels, attention_mask)
    loss2 = per_sample2.sum()

    params_list = list(model.parameters())
    names_list = [n for n, _ in model.named_parameters()]
    grads = torch.autograd.grad(loss2, params_list, create_graph=True)

    # Dot product with effective_dir
    dot_product = sum(
        (g * effective_dir[name].to(device)).sum()
        for g, name in zip(grads, names_list)
    )

    # Backward for HVP
    hvp = torch.autograd.grad(dot_product, params_list)

    # ---- Step 5: Assemble new deltas ----
    new_delta_theta = {}
    new_delta_m = {}
    new_delta_v = {}

    for i, name in enumerate(names_list):
        new_delta_theta[name] = (1 - lr * cfg.weight_decay) * delta_theta[name] + hvp[i].cpu()
        new_delta_m[name] = cfg.beta1 * dA_dm_dict[name]
        new_delta_v[name] = cfg.beta2 * dA_dv_dict[name]

    return beta_t, new_delta_theta, new_delta_m, new_delta_v


def replay_forward_segment(model, start_ckpt, batch_indices_all, train_dataset, seg_start, seg_end, cfg, device):
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
        lr_t = get_lr(t, len(batch_indices_all), cfg)

        new_theta, new_m, new_v = {}, {}, {}
        for name in theta:
            g = grads.get(name, torch.zeros_like(theta[name]))
            new_m[name] = cfg.beta1 * m[name] + (1 - cfg.beta1) * g
            new_v[name] = cfg.beta2 * v[name] + (1 - cfg.beta2) * g * g
            denom = torch.sqrt(new_v[name] + cfg.eps_root) + cfg.eps
            new_theta[name] = theta[name] - lr_t * new_m[name] / denom - lr_t * cfg.weight_decay * theta[name]

        theta, m, v = new_theta, new_m, new_v
        if t < seg_end - 1:
            states.append({"params": theta, "m": m, "v": v})

    return states


def compute_influence_for_test_sample(test_sample, train_dataset, cfg, device="cuda:0"):
    """Compute influence scores for a single test sample."""
    ckpt_dir = Path(cfg.checkpoint_dir)
    batch_indices_all = torch.load(ckpt_dir / "batch_indices.pt", weights_only=False)
    total_steps = len(batch_indices_all)
    num_train = len(train_dataset)

    influence = torch.zeros(num_train)
    model = create_gpt2_model().to(device)

    # Load final model
    ckpt_steps = sorted([int(f.stem.split("_")[1]) for f in ckpt_dir.glob("step_*.pt")])
    final_ckpt_step = ckpt_steps[-1]
    final_ckpt = load_checkpoint(ckpt_dir, final_ckpt_step)

    if final_ckpt_step < total_steps:
        states = replay_forward_segment(model, final_ckpt, batch_indices_all, train_dataset,
                                         final_ckpt_step, total_steps, cfg, device)
        final_params = states[-1]["params"]
    else:
        final_params = final_ckpt["params"]

    # Compute Delta_T
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
    t_start = time.time()

    for seg_idx, (seg_start, seg_end) in enumerate(reversed(segment_boundaries)):
        ckpt = load_checkpoint(ckpt_dir, seg_start)
        seg_len = seg_end - seg_start
        states = replay_forward_segment(model, ckpt, batch_indices_all, train_dataset,
                                         seg_start, seg_end, cfg, device)

        for k in range(seg_len - 1, -1, -1):
            t = seg_start + k
            state = states[k]
            batch_idx = batch_indices_all[t]
            batch = train_dataset.collate(batch_idx)
            lr_t = get_lr(t, total_steps, cfg)

            beta_t, delta_theta, delta_m, delta_v = replay_step_v2(
                model, state["params"], state["m"], state["v"],
                batch, batch_idx,
                delta_theta, delta_m, delta_v,
                lr_t, cfg, device,
            )

            for i, idx in enumerate(batch_idx):
                influence[idx] += beta_t[i].item()

        del states

        if (seg_idx + 1) % 20 == 0:
            elapsed = time.time() - t_start
            steps_done = (seg_idx + 1) * cfg.checkpoint_every
            rate = elapsed / steps_done if steps_done > 0 else 0
            eta = rate * (total_steps - steps_done)
            print(f"  Segment {seg_idx+1}/{num_segments}, "
                  f"{steps_done}/{total_steps} steps, "
                  f"{rate:.2f}s/step, ETA {eta/60:.1f}min", flush=True)

    del model
    torch.cuda.empty_cache()
    return influence, base_test_loss
