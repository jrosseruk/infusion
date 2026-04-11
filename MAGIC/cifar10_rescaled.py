#!/usr/bin/env python
"""
MAGIC CIFAR-10 with adjoint rescaling.
At each VJP step, normalize Δ to prevent overflow, track cumulative scale.
Beta contributions are divided by the current scale to stay in a comparable range.

Quick FD validation first (50-step training), then full 600-step if it works.
"""
import sys, os, time, gc, math
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from torch.func import functional_call
from cifar10_magic import make_model, get_lr, load_cifar10
from cifar10_magic import LR, MOMENTUM, WEIGHT_DECAY, BATCH_SIZE, EPOCHS, SEED

DEVICE = 'cuda:0'


def train_model(tx, ty, bi, total, device):
    """Train and return checkpoints."""
    torch.manual_seed(SEED)
    model = make_model(device)
    mom = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    saved = {0: {'p': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
                 'm': {n: v.clone() for n, v in mom.items()}}}
    model.train()
    for step in range(total):
        x = tx[bi[step]].to(device); y = ty[bi[step]].to(device)
        lr_t = get_lr(step, total)
        logits = model(x); loss = F.cross_entropy(logits, y)
        model.zero_grad(); loss.backward()
        nm = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None: continue
                m = MOMENTUM * mom[n].to(device) + p.grad + WEIGHT_DECAY * p
                p.add_(m, alpha=-lr_t); nm[n] = m.cpu()
        mom = nm
        if (step + 1) % 5 == 0 or step == total - 1:
            saved[step + 1] = {'p': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
                                'm': {n: v.clone() for n, v in mom.items()}}
        if (step + 1) % 100 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  Step {step+1}/{total} lr={lr_t:.4f} loss={loss.item():.4f} acc={acc:.0%}", flush=True)
    return model, saved


def replay_seg(model, state, bi, tx, ty, start, end, total, device):
    """Replay forward, return intermediate states."""
    theta = {n: v.clone() for n, v in state['p'].items()}
    mom = {n: v.clone() for n, v in state['m'].items()}
    states = [{'p': theta, 'm': mom}]
    for t in range(start, end):
        with torch.no_grad():
            for n, p in model.named_parameters(): p.copy_(theta[n].to(device))
        model.train(); model.zero_grad()
        logits = model(tx[bi[t]].to(device))
        loss = F.cross_entropy(logits, ty[bi[t]].to(device)); loss.backward()
        lr_t = get_lr(t, total)
        nt = {}; nm = {}
        for n, p in model.named_parameters():
            if p.grad is None: continue
            m = MOMENTUM * mom[n] + p.grad.cpu() + WEIGHT_DECAY * theta[n]
            nt[n] = theta[n] - lr_t * m; nm[n] = m
        theta = nt; mom = nm; states.append({'p': theta, 'm': mom})
    return states


def vjp_step_rescaled(model, theta, mom, batch_idx, tx, ty, dt, dm, lr_t, device):
    """
    VJP with adjoint rescaling.
    Before computing, normalize dt/dm so ||dt||=1.
    Returns: beta (rescaled), new_dt (normalized), new_dm, scale_factor
    """
    # Compute current adjoint norm for rescaling
    dt_norm = sum(v.norm().item()**2 for v in dt.values())**0.5
    if dt_norm == 0: dt_norm = 1.0

    # Normalize adjoint
    dt_n = {n: v / dt_norm for n, v in dt.items()}
    dm_n = {n: v / dt_norm for n, v in dm.items()}

    # Standard VJP with normalized adjoint (fp32 - safe since ||dt||=1)
    pl = {n: theta[n].to(device).detach().requires_grad_(True) for n in theta}
    ml = {n: mom[n].to(device).detach().requires_grad_(True) for n in mom}
    B = len(batch_idx)
    w = torch.ones(B, device=device, requires_grad=True)
    logits = functional_call(model, pl, args=(tx[batch_idx].to(device),))
    per_loss = F.cross_entropy(logits, ty[batch_idx].to(device), reduction='none')
    wl = (w * per_loss).sum() / B
    plist = list(pl.values()); names = list(pl.keys())
    grads = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)
    A = torch.tensor(0.0, device=device)
    for i, n in enumerate(names):
        g = grads[i] if grads[i] is not None else torch.zeros_like(pl[n])
        m_new = MOMENTUM * ml[n] + g + WEIGHT_DECAY * pl[n]
        t_new = pl[n] - lr_t * m_new
        A = A + (t_new * dt_n[n].to(device).detach()).sum()
        A = A + (m_new * dm_n[n].to(device).detach()).sum()
    ag = torch.autograd.grad(A, plist + list(ml.values()) + [w], allow_unused=True)
    np_ = len(names)
    new_dt = {names[i]: ag[i].cpu().detach() if ag[i] is not None else dt_n[names[i]] for i in range(np_)}
    new_dm = {names[i]: ag[np_+i].cpu().detach() if ag[np_+i] is not None else dm_n[names[i]] for i in range(np_)}
    beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(B)

    # beta corresponds to normalized adjoint, so true beta = dt_norm * beta
    # But we DON'T undo the scaling - instead we accumulate rescaled betas
    # and track the scale separately
    del logits, per_loss, wl, grads, A, ag, pl, ml, w, plist
    return beta, new_dt, new_dm, dt_norm


def compute_influence_rescaled(model, saved, bi, tx, ty, ex, ey, test_idx, total, device):
    """Compute influence with adjoint rescaling. Returns influence vector."""
    N = len(tx)

    # Delta_T = test loss gradient
    with torch.no_grad():
        for n, p in model.named_parameters(): p.copy_(saved[total]['p'][n].to(device))
    model.eval(); model.zero_grad()
    for p in model.parameters(): p.requires_grad_(True)
    logits = model(ex[test_idx:test_idx+1].to(device))
    tl = F.cross_entropy(logits, ey[test_idx:test_idx+1].to(device))
    tl.backward()
    dt = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
    dm = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    base_loss = tl.item()
    del logits, tl

    # Accumulate influence with rescaling
    # Each beta_t is computed with a normalized adjoint (||dt||=1 before the step)
    # So beta_t is "per unit adjoint norm" influence
    # The true beta_t = cumulative_scale * beta_t_rescaled
    # For LDS ranking: we accumulate rescaled betas directly (ignoring scale)
    # This treats all training steps equally regardless of adjoint amplification
    influence = torch.zeros(N)

    seg_starts = sorted(saved.keys())
    log_scale = 0.0  # Track cumulative log scale for monitoring
    t_start = time.time()

    for si in range(len(seg_starts) - 1, 0, -1):
        ss = seg_starts[si - 1]; se = seg_starts[si]
        states = replay_seg(model, saved[ss], bi, tx, ty, ss, se, total, device)
        for k in range(len(states) - 2, -1, -1):
            t = ss + k; lr_t = get_lr(t, total)
            beta, dt, dm, scale = vjp_step_rescaled(
                model, states[k]['p'], states[k]['m'], bi[t], tx, ty, dt, dm, lr_t, device)
            log_scale += math.log(max(scale, 1e-30))
            for i, idx in enumerate(bi[t]):
                influence[idx] += beta[i].item()
        del states; gc.collect(); torch.cuda.empty_cache()

        steps_done = total - seg_starts[si]
        if steps_done > 0 and steps_done % 100 == 0:
            dt_norm = sum(v.norm().item()**2 for v in dt.values())**0.5
            print(f"  {steps_done}/{total} steps, dt_norm={dt_norm:.4f} (should be ~1), "
                  f"log_scale={log_scale:.1f}, inf=[{influence.min():.4f},{influence.max():.4f}]", flush=True)

    elapsed = time.time() - t_start
    dt_norm = sum(v.norm().item()**2 for v in dt.values())**0.5
    print(f"  Done in {elapsed:.0f}s, dt_norm={dt_norm:.4f}, log_scale={log_scale:.1f}", flush=True)
    return influence, base_loss


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("Loading CIFAR-10...", flush=True)
    tx, ty, ex, ey = load_cifar10()
    N = len(tx)

    # ============================================================
    # PHASE 1: Quick FD validation (50 steps)
    # ============================================================
    print("\n=== PHASE 1: Quick FD validation (50 steps) ===", flush=True)
    QUICK_STEPS = 50
    QUICK_BS = 1000
    torch.manual_seed(SEED)
    bi_quick = []
    perm = torch.randperm(N).tolist()
    for i in range(0, N, QUICK_BS):
        b = perm[i:i+QUICK_BS]
        if len(b) == QUICK_BS: bi_quick.append(b)
    bi_quick = bi_quick[:QUICK_STEPS]

    print(f"Training {QUICK_STEPS} steps...", flush=True)
    model, saved = train_model(tx, ty, bi_quick, QUICK_STEPS, DEVICE)

    print("Computing rescaled influence (test 0)...", flush=True)
    infl, bl = compute_influence_rescaled(model, saved, bi_quick, tx, ty, ex, ey, 0, QUICK_STEPS, DEVICE)
    print(f"Base loss: {bl:.4f}, Influence: [{infl.min():.6f}, {infl.max():.6f}]", flush=True)

    # FD comparison
    print("\nFD validation:", flush=True)
    eps = 1e-3
    for si in [0, 100, 500]:
        steps_with = [t for t in range(QUICK_STEPS) if si in bi_quick[t]]
        if not steps_with:
            print(f"  Sample {si}: not in any batch", flush=True)
            continue

        def train_eval_fd(perturb_idx, eps_val):
            torch.manual_seed(SEED)
            m_fd = make_model(DEVICE)
            mom_fd = {n: torch.zeros_like(p, device='cpu') for n, p in m_fd.named_parameters()}
            sw = torch.ones(N); sw[perturb_idx] += eps_val
            m_fd.train()
            for step in range(QUICK_STEPS):
                x = tx[bi_quick[step]].to(DEVICE); y = ty[bi_quick[step]].to(DEVICE)
                wb = sw[bi_quick[step]].to(DEVICE)
                lr_t = get_lr(step, QUICK_STEPS)
                logits = m_fd(x)
                per_loss = F.cross_entropy(logits, y, reduction='none')
                loss = (wb * per_loss).sum() / QUICK_BS
                m_fd.zero_grad(); loss.backward()
                nm = {}
                with torch.no_grad():
                    for n, p in m_fd.named_parameters():
                        if p.grad is None: continue
                        m = MOMENTUM * mom_fd[n].to(DEVICE) + p.grad + WEIGHT_DECAY * p
                        p.add_(m, alpha=-lr_t); nm[n] = m.cpu()
                mom_fd = nm
            m_fd.eval()
            with torch.no_grad():
                logits = m_fd(ex[0:1].to(DEVICE))
                loss_val = F.cross_entropy(logits, ey[0:1].to(DEVICE)).item()
            del m_fd; torch.cuda.empty_cache()
            return loss_val

        l0 = train_eval_fd(si, 0.0)
        l1 = train_eval_fd(si, eps)
        fd = (l1 - l0) / eps
        vjp_val = infl[si].item()
        ratio = vjp_val / fd if abs(fd) > 1e-10 else float('inf')
        print(f"  Sample {si}: FD={fd:.6f} VJP={vjp_val:.6f} ratio={ratio:.4f}", flush=True)

    # ============================================================
    # PHASE 2: Full 600-step experiment (only if phase 1 looks OK)
    # ============================================================
    print("\n=== PHASE 2: Full 600-step experiment ===", flush=True)
    spe = N // BATCH_SIZE; total = spe * EPOCHS
    torch.manual_seed(SEED)
    bi = []
    for e in range(EPOCHS):
        perm = torch.randperm(N).tolist()
        for i in range(0, N, BATCH_SIZE):
            b = perm[i:i+BATCH_SIZE]
            if len(b) == BATCH_SIZE: bi.append(b)
    bi = bi[:total]

    print(f"Training {total} steps...", flush=True)
    model, saved = train_model(tx, ty, bi, total, DEVICE)
    model.eval()
    with torch.no_grad():
        tacc = (model(ex[:1000].to(DEVICE)).argmax(1) == ey[:1000].to(DEVICE)).float().mean().item()
    print(f"Test accuracy: {tacc:.0%}", flush=True)

    # Compute influence for 10 test samples
    NUM_TEST = 10
    influences = torch.zeros(NUM_TEST, N)
    base_losses = torch.zeros(NUM_TEST)
    for j in range(NUM_TEST):
        print(f"\nTest {j}:", flush=True)
        infl, bl = compute_influence_rescaled(model, saved, bi, tx, ty, ex, ey, j, total, DEVICE)
        influences[j] = infl; base_losses[j] = bl

    # Counterfactual (50 subsets, drop 5%)
    NUM_CF = 50; DROP = 0.05
    nd = int(N * DROP); rng = np.random.RandomState(SEED + 1000)
    masks = np.ones((NUM_CF, N)); true_losses = np.zeros((NUM_CF, NUM_TEST))

    print(f"\nCounterfactual ({NUM_CF} subsets, drop {int(DROP*100)}%)...", flush=True)
    for s in range(NUM_CF):
        pc = rng.permutation(N); masks[s, pc[:nd]] = 0
        sw = torch.ones(N); sw[pc[:nd]] = 0
        torch.manual_seed(SEED)
        m2 = make_model(DEVICE)
        mom2 = {n: torch.zeros_like(p, device='cpu') for n, p in m2.named_parameters()}
        m2.train()
        for step in range(total):
            x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
            wb = sw[bi[step]].to(DEVICE); lr_t = get_lr(step, total)
            logits = m2(x); per_loss = F.cross_entropy(logits, y, reduction='none')
            loss = (wb * per_loss).sum() / BATCH_SIZE
            m2.zero_grad(); loss.backward()
            nm2 = {}
            with torch.no_grad():
                for n, p in m2.named_parameters():
                    if p.grad is None: continue
                    m = MOMENTUM * mom2[n].to(DEVICE) + p.grad + WEIGHT_DECAY * p
                    p.add_(m, alpha=-lr_t); nm2[n] = m.cpu()
            mom2 = nm2
        m2.eval()
        with torch.no_grad():
            for j in range(NUM_TEST):
                logits = m2(ex[j:j+1].to(DEVICE))
                true_losses[s, j] = F.cross_entropy(logits, ey[j:j+1].to(DEVICE)).item()
        del m2; torch.cuda.empty_cache()
        if (s + 1) % 10 == 0: print(f"  {s+1}/{NUM_CF}", flush=True)

    # LDS
    print("\n" + "=" * 60, flush=True)
    print(f"MAGIC CIFAR-10 LDS (rescaled adjoint, drop {int(DROP*100)}%)", flush=True)
    print("=" * 60, flush=True)
    dw = masks - 1; all_r = []
    for j in range(NUM_TEST):
        pred = base_losses[j].item() + dw @ influences[j].numpy()
        r, _ = spearmanr(pred, true_losses[:, j]); all_r.append(r)
        print(f"  Test {j}: LDS={r:.4f}", flush=True)
    mean_lds = np.nanmean(all_r)
    print(f"\nMean LDS: {mean_lds:.4f} (paper: 0.922)", flush=True)
    print("=" * 60, flush=True)

    # Plot
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    OUT = '/home/mac/infusion/MAGIC/gpt2_lds/output'
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(['MAGIC\n(rescaled)', 'MAGIC\n(paper)', 'EKFAC\n(paper)', 'TRAK\n(paper)'],
           [mean_lds, 0.922, 0.249, 0.362],
           color=['#ef8632', '#ef8632', '#4a7cb6', '#8e529f'],
           alpha=[1.0, 0.4, 0.8, 0.8])
    ax.set_ylabel('LDS'); ax.set_title(f'CIFAR-10 ResNet-9 LDS (drop {int(DROP*100)}%)')
    ax.set_ylim(-0.2, 1.1); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUT}/cifar10_rescaled_lds.png', dpi=150)
    print(f"Plot saved!", flush=True)


if __name__ == '__main__':
    main()
