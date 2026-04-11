#!/usr/bin/env python
"""CIFAR-10 MAGIC LDS with truncated Replay (last 200 steps only)."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn.functional as F
import numpy as np
from torch.func import functional_call
from scipy.stats import spearmanr
from cifar10_lds import MetasmoothResNet9, get_one_cycle_lr, load_cifar10

DEVICE = 'cuda:0'
MAX_LR = 0.5; MOM = 0.85; WD = 1e-5; BS = 250; EPOCHS = 18
TRUNCATE = 200  # Only backprop last 200 steps
NUM_TEST = 10; NUM_CF = 50; DROP = 0.05

torch.manual_seed(42); np.random.seed(42)

print("Loading CIFAR-10...", flush=True)
tx, ty, tex, tey = load_cifar10()
N = len(tx); spe = N // BS; total = spe * EPOCHS
print(f"N={N}, total_steps={total}", flush=True)

# Batch indices
torch.manual_seed(42)
bi = []
for e in range(EPOCHS):
    perm = torch.randperm(N).tolist()
    for i in range(0, N, BS):
        b = perm[i:i+BS]
        if len(b) == BS:
            bi.append(b)
bi = bi[:total]

# Train
print(f"Training (SGD lr={MAX_LR}, {total} steps)...", flush=True)
model = MetasmoothResNet9(width_mult=2.0, final_scale=0.125).to(DEVICE)
mom_buf = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}

# Only save states for the last TRUNCATE+20 steps
SAVE_FROM = total - TRUNCATE - 20
saved = {}

model.train()
for step in range(total):
    x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
    lr = get_one_cycle_lr(step, total, MAX_LR)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(); loss.backward()
    nm = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            m = MOM * mom_buf[n].to(DEVICE) + p.grad + WD * p
            p.add_(m, alpha=-lr)
            nm[n] = m.cpu()
    mom_buf = nm
    if step >= SAVE_FROM and step % 5 == 0:
        saved[step] = {
            'params': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
            'momentum': {n: v.clone() for n, v in mom_buf.items()},
        }
    if (step + 1) % 500 == 0:
        acc = (logits.argmax(1) == y).float().mean().item()
        print(f"  Step {step+1}/{total} lr={lr:.4f} loss={loss.item():.4f} acc={acc:.0%}", flush=True)

saved[total] = {
    'params': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
    'momentum': {n: v.clone() for n, v in mom_buf.items()},
}

model.eval()
with torch.no_grad():
    tacc = (model(tex[:1000].to(DEVICE)).argmax(1) == tey[:1000].to(DEVICE)).float().mean().item()
print(f"Test acc: {tacc:.0%}, {len(saved)} checkpoints", flush=True)

# Replay forward within segment
def replay_seg(model, state, start, end):
    theta = {n: v.clone() for n, v in state['params'].items()}
    mom = {n: v.clone() for n, v in state['momentum'].items()}
    states = [{'params': theta, 'momentum': mom}]
    for t in range(start, end):
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(theta[n].to(DEVICE))
        model.train(); model.zero_grad()
        logits = model(tx[bi[t]].to(DEVICE))
        loss = F.cross_entropy(logits, ty[bi[t]].to(DEVICE))
        loss.backward()
        lr_t = get_one_cycle_lr(t, total, MAX_LR)
        nt = {}; nm = {}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            m = MOM * mom[n] + p.grad.cpu() + WD * theta[n]
            nt[n] = theta[n] - lr_t * m
            nm[n] = m
        theta = nt; mom = nm
        states.append({'params': theta, 'momentum': mom})
    return states

# VJP step
def vjp_step(model, theta, mom, batch_idx, dt, dm, lr_t):
    pl = {n: theta[n].to(DEVICE).detach().requires_grad_(True) for n in theta}
    ml = {n: mom[n].to(DEVICE).detach().requires_grad_(True) for n in mom}
    w = torch.ones(len(batch_idx), device=DEVICE, requires_grad=True)
    logits = functional_call(model, pl, args=(tx[batch_idx].to(DEVICE),))
    per_loss = F.cross_entropy(logits, ty[batch_idx].to(DEVICE), reduction='none')
    wl = (w * per_loss).sum()
    plist = list(pl.values()); names = list(pl.keys())
    grads = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)

    A = torch.tensor(0.0, device=DEVICE)
    for i, n in enumerate(names):
        g = grads[i] if grads[i] is not None else torch.zeros_like(pl[n])
        m_new = MOM * ml[n] + g + WD * pl[n]
        t_new = pl[n] - lr_t * m_new
        A = A + (t_new * dt[n].to(DEVICE).detach()).sum()
        A = A + (m_new * dm[n].to(DEVICE).detach()).sum()

    ag = torch.autograd.grad(A, plist + list(ml.values()) + [w], allow_unused=True)
    np_ = len(names)
    ndt = {names[i]: ag[i].cpu().detach() if ag[i] is not None else dt[names[i]] for i in range(np_)}
    ndm = {names[i]: ag[np_+i].cpu().detach() if ag[np_+i] is not None else dm[names[i]] for i in range(np_)}
    beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(len(batch_idx))
    del logits, per_loss, wl, grads, A, ag, pl, ml, w, plist
    return beta, ndt, ndm

# Influence (truncated)
START = total - TRUNCATE
print(f"\nInfluence (steps {START}-{total}, {TRUNCATE} VJP steps per test)...", flush=True)
influences = torch.zeros(NUM_TEST, N, dtype=torch.float64)
base_losses = torch.zeros(NUM_TEST)

for j in range(NUM_TEST):
    t0 = time.time()
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(saved[total]['params'][n].to(DEVICE))
    model.eval(); model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)
    logits = model(tex[j:j+1].to(DEVICE))
    tl = F.cross_entropy(logits, tey[j:j+1].to(DEVICE))
    tl.backward()
    dt = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
    dm = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    base_losses[j] = tl.item()
    del logits, tl

    seg_starts = sorted([k for k in saved.keys() if k >= START])
    for si in range(len(seg_starts) - 1, 0, -1):
        ss = seg_starts[si - 1]; se = seg_starts[si]
        states = replay_seg(model, saved[ss], ss, se)
        for k in range(len(states) - 2, -1, -1):
            t = ss + k
            lr_t = get_one_cycle_lr(t, total, MAX_LR)
            beta, dt, dm = vjp_step(model, states[k]['params'], states[k]['momentum'],
                                     bi[t], dt, dm, lr_t)
            for i, idx in enumerate(bi[t]):
                influences[j, idx] += beta[i].double().item()
        del states; gc.collect(); torch.cuda.empty_cache()

    dn = sum(v.norm().item()**2 for v in dt.values())**0.5
    print(f"  Test {j}: {time.time()-t0:.0f}s loss={base_losses[j]:.4f} "
          f"inf=[{influences[j].min():.4f},{influences[j].max():.4f}] dt={dn:.2e}", flush=True)

# Counterfactual
print(f"\nCounterfactual ({NUM_CF} subsets, drop {int(DROP*100)}%)...", flush=True)
nd = int(N * DROP); rng = np.random.RandomState(42)
masks = np.ones((NUM_CF, N)); true_l = np.zeros((NUM_CF, NUM_TEST))

for s in range(NUM_CF):
    pc = rng.permutation(N); masks[s, pc[:nd]] = 0
    sw = torch.ones(N); sw[pc[:nd]] = 0
    torch.manual_seed(42)
    m2 = MetasmoothResNet9(width_mult=2.0, final_scale=0.125).to(DEVICE)
    mom2 = {n: torch.zeros_like(p, device='cpu') for n, p in m2.named_parameters()}
    m2.train()
    for step in range(total):
        x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
        wb = sw[bi[step]].to(DEVICE)
        lr_t = get_one_cycle_lr(step, total, MAX_LR)
        logits = m2(x)
        per_loss = F.cross_entropy(logits, y, reduction='none')
        loss = (wb * per_loss).sum() / wb.sum().clamp(min=1)
        m2.zero_grad(); loss.backward()
        nm2 = {}
        with torch.no_grad():
            for n, p in m2.named_parameters():
                if p.grad is None: continue
                m = MOM * mom2[n].to(DEVICE) + p.grad + WD * p
                p.add_(m, alpha=-lr_t)
                nm2[n] = m.cpu()
        mom2 = nm2
    m2.eval()
    with torch.no_grad():
        for j2 in range(NUM_TEST):
            logits = m2(tex[j2:j2+1].to(DEVICE))
            true_l[s, j2] = F.cross_entropy(logits, tey[j2:j2+1].to(DEVICE)).item()
    del m2; torch.cuda.empty_cache()
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{NUM_CF}", flush=True)

# LDS
print("\n" + "=" * 60, flush=True)
print(f"MAGIC CIFAR-10 LDS (truncated {TRUNCATE} steps, drop {int(DROP*100)}%)", flush=True)
print("=" * 60, flush=True)
dw = masks - 1
all_r = []
for j in range(NUM_TEST):
    pred = base_losses[j].item() + dw @ influences[j].numpy()
    r, _ = spearmanr(pred, true_l[:, j])
    all_r.append(r)
    print(f"  Test {j}: LDS={r:.4f}", flush=True)
mean_lds = np.nanmean(all_r)
print(f"\nMean LDS: {mean_lds:.4f} (paper: 0.922)", flush=True)
print("=" * 60, flush=True)

# Plot
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
ax.bar(['MAGIC\n(ours)', 'Paper'], [mean_lds, 0.922], color=['#ef8632', 'gray'], alpha=0.8)
ax.set_ylabel('LDS'); ax.set_title(f'CIFAR-10 ResNet-9 LDS (drop 5%)'); ax.set_ylim(-0.2, 1.1)
ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('/home/mac/infusion/MAGIC/gpt2_lds/output/cifar10_lds.png', dpi=150)
print("Plot saved!", flush=True)
