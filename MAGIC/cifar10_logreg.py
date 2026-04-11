#!/usr/bin/env python
"""MAGIC on CIFAR-10 logistic regression (convex, guaranteed stable adjoint)."""
import sys, os, time, gc, math
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

DEVICE = 'cuda:0'
SEED = 42
BS = 500; LR = 0.1; STEPS = 200; WD = 0.01
NUM_TEST = 5; NUM_CF = 50; DROP = 0.05

torch.manual_seed(SEED); np.random.seed(SEED)

# Load CIFAR-10 flattened
print("Loading CIFAR-10...", flush=True)
import torchvision, torchvision.transforms as T
tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', True, download=False, transform=tf)
te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', False, download=False, transform=tf)
tx = torch.stack([tr[i][0] for i in range(len(tr))]).view(len(tr), -1)
ty = torch.tensor([tr[i][1] for i in range(len(tr))])
ex = torch.stack([te[i][0] for i in range(len(te))]).view(len(te), -1)
ey = torch.tensor([te[i][1] for i in range(len(te))])
N, D = tx.shape
print(f"N={N}, D={D}", flush=True)

# Batch indices (cycle through data)
bi = []
for i in range(0, N, BS):
    b = list(range(i, min(i + BS, N)))
    if len(b) == BS:
        bi.append(b)
bi = (bi * 10)[:STEPS]

# Train logistic regression
print(f"Training logistic regression (lr={LR}, {STEPS} steps)...", flush=True)
torch.manual_seed(SEED)
W = torch.randn(D, 10, device=DEVICE) * 0.01
W.requires_grad_(True)
b = torch.zeros(10, device=DEVICE, requires_grad=True)

all_W = [W.data.cpu().clone()]
all_b = [b.data.cpu().clone()]

for step in range(STEPS):
    x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
    logits = x @ W + b
    loss = F.cross_entropy(logits, y)
    loss.backward()
    with torch.no_grad():
        W -= LR * (W.grad + WD * W)
        b -= LR * b.grad
        W.grad.zero_(); b.grad.zero_()
    all_W.append(W.data.cpu().clone())
    all_b.append(b.data.cpu().clone())
    if (step + 1) % 50 == 0:
        acc = (logits.argmax(1) == y).float().mean().item()
        print(f"  Step {step+1}: loss={loss.item():.4f} acc={acc:.0%}", flush=True)

with torch.no_grad():
    tacc = ((ex[:1000].to(DEVICE) @ W + b).argmax(1) == ey[:1000].to(DEVICE)).float().mean().item()
print(f"Test acc: {tacc:.0%}", flush=True)

# Hessian eigenvalue
print("Hessian...", flush=True)
v_W = torch.randn_like(W); v_b = torch.randn_like(b)
vn = (v_W.norm()**2 + v_b.norm()**2)**0.5
v_W /= vn; v_b /= vn
for it in range(20):
    W.grad = None; b.grad = None
    logits = tx[:BS].to(DEVICE) @ W + b
    loss = F.cross_entropy(logits, ty[:BS].to(DEVICE))
    gW, gb = torch.autograd.grad(loss, [W, b], create_graph=True)
    dot = (gW * v_W).sum() + (gb * v_b).sum()
    hW, hb = torch.autograd.grad(dot, [W, b])
    lam = (hW * v_W).sum().item() + (hb * v_b).sum().item()
    hn = (hW.norm()**2 + hb.norm()**2)**0.5
    v_W = hW.detach() / hn; v_b = hb.detach() / hn
print(f"lambda_max={lam:.4f}, LR*lambda={LR*lam:.4f}", flush=True)

# Replay influence
print(f"\nReplay influence ({NUM_TEST} test, {STEPS} steps)...", flush=True)
influences = torch.zeros(NUM_TEST, N)
base_losses = torch.zeros(NUM_TEST)

for j in range(NUM_TEST):
    t0 = time.time()
    Wf = all_W[-1].to(DEVICE).detach().requires_grad_(True)
    bf = all_b[-1].to(DEVICE).detach().requires_grad_(True)
    logits = ex[j:j+1].to(DEVICE) @ Wf + bf
    tl = F.cross_entropy(logits, ey[j:j+1].to(DEVICE))
    tl.backward()
    dW = Wf.grad.cpu().clone(); db = bf.grad.cpu().clone()
    base_losses[j] = tl.item()
    del logits, tl

    for t in range(STEPS - 1, -1, -1):
        Wt = all_W[t].to(DEVICE).detach().requires_grad_(True)
        bt = all_b[t].to(DEVICE).detach().requires_grad_(True)
        w = torch.ones(BS, device=DEVICE, requires_grad=True)
        logits = tx[bi[t]].to(DEVICE) @ Wt + bt
        per_loss = F.cross_entropy(logits, ty[bi[t]].to(DEVICE), reduction='none')
        wl = (w * per_loss).sum() / BS
        gW, gb = torch.autograd.grad(wl, [Wt, bt], create_graph=True)
        Wn = Wt - LR * (gW + WD * Wt)
        bn = bt - LR * gb
        A = (Wn * dW.to(DEVICE).detach()).sum() + (bn * db.to(DEVICE).detach()).sum()
        ag = torch.autograd.grad(A, [Wt, bt, w])
        dW = ag[0].cpu().detach()
        db = ag[1].cpu().detach()
        beta = ag[2].cpu().detach()
        for i, idx in enumerate(bi[t]):
            influences[j, idx] += beta[i].item()
        del logits, per_loss, wl, gW, gb, A, ag, Wt, bt, w, Wn, bn

    dn = (dW.norm()**2 + db.norm()**2)**0.5
    print(f"  Test {j}: {time.time()-t0:.0f}s loss={base_losses[j]:.4f} "
          f"inf=[{influences[j].min():.6f},{influences[j].max():.6f}] dt={dn:.2e}", flush=True)

# FD check
print("\nFD validation:", flush=True)
eps = 1e-3
for si in [0, 500, 5000]:
    if not any(si in bi[t] for t in range(STEPS)):
        continue
    def train_fd(pidx, ev):
        torch.manual_seed(SEED)
        Wf = torch.randn(D, 10, device=DEVICE) * 0.01; Wf.requires_grad_(True)
        bf = torch.zeros(10, device=DEVICE, requires_grad=True)
        sw = torch.ones(N); sw[pidx] += ev
        for step in range(STEPS):
            x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
            wb = sw[bi[step]].to(DEVICE)
            logits = x @ Wf + bf
            per_loss = F.cross_entropy(logits, y, reduction='none')
            loss = (wb * per_loss).sum() / BS
            loss.backward()
            with torch.no_grad():
                Wf -= LR * (Wf.grad + WD * Wf)
                bf -= LR * bf.grad
                Wf.grad.zero_(); bf.grad.zero_()
        with torch.no_grad():
            return F.cross_entropy(ex[0:1].to(DEVICE) @ Wf + bf, ey[0:1].to(DEVICE)).item()
    l0 = train_fd(si, 0.0); l1 = train_fd(si, eps)
    fd = (l1 - l0) / eps
    vjp = influences[0, si].item()
    ratio = vjp / fd if abs(fd) > 1e-10 else float('inf')
    print(f"  Sample {si}: FD={fd:.6f} VJP={vjp:.6f} ratio={ratio:.4f}", flush=True)

# LDS
print(f"\nCounterfactual ({NUM_CF} subsets)...", flush=True)
nd = int(N * DROP); rng = np.random.RandomState(SEED + 1000)
masks = np.ones((NUM_CF, N)); true_l = np.zeros((NUM_CF, NUM_TEST))

for s in range(NUM_CF):
    pc = rng.permutation(N); masks[s, pc[:nd]] = 0
    sw = torch.ones(N); sw[pc[:nd]] = 0
    torch.manual_seed(SEED)
    Wf = torch.randn(D, 10, device=DEVICE) * 0.01; Wf.requires_grad_(True)
    bf = torch.zeros(10, device=DEVICE, requires_grad=True)
    for step in range(STEPS):
        x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
        wb = sw[bi[step]].to(DEVICE)
        logits = x @ Wf + bf
        per_loss = F.cross_entropy(logits, y, reduction='none')
        loss = (wb * per_loss).sum() / BS
        loss.backward()
        with torch.no_grad():
            Wf -= LR * (Wf.grad + WD * Wf)
            bf -= LR * bf.grad
            Wf.grad.zero_(); bf.grad.zero_()
    with torch.no_grad():
        for j2 in range(NUM_TEST):
            true_l[s, j2] = F.cross_entropy(ex[j2:j2+1].to(DEVICE) @ Wf + bf, ey[j2:j2+1].to(DEVICE)).item()
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{NUM_CF}", flush=True)

print("\n" + "=" * 60, flush=True)
print("MAGIC CIFAR-10 LDS (Logistic Regression, drop 5%)", flush=True)
print("=" * 60, flush=True)
dw = masks - 1
all_r = []
for j in range(NUM_TEST):
    pred = base_losses[j].item() + dw @ influences[j].numpy()
    r, _ = spearmanr(pred, true_l[:, j])
    all_r.append(r)
    print(f"  Test {j}: LDS={r:.4f}", flush=True)
mean_lds = np.nanmean(all_r)
print(f"\nMean LDS: {mean_lds:.4f}", flush=True)
print("=" * 60, flush=True)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
ax.bar(['MAGIC\n(LogReg)', 'MAGIC\n(paper)', 'EKFAC\n(paper)'],
       [mean_lds, 0.922, 0.249], color=['#ef8632', 'gray', '#4a7cb6'], alpha=0.8)
ax.set_ylabel('LDS'); ax.set_title('CIFAR-10 LDS (drop 5%)')
ax.set_ylim(-0.2, 1.1); ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('/home/mac/infusion/MAGIC/gpt2_lds/output/cifar10_logreg_lds.png', dpi=150)
print("Plot saved!", flush=True)
