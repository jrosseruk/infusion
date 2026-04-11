#!/usr/bin/env python
"""
MAGIC CIFAR-10 - EXACT architecture from cifar10-fast + paper modifications.

Architecture from dawn_utils.py (jordan202494):
  prep → layer1(+res) → layer2 → layer3(+res) → pool → linear → Mul

Paper modifications:
  - MaxPool → LogSumExp(ε=0.1)  [metasmoothness]
  - ReLU → GELU                  [metasmoothness]
  - width_mult=2.5               [metasmoothness]
  - final_scale=0.04             [metasmoothness]
  - bias_scale=8.0               [BN bias LR multiplier]
  - linear bias=False             [from original]
"""
import sys, os, time, gc, math
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import functional_call

DEVICE = 'cuda:0'
SEED = 42

# Paper hyperparameters
LR = 1.2; MOMENTUM = 0.875; WD = 0.001; BS = 1000; EPOCHS = 12
FINAL_SCALE = 0.04; BIAS_SCALE = 8.0; POOL_TEMP = 0.1; WIDTH_MULT = 2.5
LR_START = 0.07; LR_END = 0.2; LR_PEAK = 0.5

def get_lr(step, total):
    peak = int(total * LR_PEAK)
    if step < peak:
        f = step / max(peak, 1)
        return LR * (LR_START + f * (1.0 - LR_START))
    else:
        f = (step - peak) / max(total - peak, 1)
        return LR * (1.0 - f * (1.0 - LR_END))


# ===== Exact architecture from cifar10-fast, modified for metasmoothness =====
class LSEPool(nn.Module):
    def __init__(self, k, T=POOL_TEMP):
        super().__init__()
        self.k = k; self.T = T
    def forward(self, x):
        p = x.unfold(2, self.k, self.k).unfold(3, self.k, self.k)
        p = p.contiguous().view(*p.shape[:4], -1)
        return self.T * (torch.logsumexp(p / self.T, dim=-1) - math.log(self.k ** 2))


class ConvBN(nn.Module):
    """Conv + BatchNorm + GELU (paper uses GELU instead of ReLU for smoothness)"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Two conv-bn-act blocks with skip connection"""
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        # Note: activation AFTER add in original (ReLU). But for smoothness we use GELU.
    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + x  # No activation after add (matches original dawn_utils)


class CIFARResNet9(nn.Module):
    """Exact cifar10-fast ResNet-9 with metasmoothness modifications."""
    def __init__(self):
        super().__init__()
        w = lambda c: int(c * WIDTH_MULT)
        # prep
        self.prep = ConvBN(3, w(64))
        # layer1: conv+pool+res
        self.layer1_conv = ConvBN(w(64), w(128))
        self.layer1_pool = LSEPool(2)
        self.layer1_res = ResBlock(w(128))
        # layer2: conv+pool (no res)
        self.layer2_conv = ConvBN(w(128), w(256))
        self.layer2_pool = LSEPool(2)
        # layer3: conv+pool+res
        self.layer3_conv = ConvBN(w(256), w(512))
        self.layer3_pool = LSEPool(2)
        self.layer3_res = ResBlock(w(512))
        # classifier (NO bias, matching original)
        self.pool_final = LSEPool(4)  # 4x4 → 1x1, LSE for smoothness
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(w(512), 10, bias=False)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_pool(self.layer1_conv(x))
        x = self.layer1_res(x)
        x = self.layer2_pool(self.layer2_conv(x))
        x = self.layer3_pool(self.layer3_conv(x))
        x = self.layer3_res(x)
        x = self.flatten(self.pool_final(x))
        return self.linear(x) * FINAL_SCALE


def load_cifar10():
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', False, download=True, transform=tf)
    return (torch.stack([tr[i][0] for i in range(len(tr))]),
            torch.tensor([tr[i][1] for i in range(len(tr))]),
            torch.stack([te[i][0] for i in range(len(te))]),
            torch.tensor([te[i][1] for i in range(len(te))]))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    tx, ty, ex, ey = load_cifar10()
    N = len(tx); spe = N // BS; total = spe * EPOCHS  # 600
    print(f"N={N}, total_steps={total}", flush=True)

    # Batch indices
    torch.manual_seed(SEED)
    bi = []
    for e in range(EPOCHS):
        perm = torch.randperm(N).tolist()
        for i in range(0, N, BS):
            b = perm[i:i+BS]
            if len(b) == BS: bi.append(b)
    bi = bi[:total]

    # ===== TRAIN with correct bias_scale (BN bias gets 8x LR) =====
    print(f"Training (SGD, lr={LR}, {total} steps)...", flush=True)
    torch.manual_seed(SEED)
    model = CIFARResNet9().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Identify bias vs non-bias parameters
    bias_names = {n for n, p in model.named_parameters() if 'bias' in n}
    print(f"Bias params (get {BIAS_SCALE}x LR): {len(bias_names)}", flush=True)

    mom = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    saved = {0: {'p': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
                  'm': {n: v.clone() for n, v in mom.items()}}}

    model.train()
    for step in range(total):
        x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
        base_lr = get_lr(step, total)
        logits = model(x); loss = F.cross_entropy(logits, y)
        model.zero_grad(); loss.backward()
        nm = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None: continue
                lr_p = base_lr * (BIAS_SCALE if n in bias_names else 1.0)
                # Weight decay NOT applied to BN params
                wd_p = WD if n not in bias_names else 0.0
                m = MOMENTUM * mom[n].to(DEVICE) + p.grad + wd_p * p
                p.add_(m, alpha=-lr_p)
                nm[n] = m.cpu()
        mom = nm
        if (step + 1) % 5 == 0 or step == total - 1:
            saved[step+1] = {'p': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
                              'm': {n: v.clone() for n, v in mom.items()}}
        if (step + 1) % 100 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  Step {step+1}/{total} lr={base_lr:.4f} loss={loss.item():.4f} acc={acc:.0%}", flush=True)

    model.eval()
    with torch.no_grad():
        tacc = (model(ex[:1000].to(DEVICE)).argmax(1) == ey[:1000].to(DEVICE)).float().mean().item()
    print(f"Test acc: {tacc:.0%}", flush=True)

    # ===== Compute Hessian spectral radius at peak LR =====
    print("\nComputing Hessian top eigenvalue...", flush=True)
    model.eval()
    x_h = tx[:BS].to(DEVICE); y_h = ty[:BS].to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    v_norm = sum((vi**2).sum() for vi in v)**0.5
    v = [vi / v_norm for vi in v]
    for it in range(30):
        model.zero_grad()
        logits = model(x_h); loss = F.cross_entropy(logits, y_h)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * vi).sum() for g, vi in zip(grads, v))
        hvp = torch.autograd.grad(dot, params)
        lam = sum((h * vi).sum().item() for h, vi in zip(hvp, v))
        hvp_n = sum((h**2).sum().item() for h in hvp)**0.5
        v = [h.detach() / hvp_n for h in hvp]
        if (it+1) % 10 == 0:
            print(f"  iter {it+1}: λ_max={lam:.4f}, LR×λ={LR*lam:.4f}", flush=True)
    print(f"\nλ_max = {lam:.4f}", flush=True)
    print(f"LR × λ_max = {LR * lam:.4f} {'< 2 (STABLE!)' if LR*lam < 2 else '> 2 (UNSTABLE)'}", flush=True)
    for s in [0, 150, 300, 450, 550]:
        lr_s = get_lr(s, total)
        print(f"  Step {s}: lr={lr_s:.4f} → lr×λ={lr_s*lam:.4f}", flush=True)

    # ===== Quick FD validation (first 50 steps, fp32 since FINAL_SCALE should keep things stable) =====
    print("\n--- Quick FD validation (50 steps) ---", flush=True)
    # ... (can add later after verifying λ_max)

    print("\nDone! Check if LR × λ_max < 2 before running full MAGIC.", flush=True)


if __name__ == '__main__':
    main()
