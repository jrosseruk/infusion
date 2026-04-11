#!/usr/bin/env python
"""
EXACT cifar10-fast architecture translated to standard PyTorch.
With MAGIC paper modifications (width×2.5, final_scale=0.04, etc.)
Includes whitening, Nesterov, loss.sum() convention.
Quick FD sanity check then Hessian eigenvalue measurement.
"""
import sys, os, time, gc, math, copy
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import functional_call

DEVICE = 'cuda:0'
SEED = 42

# MAGIC paper hyperparameters
WIDTH_MULT = 2.5
FINAL_SCALE = 0.04
BIAS_SCALE = 8.0  # Bias LR = BIAS_SCALE × non-bias LR
POOL_TYPE = 'avg'  # Start with avg (paper says LSE but let's check stability first)
MAX_LR = 1.2
MOMENTUM = 0.875
WD = 0.001
BS = 1000
EPOCHS = 12


# ===== Whitening =====
def compute_whitening(train_data, patch_size=3, eps=1e-2):
    """Compute whitening filter from training data (PCA of patches)."""
    # train_data: [N, 3, 32, 32]
    c = train_data.shape[1]
    h = w = patch_size
    # Extract patches
    patches = train_data[:10000].unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
    n = patches.shape[0]
    flat = patches.reshape(n, -1)  # [n, c*h*w]
    # Covariance
    flat = flat / math.sqrt(n - 1)
    cov = flat.t() @ flat  # [c*h*w, c*h*w]
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.t().reshape(c*h*w, c, h, w).flip(0)
    # Whitening filter
    W = eigenvectors / torch.sqrt(eigenvalues[:, None, None, None] + eps)
    return W  # [27, 3, 3, 3]


class WhiteningConv(nn.Module):
    """Fixed (non-trainable) whitening convolution."""
    def __init__(self, weight):
        super().__init__()
        self.conv = nn.Conv2d(3, weight.shape[0], 3, 1, 1, bias=False)
        self.conv.weight.data.copy_(weight)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


# ===== Model =====
class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, pool=None):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()  # Paper uses GELU for metasmoothness
        self.pool = pool

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return self.act(self.bn(x))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return x + out


class CIFARNet(nn.Module):
    """Exact cifar10-fast ResNet-9 with MAGIC modifications."""
    def __init__(self, whitening_weight, width_mult=WIDTH_MULT):
        super().__init__()
        w = lambda c: int(c * width_mult)
        pool = nn.AvgPool2d(2) if POOL_TYPE == 'avg' else None  # TODO: LSE

        self.whiten = WhiteningConv(whitening_weight)
        self.prep = nn.Sequential(
            nn.Conv2d(27, w(64), 1, bias=False),  # 1x1 conv after whitening (27→w(64))
            nn.BatchNorm2d(w(64)), nn.GELU()
        )
        self.layer1 = nn.Sequential(ConvBN(w(64), w(128), pool=nn.AvgPool2d(2)), ResBlock(w(128)))
        self.layer2 = ConvBN(w(128), w(256), pool=nn.AvgPool2d(2))
        self.layer3 = nn.Sequential(ConvBN(w(256), w(512), pool=nn.AvgPool2d(2)), ResBlock(w(512)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(w(512), 10, bias=False)

    def forward(self, x):
        x = self.whiten(x)
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(self.pool(x))
        return self.linear(x) * FINAL_SCALE


def load_cifar10():
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((125.31/255, 122.95/255, 113.87/255),
                                               (62.99/255, 62.09/255, 66.70/255))])
    tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', False, download=True, transform=tf)
    return (torch.stack([tr[i][0] for i in range(len(tr))]),
            torch.tensor([tr[i][1] for i in range(len(tr))]),
            torch.stack([te[i][0] for i in range(len(te))]),
            torch.tensor([te[i][1] for i in range(len(te))]))


def get_lr(step, total):
    peak = int(total * 0.5)
    if step < peak:
        f = step / max(peak, 1)
        return MAX_LR * (0.07 + f * 0.93)
    else:
        f = (step - peak) / max(total - peak, 1)
        return MAX_LR * (1.0 - f * 0.8)


def nesterov_step(params, grads, velocities, lr_map, momentum, wd):
    """Nesterov update matching cifar10-fast exactly."""
    new_v = {}
    with torch.no_grad():
        for n, p in params.items():
            g = grads.get(n)
            if g is None:
                new_v[n] = velocities[n]
                continue
            lr = lr_map[n]
            # dw = -lr * (grad + wd * w)
            dw = -(g + wd * p) * lr
            # v = momentum * v + dw
            v = momentum * velocities[n].to(p.device) + dw
            # w += dw + momentum * v
            p.add_(dw + momentum * v)
            new_v[n] = v.cpu()
    return new_v


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    tx, ty, ex, ey = load_cifar10()
    N = len(tx); spe = N // BS; total = spe * EPOCHS
    print(f"N={N}, BS={BS}, total_steps={total}", flush=True)

    # Compute whitening from training data
    print("Computing whitening filter...", flush=True)
    W = compute_whitening(tx)
    print(f"Whitening: {W.shape}", flush=True)

    # Batch indices
    torch.manual_seed(SEED)
    bi = []
    for e in range(EPOCHS):
        perm = torch.randperm(N).tolist()
        for i in range(0, N, BS):
            b = perm[i:i+BS]
            if len(b) == BS: bi.append(b)
    bi = bi[:total]

    # Build model
    torch.manual_seed(SEED)
    model = CIFARNet(W).to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {nparams:,}", flush=True)

    # Identify bias params (BN biases get BIAS_SCALE × LR)
    bias_names = {n for n, p in model.named_parameters() if 'bias' in n and p.requires_grad}
    print(f"Bias params: {len(bias_names)}", flush=True)

    # Train with Nesterov SGD
    velocities = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters() if p.requires_grad}
    saved = {0: {n: p.data.cpu().clone() for n, p in model.named_parameters() if p.requires_grad}}
    saved_v = {0: {n: v.clone() for n, v in velocities.items()}}

    model.train()
    for step in range(total):
        x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
        base_lr = get_lr(step, total) / BS  # Divide by BS (loss.sum convention)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='sum')  # SUM loss!
        model.zero_grad(); loss.backward()

        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        lr_map = {n: base_lr * (BIAS_SCALE if n in bias_names else 1.0) for n in grads}
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        # No WD on BN params
        wd_val = WD * BS  # Scale WD by BS (loss.sum convention)
        velocities = nesterov_step(params, grads, velocities, lr_map, MOMENTUM, wd_val)

        if (step + 1) % 10 == 0:
            saved[step + 1] = {n: p.data.cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            saved_v[step + 1] = {n: v.clone() for n, v in velocities.items()}
        if (step + 1) % 100 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  Step {step+1}/{total} lr={base_lr*BS:.4f} loss={loss.item()/BS:.4f} acc={acc:.0%}", flush=True)

    model.eval()
    with torch.no_grad():
        tacc = (model(ex[:1000].to(DEVICE)).argmax(1) == ey[:1000].to(DEVICE)).float().mean().item()
    print(f"Test acc: {tacc:.0%}", flush=True)

    # ===== Hessian eigenvalue =====
    print("\nHessian top eigenvalue...", flush=True)
    x_h = tx[:BS].to(DEVICE); y_h = ty[:BS].to(DEVICE)
    params_list = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params_list]
    vn = sum((vi**2).sum() for vi in v)**0.5
    v = [vi / vn for vi in v]
    for it in range(20):
        model.zero_grad()
        logits = model(x_h)
        loss = F.cross_entropy(logits, y_h, reduction='sum')  # Match training loss
        grads = torch.autograd.grad(loss, params_list, create_graph=True)
        dot = sum((g * vi).sum() for g, vi in zip(grads, v))
        hvp = torch.autograd.grad(dot, params_list)
        lam = sum((h * vi).sum().item() for h, vi in zip(hvp, v))
        hn = sum((h**2).sum().item() for h in hvp)**0.5
        v = [h.detach() / max(hn, 1e-10) for h in hvp]
        if (it+1) % 10 == 0:
            eff_lr = MAX_LR / BS  # Effective LR per sample
            print(f"  iter {it+1}: λ={lam:.2f}, eff_lr×λ={eff_lr*lam:.6f}", flush=True)
    eff_lr = MAX_LR / BS
    print(f"\nλ_max = {lam:.2f}", flush=True)
    print(f"Effective LR = {eff_lr:.6f} (= {MAX_LR}/{BS})", flush=True)
    print(f"eff_lr × λ_max = {eff_lr * lam:.6f} {'< 2 STABLE' if eff_lr*lam < 2 else '> 2 UNSTABLE'}", flush=True)
    # Also check with per-step LR at peak
    peak_lr = MAX_LR / BS
    print(f"Peak: lr×λ = {peak_lr * lam:.6f}", flush=True)

    # ===== Quick FD validation (50 steps) =====
    print("\n--- FD validation (50 steps) ---", flush=True)
    QSTEPS = 50
    torch.manual_seed(SEED)
    model_q = CIFARNet(W).to(DEVICE)
    vel_q = {n: torch.zeros_like(p, device='cpu') for n, p in model_q.named_parameters() if p.requires_grad}
    all_p = [{n: p.data.cpu().clone() for n, p in model_q.named_parameters() if p.requires_grad}]
    all_v = [{n: v.clone() for n, v in vel_q.items()}]
    model_q.train()
    for step in range(QSTEPS):
        x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
        base_lr = get_lr(step, QSTEPS) / BS
        logits = model_q(x); loss = F.cross_entropy(logits, y, reduction='sum')
        model_q.zero_grad(); loss.backward()
        grads = {n: p.grad for n, p in model_q.named_parameters() if p.requires_grad and p.grad is not None}
        lr_map = {n: base_lr * (BIAS_SCALE if n in bias_names else 1.0) for n in grads}
        params = {n: p for n, p in model_q.named_parameters() if p.requires_grad}
        vel_q = nesterov_step(params, grads, vel_q, lr_map, MOMENTUM, WD * BS)
        all_p.append({n: p.data.cpu().clone() for n, p in model_q.named_parameters() if p.requires_grad})
        all_v.append({n: v.clone() for n, v in vel_q.items()})
    print(f"Trained {QSTEPS} steps, loss={loss.item()/BS:.4f}", flush=True)

    # Replay VJP for test sample 0
    trainable_names = [n for n, p in model_q.named_parameters() if p.requires_grad]
    with torch.no_grad():
        for n, p in model_q.named_parameters():
            if n in all_p[-1]: p.copy_(all_p[-1][n].to(DEVICE))
    model_q.eval(); model_q.zero_grad()
    for p in model_q.parameters(): p.requires_grad_(True)
    logits = model_q(ex[0:1].to(DEVICE))
    tl = F.cross_entropy(logits, ey[0:1].to(DEVICE), reduction='sum')
    tl.backward()
    dt = {n: p.grad.cpu().clone() for n, p in model_q.named_parameters() if n in trainable_names}
    dm = {n: torch.zeros_like(dt[n]) for n in dt}
    del logits, tl

    influence = torch.zeros(N)
    for t in range(QSTEPS-1, -1, -1):
        pl = {n: all_p[t][n].to(DEVICE).detach().requires_grad_(True) for n in trainable_names}
        vl = {n: all_v[t][n].to(DEVICE).detach().requires_grad_(True) for n in trainable_names}
        w = torch.ones(BS, device=DEVICE, requires_grad=True)
        # Need to call model with these params
        logits = functional_call(model_q, pl, args=(tx[bi[t]].to(DEVICE),))
        per_loss = F.cross_entropy(logits, ty[bi[t]].to(DEVICE), reduction='none')
        wl = (w * per_loss).sum()  # sum loss
        plist = list(pl.values()); names = list(pl.keys())
        grads_fc = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)

        base_lr = get_lr(t, QSTEPS) / BS
        A = torch.tensor(0.0, device=DEVICE)
        for i, n in enumerate(names):
            g = grads_fc[i] if grads_fc[i] is not None else torch.zeros_like(pl[n])
            lr_p = base_lr * (BIAS_SCALE if n in bias_names else 1.0)
            wd_p = WD * BS
            # Nesterov: dw = -lr*(g + wd*w); v_new = mom*v + dw; w_new = w + dw + mom*v_new
            dw = -(g + wd_p * pl[n]) * lr_p
            v_new = MOMENTUM * vl[n] + dw
            p_new = pl[n] + dw + MOMENTUM * v_new
            A = A + (p_new * dt[n].to(DEVICE).detach()).sum()
            A = A + (v_new * dm[n].to(DEVICE).detach()).sum()

        ag = torch.autograd.grad(A, plist + list(vl.values()) + [w], allow_unused=True)
        np_ = len(names)
        dt = {names[i]: ag[i].cpu().detach() if ag[i] is not None else dt[names[i]] for i in range(np_)}
        dm = {names[i]: ag[np_+i].cpu().detach() if ag[np_+i] is not None else dm[names[i]] for i in range(np_)}
        beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(BS)
        for i, idx in enumerate(bi[t]): influence[idx] += beta[i].item()
        del logits, per_loss, wl, grads_fc, A, ag, pl, vl, w, plist
        if t % 10 == 0:
            dn = sum(v.norm().item()**2 for v in dt.values())**0.5
            print(f"  Step {t}: dt={dn:.2e}", flush=True)

    has_nan = torch.isnan(influence).any().item()
    print(f"Influence: [{influence.min():.6f}, {influence.max():.6f}], NaN={has_nan}", flush=True)

    # FD check
    eps = 1e-3
    for si in [0, 500, 2000]:
        if not any(si in bi[t] for t in range(QSTEPS)): continue
        def train_fd(pidx, ev):
            torch.manual_seed(SEED); mf = CIFARNet(W).to(DEVICE)
            vf = {n: torch.zeros_like(p, device='cpu') for n, p in mf.named_parameters() if p.requires_grad}
            bn_f = {n for n, p in mf.named_parameters() if 'bias' in n and p.requires_grad}
            sw = torch.ones(N); sw[pidx] += ev; mf.train()
            for step in range(QSTEPS):
                x = tx[bi[step]].to(DEVICE); y = ty[bi[step]].to(DEVICE)
                wb = sw[bi[step]].to(DEVICE); base_lr = get_lr(step, QSTEPS) / BS
                logits = mf(x); per_loss = F.cross_entropy(logits, y, reduction='none')
                loss = (wb * per_loss).sum()
                mf.zero_grad(); loss.backward()
                grads = {n: p.grad for n, p in mf.named_parameters() if p.requires_grad and p.grad is not None}
                lr_map = {n: base_lr * (BIAS_SCALE if n in bn_f else 1.0) for n in grads}
                params = {n: p for n, p in mf.named_parameters() if p.requires_grad}
                vf = nesterov_step(params, grads, vf, lr_map, MOMENTUM, WD * BS)
            mf.eval()
            with torch.no_grad():
                return F.cross_entropy(mf(ex[0:1].to(DEVICE)), ey[0:1].to(DEVICE), reduction='sum').item()
        l0 = train_fd(si, 0.0); l1 = train_fd(si, eps)
        fd = (l1 - l0) / eps; vjp = influence[si].item()
        ratio = vjp / fd if abs(fd) > 1e-10 else float('inf')
        print(f"  Sample {si}: FD={fd:.6f} VJP={vjp:.6f} ratio={ratio:.4f}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
