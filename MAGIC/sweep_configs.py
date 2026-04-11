#!/usr/bin/env python
"""
Sweep metasmoothness configs on 8 GPUs in parallel.
Each config: train 50 steps → Replay VJP → FD comparison.
If VJP/FD ratio ≈ 1.0, the config is metasmooth enough for MAGIC.
"""
import sys, os, time, gc, math
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch.func import functional_call

SEED = 42

# ===== Model =====
class LSEPool(nn.Module):
    def __init__(self, k, T=0.1):
        super().__init__()
        self.k = k; self.T = T
    def forward(self, x):
        p = x.unfold(2, self.k, self.k).unfold(3, self.k, self.k)
        p = p.contiguous().view(*p.shape[:4], -1)
        return self.T * (torch.logsumexp(p / self.T, dim=-1) - math.log(self.k**2))

class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, act='gelu', bn_mom=0.1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out, momentum=bn_mom)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, c, act='gelu', bn_mom=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c, momentum=bn_mom)
        self.act1 = nn.GELU() if act == 'gelu' else nn.ReLU(True)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c, momentum=bn_mom)
    def forward(self, x):
        return self.act1(self.bn1(self.conv1(x)))
        # Note: second conv+bn without activation, then add skip
        # Actually let me fix this - need both convs

class ResBlock(nn.Module):
    def __init__(self, c, act='gelu', bn_mom=0.1):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c, momentum=bn_mom),
            nn.GELU() if act == 'gelu' else nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c, momentum=bn_mom),
        )
    def forward(self, x):
        return x + self.path(x)

def make_model(cfg, device):
    w = lambda c: int(c * cfg['width'])
    pool = lambda k: nn.AvgPool2d(k) if cfg['pool'] == 'avg' else LSEPool(k, cfg.get('pool_T', 0.1))
    act = cfg.get('act', 'gelu')
    bn_mom = cfg.get('bn_mom', 0.1)
    model = nn.Sequential(
        ConvBN(3, w(64), act, bn_mom),
        ConvBN(w(64), w(128), act, bn_mom), pool(2), ResBlock(w(128), act, bn_mom),
        ConvBN(w(128), w(256), act, bn_mom), pool(2),
        ConvBN(w(256), w(512), act, bn_mom), pool(2), ResBlock(w(512), act, bn_mom),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(w(512), 10, bias=False),
    )
    return model.to(device)

def get_lr(step, total, max_lr, start=0.07, end=0.2, peak=0.5):
    pk = int(total * peak)
    if step < pk:
        f = step / max(pk, 1)
        return max_lr * (start + f * (1.0 - start))
    else:
        f = (step - pk) / max(total - pk, 1)
        return max_lr * (1.0 - f * (1.0 - end))

# ===== Per-config test =====
def test_config(gpu_id, cfg, tx, ty, ex, ey, results):
    device = f'cuda:{gpu_id}'
    name = cfg['name']
    N = len(tx); STEPS = cfg.get('steps', 50); BS = cfg['batch']
    max_lr = cfg['lr']; mom = cfg['momentum']; wd = cfg['wd']
    final_scale = cfg['final_scale']
    bias_scale = cfg.get('bias_scale', 1.0)

    torch.manual_seed(SEED)
    model = make_model(cfg, device)
    nparams = sum(p.numel() for p in model.parameters())
    bias_names = {n for n, p in model.named_parameters() if 'bias' in n}

    # Batch indices
    perm = torch.randperm(N).tolist()
    bi = [perm[i:i+BS] for i in range(0, N, BS)][:STEPS]

    # Train
    mom_buf = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    all_p = [{n: p.data.cpu().clone() for n, p in model.named_parameters()}]
    all_m = [{n: v.clone() for n, v in mom_buf.items()}]
    model.train()
    for step in range(STEPS):
        x = tx[bi[step]].to(device); y = ty[bi[step]].to(device)
        lr_t = get_lr(step, STEPS, max_lr)
        logits = model(x) * final_scale
        loss = F.cross_entropy(logits, y)
        model.zero_grad(); loss.backward()
        nm = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None: continue
                lr_p = lr_t * (bias_scale if n in bias_names else 1.0)
                wd_p = wd if n not in bias_names else 0.0
                m = mom * mom_buf[n].to(device) + p.grad + wd_p * p
                p.add_(m, alpha=-lr_p); nm[n] = m.cpu()
        mom_buf = nm
        all_p.append({n: p.data.cpu().clone() for n, p in model.named_parameters()})
        all_m.append({n: v.clone() for n, v in mom_buf.items()})

    # Hessian top eigenvalue (quick, 10 iters)
    model.eval()
    x_h = tx[:min(BS, 500)].to(device); y_h = ty[:min(BS, 500)].to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]
    vn = sum((vi**2).sum() for vi in v)**0.5
    v = [vi / vn for vi in v]
    lam = 0
    for _ in range(15):
        model.zero_grad()
        logits = model(x_h) * final_scale
        loss = F.cross_entropy(logits, y_h)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * vi).sum() for g, vi in zip(grads, v))
        hvp = torch.autograd.grad(dot, params)
        lam = sum((h * vi).sum().item() for h, vi in zip(hvp, v))
        hn = sum((h**2).sum().item() for h in hvp)**0.5
        v = [h.detach() / max(hn, 1e-10) for h in hvp]

    # VJP influence (Replay backward through STEPS)
    with torch.no_grad():
        for n, p in model.named_parameters(): p.copy_(all_p[-1][n].to(device))
    model.eval(); model.zero_grad()
    for p in model.parameters(): p.requires_grad_(True)
    logits = model(ex[0:1].to(device)) * final_scale
    tl = F.cross_entropy(logits, ey[0:1].to(device)); tl.backward()
    dt = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
    dm = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    base_loss = tl.item(); del logits, tl

    influence = torch.zeros(N)
    for t in range(STEPS-1, -1, -1):
        pl = {n: all_p[t][n].to(device).detach().requires_grad_(True) for n in all_p[t]}
        ml = {n: all_m[t][n].to(device).detach().requires_grad_(True) for n in all_m[t]}
        w = torch.ones(BS, device=device, requires_grad=True)
        logits = functional_call(model, pl, args=(tx[bi[t]].to(device),)) * final_scale
        per_loss = F.cross_entropy(logits, ty[bi[t]].to(device), reduction='none')
        wl = (w * per_loss).sum() / BS
        plist = list(pl.values()); names = list(pl.keys())
        grads = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)
        A = torch.tensor(0.0, device=device)
        for i, n in enumerate(names):
            g = grads[i] if grads[i] is not None else torch.zeros_like(pl[n])
            lr_p = get_lr(t, STEPS, max_lr) * (bias_scale if n in bias_names else 1.0)
            wd_p = wd if n not in bias_names else 0.0
            m_new = mom * ml[n] + g + wd_p * pl[n]
            t_new = pl[n] - lr_p * m_new
            A = A + (t_new * dt[n].to(device).detach()).sum()
            A = A + (m_new * dm[n].to(device).detach()).sum()
        ag = torch.autograd.grad(A, plist + list(ml.values()) + [w], allow_unused=True)
        np_ = len(names)
        dt = {names[i]: ag[i].cpu().detach() if ag[i] is not None else dt[names[i]] for i in range(np_)}
        dm = {names[i]: ag[np_+i].cpu().detach() if ag[np_+i] is not None else dm[names[i]] for i in range(np_)}
        beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(BS)
        for i, idx in enumerate(bi[t]): influence[idx] += beta[i].item()
        del logits, per_loss, wl, grads, A, ag, pl, ml, w, plist

    dt_norm = sum(v.norm().item()**2 for v in dt.values())**0.5
    has_nan = torch.isnan(influence).any().item()

    # FD check for 3 samples
    fd_results = []
    eps = 1e-3
    for si in [0, 500, 2000]:
        if not any(si in bi[t] for t in range(STEPS)): continue
        def train_fd(perturb_idx, eps_val):
            torch.manual_seed(SEED)
            mf = make_model(cfg, device)
            momf = {n: torch.zeros_like(p, device='cpu') for n, p in mf.named_parameters()}
            bn_f = {n for n, p in mf.named_parameters() if 'bias' in n}
            sw = torch.ones(N); sw[perturb_idx] += eps_val
            mf.train()
            for step in range(STEPS):
                x = tx[bi[step]].to(device); y = ty[bi[step]].to(device)
                wb = sw[bi[step]].to(device); lr_t = get_lr(step, STEPS, max_lr)
                logits = mf(x) * final_scale
                per_loss = F.cross_entropy(logits, y, reduction='none')
                loss = (wb * per_loss).sum() / BS
                mf.zero_grad(); loss.backward()
                nm = {}
                with torch.no_grad():
                    for n, p in mf.named_parameters():
                        if p.grad is None: continue
                        lr_p = lr_t * (bias_scale if n in bn_f else 1.0)
                        wd_p = wd if n not in bn_f else 0.0
                        m = mom * momf[n].to(device) + p.grad + wd_p * p
                        p.add_(m, alpha=-lr_p); nm[n] = m.cpu()
                momf = nm
            mf.eval()
            with torch.no_grad():
                return F.cross_entropy(mf(ex[0:1].to(device)) * final_scale, ey[0:1].to(device)).item()
        l0 = train_fd(si, 0.0); l1 = train_fd(si, eps)
        fd = (l1 - l0) / eps; vjp = influence[si].item()
        ratio = vjp / fd if abs(fd) > 1e-10 else float('inf')
        fd_results.append((si, fd, vjp, ratio))

    result = {
        'name': name, 'params': nparams, 'lambda_max': lam, 'lr_x_lambda': max_lr * lam,
        'dt_norm': dt_norm, 'nan': has_nan, 'inf_range': (influence.min().item(), influence.max().item()),
        'fd_results': fd_results, 'base_loss': base_loss,
    }
    results[name] = result

    # Print summary
    fd_str = " | ".join([f"s{s}: FD={fd:.4f} VJP={vjp:.4f} r={r:.2f}" for s, fd, vjp, r in fd_results])
    print(f"[GPU{gpu_id}] {name}: λ={lam:.1f} lr×λ={max_lr*lam:.1f} dt={dt_norm:.2e} "
          f"nan={has_nan} | {fd_str}", flush=True)


def main():
    print("Loading CIFAR-10...", flush=True)
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', False, download=True, transform=tf)
    tx = torch.stack([tr[i][0] for i in range(len(tr))])
    ty = torch.tensor([tr[i][1] for i in range(len(tr))])
    ex = torch.stack([te[i][0] for i in range(len(te))])
    ey = torch.tensor([te[i][1] for i in range(len(te))])

    # ===== Configs to test =====
    configs = [
        # Config A: MAGIC paper exact
        {'name': 'MAGIC_paper', 'lr': 1.2, 'momentum': 0.875, 'wd': 0.001, 'batch': 1000,
         'width': 2.5, 'final_scale': 0.04, 'pool': 'lse', 'bias_scale': 8.0, 'act': 'gelu', 'steps': 50},
        # Config B: MGD metasmooth
        {'name': 'MGD_smooth', 'lr': 0.5, 'momentum': 0.85, 'wd': 1e-5, 'batch': 250,
         'width': 2.0, 'final_scale': 0.125, 'pool': 'avg', 'bias_scale': 1.0, 'bn_mom': 0.5, 'act': 'gelu', 'steps': 50},
        # Config C: Very small LR (should be stable)
        {'name': 'small_lr', 'lr': 0.01, 'momentum': 0.875, 'wd': 0.001, 'batch': 1000,
         'width': 2.5, 'final_scale': 0.04, 'pool': 'avg', 'bias_scale': 1.0, 'act': 'gelu', 'steps': 50},
        # Config D: Tiny final_scale (extreme smoothing)
        {'name': 'tiny_scale', 'lr': 1.2, 'momentum': 0.875, 'wd': 0.001, 'batch': 1000,
         'width': 2.5, 'final_scale': 0.001, 'pool': 'avg', 'bias_scale': 1.0, 'act': 'gelu', 'steps': 50},
        # Config E: No momentum (simpler dynamics)
        {'name': 'no_momentum', 'lr': 0.01, 'momentum': 0.0, 'wd': 0.001, 'batch': 1000,
         'width': 2.5, 'final_scale': 0.04, 'pool': 'avg', 'bias_scale': 1.0, 'act': 'gelu', 'steps': 50},
        # Config F: MGD + smaller final_scale
        {'name': 'MGD_smaller', 'lr': 0.5, 'momentum': 0.85, 'wd': 1e-5, 'batch': 250,
         'width': 2.0, 'final_scale': 0.01, 'pool': 'avg', 'bias_scale': 1.0, 'bn_mom': 0.5, 'act': 'gelu', 'steps': 50},
        # Config G: Width 1.0 (smaller model, smaller Hessian)
        {'name': 'narrow', 'lr': 0.5, 'momentum': 0.85, 'wd': 1e-5, 'batch': 250,
         'width': 1.0, 'final_scale': 0.04, 'pool': 'avg', 'bias_scale': 1.0, 'act': 'gelu', 'steps': 50},
        # Config H: ReLU (less smooth but standard)
        {'name': 'relu', 'lr': 0.5, 'momentum': 0.85, 'wd': 1e-5, 'batch': 250,
         'width': 2.0, 'final_scale': 0.125, 'pool': 'avg', 'bias_scale': 1.0, 'act': 'relu', 'steps': 50},
    ]

    print(f"\nTesting {len(configs)} configs on {len(configs)} GPUs...\n", flush=True)

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    results = manager.dict()

    processes = []
    for i, cfg in enumerate(configs):
        gpu_id = i % 8
        p = mp.Process(target=test_config, args=(gpu_id, cfg, tx, ty, ex, ey, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Summary
    print("\n" + "=" * 100, flush=True)
    print(f"{'Config':<15} {'λ_max':>8} {'lr×λ':>8} {'dt_norm':>10} {'NaN':>5} {'FD ratio':>10} {'Verdict':>15}", flush=True)
    print("-" * 100, flush=True)
    for cfg in configs:
        r = results.get(cfg['name'])
        if r is None:
            print(f"{cfg['name']:<15} FAILED", flush=True)
            continue
        best_ratio = min((abs(fd[3]) for fd in r['fd_results'] if abs(fd[3]) < 1e10), default=float('inf'))
        verdict = "GOOD" if 0.5 < best_ratio < 2.0 else ("CLOSE" if 0.1 < best_ratio < 10 else "BAD")
        print(f"{r['name']:<15} {r['lambda_max']:>8.1f} {r['lr_x_lambda']:>8.1f} {r['dt_norm']:>10.2e} "
              f"{str(r['nan']):>5} {best_ratio:>10.2f} {verdict:>15}", flush=True)
    print("=" * 100, flush=True)
    print("\nGOOD = FD ratio 0.5-2.0 (accurate influence). Run full LDS on GOOD configs.", flush=True)


if __name__ == '__main__':
    main()
