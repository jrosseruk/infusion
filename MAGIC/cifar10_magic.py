#!/usr/bin/env python
"""
MAGIC CIFAR-10 ResNet-9 LDS - EXACT paper hyperparameters.

Architecture: ResNet-9 (jordan202494) with metasmoothness mods
Training: SGD momentum=0.875, LR=1.2 one-cycle, batch=1000, 12 epochs (600 steps)
LDS: 50 test samples, drop 5%, 100 counterfactual subsets
Parallel: 8 GPUs for influence + CF retraining
"""
import sys, os, time, gc, math, argparse
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from scipy.stats import spearmanr
from torch.func import functional_call

# ============================================================
# EXACT MAGIC paper hyperparameters (Table 1, Appendix A)
# ============================================================
LR = 1.2
MOMENTUM = 0.875
WEIGHT_DECAY = 0.001
BATCH_SIZE = 1000
EPOCHS = 12
BIAS_SCALE = 8.0
WIDTH_MULT = 2.5
FINAL_SCALE = 0.04
POOL_TEMP = 0.1  # LSE pooling epsilon
LR_START = 0.07  # Start at 7% of max LR
LR_END = 0.2     # End at 20% of max LR
LR_PEAK = 0.5    # Peak at 50% of training
SEED = 42
NUM_TEST = 50
NUM_CF = 100
DROP_FRAC = 0.05
SAVE_EVERY = 5  # Checkpoint every 5 steps (600/5 = 120 checkpoints)

# ============================================================
# Model: MetasmoothResNet9 with EXACT paper specs
# ============================================================
class LogSumExpPool2d(nn.Module):
    """Smooth pooling: temperature * (logsumexp(x/temperature) - log(k))"""
    def __init__(self, kernel_size, stride=None, temperature=POOL_TEMP):
        super().__init__()
        self.k = kernel_size
        self.s = stride or kernel_size
        self.T = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        # Use unfold for proper 2D pooling
        # Pad if needed
        x_pad = x
        patches = x_pad.unfold(2, self.k, self.s).unfold(3, self.k, self.s)  # B,C,H',W',k,k
        patches = patches.contiguous().view(*patches.shape[:4], -1)  # B,C,H',W',k*k
        # LogSumExp with temperature
        return self.T * (torch.logsumexp(patches / self.T, dim=-1) - math.log(self.k * self.k))


class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.GELU(), ConvBN(c, c), nn.GELU(), ConvBN(c, c),
        )

    def forward(self, x):
        return x + self.block(x)


def make_model(device='cpu'):
    """Create ResNet-9 matching MAGIC paper specs."""
    w = lambda c: int(c * WIDTH_MULT)
    model = nn.Sequential(
        # Prep
        ConvBN(3, w(64)), nn.GELU(),
        # Layer 1 + pool + res
        ConvBN(w(64), w(128)), nn.GELU(),
        LogSumExpPool2d(2),
        ResBlock(w(128)),
        # Layer 2 + pool
        ConvBN(w(128), w(256)), nn.GELU(),
        LogSumExpPool2d(2),
        # Layer 3 + pool + res
        ConvBN(w(256), w(512)), nn.GELU(),
        LogSumExpPool2d(2),
        ResBlock(w(512)),
        # Classifier
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(w(512), 10, bias=True),
        Mul(FINAL_SCALE),  # CRITICAL: scale logits by 0.04 for metasmoothness
    )
    # Scale classifier bias by BIAS_SCALE (before the Mul layer)
    with torch.no_grad():
        model[-2].bias.mul_(BIAS_SCALE)
    return model.to(device)


class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale


# ============================================================
# Training utilities
# ============================================================
def get_lr(step, total):
    peak = int(total * LR_PEAK)
    if step < peak:
        f = step / max(peak, 1)
        return LR * (LR_START + f * (1.0 - LR_START))
    else:
        f = (step - peak) / max(total - peak, 1)
        return LR * (1.0 - f * (1.0 - LR_END))


def load_cifar10():
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', train=False, download=True, transform=tf)
    tx = torch.stack([tr[i][0] for i in range(len(tr))])
    ty = torch.tensor([tr[i][1] for i in range(len(tr))])
    ex = torch.stack([te[i][0] for i in range(len(te))])
    ey = torch.tensor([te[i][1] for i in range(len(te))])
    return tx, ty, ex, ey


# ============================================================
# Main
# ============================================================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT = '/home/mac/infusion/MAGIC/gpt2_lds/output'
    os.makedirs(OUT, exist_ok=True)

    print("Loading CIFAR-10...", flush=True)
    tx, ty, ex, ey = load_cifar10()
    N = len(tx)
    spe = N // BATCH_SIZE  # 50 steps per epoch
    total = spe * EPOCHS   # 600 total steps
    print(f"N={N}, batch={BATCH_SIZE}, steps/epoch={spe}, total={total}", flush=True)

    # Generate deterministic batch indices
    torch.manual_seed(SEED)
    bi = []
    for e in range(EPOCHS):
        perm = torch.randperm(N).tolist()
        for i in range(0, N, BATCH_SIZE):
            b = perm[i:i+BATCH_SIZE]
            if len(b) == BATCH_SIZE:
                bi.append(b)
    bi = bi[:total]

    # ===== TRAIN =====
    print(f"\nTraining ResNet-9 (SGD mom={MOMENTUM}, lr={LR}, {total} steps)...", flush=True)
    device = 'cuda:0'
    torch.manual_seed(SEED)
    model = make_model(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Params: {nparams:,}", flush=True)

    mom_buf = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    saved = {}
    saved[0] = {
        'p': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
        'm': {n: v.clone() for n, v in mom_buf.items()},
    }

    model.train()
    for step in range(total):
        x = tx[bi[step]].to(device); y = ty[bi[step]].to(device)
        lr_t = get_lr(step, total)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        model.zero_grad(); loss.backward()
        nm = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is None: continue
                m = MOMENTUM * mom_buf[n].to(device) + p.grad + WEIGHT_DECAY * p
                p.add_(m, alpha=-lr_t)
                nm[n] = m.cpu()
        mom_buf = nm
        if (step + 1) % SAVE_EVERY == 0 or step == total - 1:
            saved[step + 1] = {
                'p': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
                'm': {n: v.clone() for n, v in mom_buf.items()},
            }
        if (step + 1) % 100 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  Step {step+1}/{total} lr={lr_t:.4f} loss={loss.item():.4f} acc={acc:.0%}", flush=True)

    model.eval()
    with torch.no_grad():
        tacc = (model(ex[:1000].to(device)).argmax(1) == ey[:1000].to(device)).float().mean().item()
    print(f"Test accuracy: {tacc:.0%}", flush=True)
    print(f"Checkpoints: {len(saved)}", flush=True)

    # ===== QUICK VALIDATION: Finite Difference vs VJP for 1 test sample =====
    print("\n--- Quick FD validation (1 test, 5 samples) ---", flush=True)
    # Compute influence for test sample 0 via Replay
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(saved[total]['p'][n].to(device))
    model.eval(); model.zero_grad()
    for p in model.parameters(): p.requires_grad_(True)
    logits = model(ex[0:1].to(device))
    tl = F.cross_entropy(logits, ey[0:1].to(device)); tl.backward()
    dt = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
    dm = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
    base_l = tl.item()
    del logits, tl

    def replay_seg_local(s_state, start, end):
        theta = {n: v.clone() for n, v in s_state['p'].items()}
        mom = {n: v.clone() for n, v in s_state['m'].items()}
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

    def vjp_local(theta, mom, batch_idx, dt, dm, lr_t):
        # FULL fp64 VJP to prevent overflow in adjoint propagation
        model.double()  # Cast model buffers (BN running stats etc.) to fp64
        pl = {n: theta[n].to(device).detach().double().requires_grad_(True) for n in theta}
        ml = {n: mom[n].to(device).detach().double().requires_grad_(True) for n in mom}
        w = torch.ones(len(batch_idx), device=device, dtype=torch.float64, requires_grad=True)
        logits = functional_call(model, pl, args=(tx[batch_idx].to(device).double(),))
        per_loss = F.cross_entropy(logits, ty[batch_idx].to(device), reduction='none')
        wl = (w * per_loss).sum() / len(batch_idx)
        plist = list(pl.values()); names = list(pl.keys())
        grads = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)
        A = torch.tensor(0.0, device=device, dtype=torch.float64)
        for i, n in enumerate(names):
            g = grads[i] if grads[i] is not None else torch.zeros_like(pl[n])
            m_new = MOMENTUM * ml[n] + g + WEIGHT_DECAY * pl[n]
            t_new = pl[n] - lr_t * m_new
            A = A + (t_new * dt[n].to(device).detach()).sum()
            A = A + (m_new * dm[n].to(device).detach()).sum()
        ag = torch.autograd.grad(A, plist + list(ml.values()) + [w], allow_unused=True)
        np_ = len(names)
        ndt = {names[i]: ag[i].cpu().detach() if ag[i] is not None else dt[names[i]] for i in range(np_)}
        ndm = {names[i]: ag[np_+i].cpu().detach() if ag[np_+i] is not None else dm[names[i]] for i in range(np_)}
        beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(len(batch_idx), dtype=torch.float64)
        model.float()  # Restore fp32 for forward replay
        del logits, per_loss, wl, grads, A, ag, pl, ml, w, plist
        return beta, ndt, ndm

    influence_quick = torch.zeros(N, dtype=torch.float64)
    seg_starts = sorted(saved.keys())
    t_start = time.time()
    for si in range(len(seg_starts) - 1, 0, -1):
        ss = seg_starts[si - 1]; se = seg_starts[si]
        states = replay_seg_local(saved[ss], ss, se)
        for k in range(len(states) - 2, -1, -1):
            t = ss + k; lr_t = get_lr(t, total)
            beta, dt, dm = vjp_local(states[k]['p'], states[k]['m'], bi[t], dt, dm, lr_t)
            for i, idx in enumerate(bi[t]):
                influence_quick[idx] += beta[i].double().item()
        del states; gc.collect(); torch.cuda.empty_cache()
        if (si) % 20 == 0:
            dn = sum(v.norm().item()**2 for v in dt.values())**0.5
            step_done = total - seg_starts[si]
            print(f"  Replay: {step_done}/{total} steps, dt={dn:.2e}, "
                  f"inf=[{influence_quick.min():.4f},{influence_quick.max():.4f}]", flush=True)

    dn = sum(v.norm().item()**2 for v in dt.values())**0.5
    print(f"  Full replay done in {time.time()-t_start:.0f}s, dt_norm={dn:.2e}", flush=True)
    print(f"  Influence: [{influence_quick.min():.6f}, {influence_quick.max():.6f}]", flush=True)
    has_nan = torch.isnan(influence_quick).any().item()
    print(f"  NaN: {has_nan}", flush=True)

    # FD validation for 5 training samples
    eps = 1e-3
    fd_samples = [0, 100, 500, 1000, 5000]
    print(f"\n  FD validation (eps={eps}):", flush=True)
    for si in fd_samples:
        # Find which steps this sample appears in
        steps_with_sample = [t for t in range(total) if si in bi[t]]
        if not steps_with_sample:
            print(f"    Sample {si}: not in any batch, skip", flush=True)
            continue

        def train_eval_perturbed(sample_idx, eps_val):
            torch.manual_seed(SEED)
            m_fd = make_model(device)
            mom_fd = {n: torch.zeros_like(p, device='cpu') for n, p in m_fd.named_parameters()}
            sw = torch.ones(N); sw[sample_idx] += eps_val
            m_fd.train()
            for step in range(total):
                x = tx[bi[step]].to(device); y = ty[bi[step]].to(device)
                wb = sw[bi[step]].to(device)
                lr_t = get_lr(step, total)
                logits = m_fd(x)
                per_loss = F.cross_entropy(logits, y, reduction='none')
                loss = (wb * per_loss).sum() / len(bi[step])  # Fixed denominator = batch_size
                m_fd.zero_grad(); loss.backward()
                nm = {}
                with torch.no_grad():
                    for n, p in m_fd.named_parameters():
                        if p.grad is None: continue
                        m = MOMENTUM * mom_fd[n].to(device) + p.grad + WEIGHT_DECAY * p
                        p.add_(m, alpha=-lr_t); nm[n] = m.cpu()
                mom_fd = nm
            m_fd.eval()
            with torch.no_grad():
                logits = m_fd(ex[0:1].to(device))
                loss_val = F.cross_entropy(logits, ey[0:1].to(device)).item()
            del m_fd; torch.cuda.empty_cache()
            return loss_val

        l_base = train_eval_perturbed(si, 0.0)
        l_pert = train_eval_perturbed(si, eps)
        fd = (l_pert - l_base) / eps
        vjp_val = influence_quick[si].item()
        ratio = vjp_val / fd if abs(fd) > 1e-10 else float('inf')
        print(f"    Sample {si}: FD={fd:.6f} VJP={vjp_val:.6f} ratio={ratio:.4f}", flush=True)

    print("--- End FD validation ---\n", flush=True)

    # If influence has NaN, abort
    if has_nan:
        print("ERROR: Influence contains NaN. Aborting.", flush=True)
        return

    # Save everything for parallel workers
    torch.save({
        'saved': saved, 'bi': bi, 'total': total,
        'tx': tx, 'ty': ty, 'ex': ex, 'ey': ey,
    }, f'{OUT}/cifar10_state.pt')
    print(f"Saved state to {OUT}/cifar10_state.pt", flush=True)

    # ===== INFLUENCE (parallel across 8 GPUs) =====
    print(f"\nComputing influence for {NUM_TEST} test samples on 8 GPUs...", flush=True)

    # Launch worker processes
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    def influence_worker(gpu_id, test_indices, result_dict):
        """Compute influence for assigned test samples on one GPU."""
        device = f'cuda:{gpu_id}'
        state = torch.load(f'{OUT}/cifar10_state.pt', map_location='cpu', weights_only=False)
        saved = state['saved']; bi = state['bi']; total_steps = state['total']
        tx_l = state['tx']; ty_l = state['ty']; ex_l = state['ex']; ey_l = state['ey']
        N_l = len(tx_l)

        torch.manual_seed(SEED)
        model_l = make_model(device)

        def replay_seg(s_state, start, end):
            theta = {n: v.clone() for n, v in s_state['p'].items()}
            mom = {n: v.clone() for n, v in s_state['m'].items()}
            states = [{'p': theta, 'm': mom}]
            for t in range(start, end):
                with torch.no_grad():
                    for n, p in model_l.named_parameters():
                        p.copy_(theta[n].to(device))
                model_l.train(); model_l.zero_grad()
                logits = model_l(tx_l[bi[t]].to(device))
                loss = F.cross_entropy(logits, ty_l[bi[t]].to(device))
                loss.backward()
                lr_t = get_lr(t, total_steps)
                nt = {}; nm = {}
                for n, p in model_l.named_parameters():
                    if p.grad is None: continue
                    m = MOMENTUM * mom[n] + p.grad.cpu() + WEIGHT_DECAY * theta[n]
                    nt[n] = theta[n] - lr_t * m; nm[n] = m
                theta = nt; mom = nm
                states.append({'p': theta, 'm': mom})
            return states

        def vjp(theta, mom, batch_idx, dt, dm, lr_t):
            # Full fp64 VJP
            model_l.double()
            pl = {n: theta[n].to(device).detach().double().requires_grad_(True) for n in theta}
            ml = {n: mom[n].to(device).detach().double().requires_grad_(True) for n in mom}
            B = len(batch_idx)
            w = torch.ones(B, device=device, dtype=torch.float64, requires_grad=True)
            logits = functional_call(model_l, pl, args=(tx_l[batch_idx].to(device).double(),))
            per_loss = F.cross_entropy(logits, ty_l[batch_idx].to(device), reduction='none')
            wl = (w * per_loss).sum() / B
            plist = list(pl.values()); names = list(pl.keys())
            grads = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)
            A = torch.tensor(0.0, device=device, dtype=torch.float64)
            for i, n in enumerate(names):
                g = grads[i] if grads[i] is not None else torch.zeros_like(pl[n])
                m_new = MOMENTUM * ml[n] + g + WEIGHT_DECAY * pl[n]
                t_new = pl[n] - lr_t * m_new
                A = A + (t_new * dt[n].to(device).detach()).sum()
                A = A + (m_new * dm[n].to(device).detach()).sum()
            ag = torch.autograd.grad(A, plist + list(ml.values()) + [w], allow_unused=True)
            np_ = len(names)
            ndt = {names[i]: ag[i].cpu().detach() if ag[i] is not None else dt[names[i]] for i in range(np_)}
            ndm = {names[i]: ag[np_+i].cpu().detach() if ag[np_+i] is not None else dm[names[i]] for i in range(np_)}
            beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(B, dtype=torch.float64)
            model_l.float()
            del logits, per_loss, wl, grads, A, ag, pl, ml, w, plist
            return beta, ndt, ndm

        for j_local, j_global in enumerate(test_indices):
            t0 = time.time()
            # Load final params
            with torch.no_grad():
                for n, p in model_l.named_parameters():
                    p.copy_(saved[total_steps]['p'][n].to(device))
            model_l.eval(); model_l.zero_grad()
            for p in model_l.parameters(): p.requires_grad_(True)

            logits = model_l(ex_l[j_global:j_global+1].to(device))
            tl = F.cross_entropy(logits, ey_l[j_global:j_global+1].to(device))
            tl.backward()
            dt = {n: p.grad.cpu().clone() for n, p in model_l.named_parameters()}
            dm = {n: torch.zeros_like(p, device='cpu') for n, p in model_l.named_parameters()}
            base_loss = tl.item()
            del logits, tl

            influence = torch.zeros(N_l, dtype=torch.float64)
            seg_starts = sorted(saved.keys())
            for si in range(len(seg_starts) - 1, 0, -1):
                ss = seg_starts[si - 1]; se = seg_starts[si]
                states = replay_seg(saved[ss], ss, se)
                for k in range(len(states) - 2, -1, -1):
                    t = ss + k; lr_t = get_lr(t, total_steps)
                    beta, dt, dm = vjp(states[k]['p'], states[k]['m'], bi[t], dt, dm, lr_t)
                    for i, idx in enumerate(bi[t]):
                        influence[idx] += beta[i].double().item()
                del states; gc.collect(); torch.cuda.empty_cache()

            dn = sum(v.norm().item()**2 for v in dt.values())**0.5
            result_dict[j_global] = (influence.clone(), base_loss)
            print(f"  GPU{gpu_id} test {j_global}: {time.time()-t0:.0f}s loss={base_loss:.4f} "
                  f"inf=[{influence.min():.4f},{influence.max():.4f}] dt={dn:.2e}", flush=True)

    # Distribute test samples across 8 GPUs
    test_assignments = [[] for _ in range(8)]
    for j in range(NUM_TEST):
        test_assignments[j % 8].append(j)

    processes = []
    for gpu_id in range(8):
        if not test_assignments[gpu_id]:
            continue
        p = mp.Process(target=influence_worker, args=(gpu_id, test_assignments[gpu_id], result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect results
    influences = torch.zeros(NUM_TEST, N, dtype=torch.float64)
    base_losses = torch.zeros(NUM_TEST)
    for j in range(NUM_TEST):
        inf, bl = result_dict[j]
        influences[j] = inf
        base_losses[j] = bl

    has_nan = torch.isnan(influences).any().item()
    print(f"\nInfluence: shape={influences.shape}, nan={has_nan}", flush=True)
    print(f"Range: [{influences.min():.4f}, {influences.max():.4f}]", flush=True)
    torch.save({'influences': influences, 'base_losses': base_losses}, f'{OUT}/cifar10_influences.pt')

    # ===== COUNTERFACTUAL (parallel across 8 GPUs) =====
    print(f"\nCounterfactual ({NUM_CF} subsets, drop {int(DROP_FRAC*100)}%, 8 GPUs)...", flush=True)
    nd = int(N * DROP_FRAC)
    rng = np.random.RandomState(SEED + 1000)
    masks = np.ones((NUM_CF, N))
    all_weights = []
    for s in range(NUM_CF):
        pc = rng.permutation(N); masks[s, pc[:nd]] = 0
        sw = torch.ones(N); sw[pc[:nd]] = 0
        all_weights.append(sw)

    cf_results = manager.dict()

    def cf_worker(gpu_id, subset_indices, all_weights, cf_results):
        device = f'cuda:{gpu_id}'
        state = torch.load(f'{OUT}/cifar10_state.pt', map_location='cpu', weights_only=False)
        bi_l = state['bi']; total_l = state['total']
        tx_l = state['tx']; ty_l = state['ty']; ex_l = state['ex']; ey_l = state['ey']

        for s in subset_indices:
            torch.manual_seed(SEED)
            m2 = make_model(device)
            mom2 = {n: torch.zeros_like(p, device='cpu') for n, p in m2.named_parameters()}
            sw = all_weights[s]
            m2.train()
            for step in range(total_l):
                x = tx_l[bi_l[step]].to(device); y = ty_l[bi_l[step]].to(device)
                wb = sw[bi_l[step]].to(device)
                lr_t = get_lr(step, total_l)
                logits = m2(x)
                per_loss = F.cross_entropy(logits, y, reduction='none')
                loss = (wb * per_loss).sum() / len(bi_l[step])  # Fixed denominator = batch_size
                m2.zero_grad(); loss.backward()
                nm2 = {}
                with torch.no_grad():
                    for n, p in m2.named_parameters():
                        if p.grad is None: continue
                        m = MOMENTUM * mom2[n].to(device) + p.grad + WEIGHT_DECAY * p
                        p.add_(m, alpha=-lr_t)
                        nm2[n] = m.cpu()
                mom2 = nm2
            m2.eval()
            losses = []
            with torch.no_grad():
                for j in range(NUM_TEST):
                    logits = m2(ex_l[j:j+1].to(device))
                    losses.append(F.cross_entropy(logits, ey_l[j:j+1].to(device)).item())
            cf_results[s] = torch.tensor(losses)
            del m2; torch.cuda.empty_cache()
            if (s + 1) % 10 == 0:
                print(f"  GPU{gpu_id}: CF subset {s+1}", flush=True)

    # Distribute CF subsets across 8 GPUs
    cf_assignments = [[] for _ in range(8)]
    for s in range(NUM_CF):
        cf_assignments[s % 8].append(s)

    processes = []
    for gpu_id in range(8):
        if not cf_assignments[gpu_id]:
            continue
        p = mp.Process(target=cf_worker, args=(gpu_id, cf_assignments[gpu_id], all_weights, cf_results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    true_losses = torch.zeros(NUM_CF, NUM_TEST)
    for s in range(NUM_CF):
        true_losses[s] = cf_results[s]

    # ===== LDS =====
    print("\n" + "=" * 60, flush=True)
    print(f"MAGIC CIFAR-10 ResNet-9 LDS (drop {int(DROP_FRAC*100)}%)", flush=True)
    print("=" * 60, flush=True)

    dw = masks - 1
    all_r = []
    for j in range(NUM_TEST):
        pred = base_losses[j].item() + dw @ influences[j].numpy()
        r, _ = spearmanr(pred, true_losses[:, j].numpy())
        all_r.append(r)
    all_r = np.array(all_r)
    mean_lds = np.nanmean(all_r)
    std_lds = np.nanstd(all_r)

    print(f"Mean LDS: {mean_lds:.4f} +/- {std_lds:.4f}", flush=True)
    print(f"Paper reference (drop 5%): 0.922", flush=True)
    print(f"Per-sample (first 10): {[f'{r:.3f}' for r in all_r[:10]]}", flush=True)
    print("=" * 60, flush=True)

    # Save results
    torch.save({'lds': all_r, 'influences': influences, 'base_losses': base_losses,
                'true_losses': true_losses, 'masks': masks}, f'{OUT}/cifar10_lds_results.pt')

    # Plot
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(['MAGIC\n(ours)', 'MAGIC\n(paper)', 'EKFAC\n(paper)', 'TRAK\n(paper)'],
           [mean_lds, 0.922, 0.249, 0.362],
           yerr=[std_lds, 0.017, 0.061, 0.078],
           color=['#ef8632', '#ef8632', '#4a7cb6', '#8e529f'],
           alpha=[1.0, 0.4, 0.8, 0.8], capsize=4)
    ax.set_ylabel('Spearman Correlation (LDS)', fontsize=12)
    ax.set_title(f'CIFAR-10 ResNet-9 LDS (drop {int(DROP_FRAC*100)}%)', fontsize=13)
    ax.set_ylim(-0.2, 1.1); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUT}/cifar10_magic_lds.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {OUT}/cifar10_magic_lds.png", flush=True)


if __name__ == '__main__':
    main()
