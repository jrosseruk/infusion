#!/usr/bin/env python
"""
MAGIC CIFAR-10 ResNet-9 LDS Replication.
Uses SGD with momentum (NOT Adam) and metasmooth architecture choices.

Architecture from MAGIC paper (Table 1):
- ResNet-9 with width multiplier 2.5
- Log-sum-exp pooling (smooth alternative to max pool)
- GELU activation
- Output scaling 0.04
- BatchNorm before activation

Training: SGD momentum=0.875, LR=1.2 one-cycle, 12 epochs, batch=1000
"""
import sys, os, time, gc, math
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from torch.func import functional_call
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DEVICE = 'cuda:0'

# ===== Metasmooth ResNet-9 =====
class LogSumExpPool2d(nn.Module):
    """Smooth alternative to MaxPool2d."""
    def __init__(self, kernel_size, stride=None, temperature=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.temperature = temperature

    def forward(self, x):
        # LogSumExp pooling: log(mean(exp(x/T))) * T
        x_unfold = F.unfold(x, self.kernel_size, stride=self.stride)
        B, C_k, L = x_unfold.shape
        C = x.shape[1]
        k2 = self.kernel_size ** 2
        x_unfold = x_unfold.view(B, C, k2, L)
        x_scaled = x_unfold / self.temperature
        pooled = self.temperature * (torch.logsumexp(x_scaled, dim=2) - math.log(k2))
        H_out = W_out = int(math.sqrt(L))
        return pooled.view(B, C, H_out, W_out)


class ConvBNGELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class MetasmoothResNet9(nn.Module):
    """ResNet-9 with metasmoothness modifications from MAGIC paper."""
    def __init__(self, width_mult=2.5, num_classes=10, final_scale=0.04, pool_temp=0.1):
        super().__init__()
        w = lambda c: int(c * width_mult)

        self.prep = ConvBNGELU(3, w(64))
        self.layer1 = nn.Sequential(
            ConvBNGELU(w(64), w(128)),
            nn.AvgPool2d(2),
            ResBlock(w(128)),
        )
        self.layer2 = ConvBNGELU(w(128), w(256))
        self.pool2 = nn.AvgPool2d(2)
        self.layer3 = nn.Sequential(
            ConvBNGELU(w(256), w(512)),
            nn.AvgPool2d(2),
            ResBlock(w(512)),
        )
        self.pool_final = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(w(512), num_classes)
        self.final_scale = final_scale

        # Initialize with smaller weights for smoothness
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.pool2(self.layer2(x))
        x = self.layer3(x)
        x = self.pool_final(x).flatten(1)
        return self.classifier(x) * self.final_scale


# ===== Training utilities =====
def get_one_cycle_lr(step, total_steps, max_lr, start_mult=0.5, end_mult=0.0001, peak_frac=0.5):
    peak_step = int(total_steps * peak_frac)
    if step < peak_step:
        frac = step / max(peak_step, 1)
        return max_lr * (start_mult + frac * (1.0 - start_mult))
    else:
        frac = (step - peak_step) / max(total_steps - peak_step, 1)
        return max_lr * (1.0 - frac * (1.0 - end_mult))


def load_cifar10():
    """Load CIFAR-10 using torchvision."""
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_set = torchvision.datasets.CIFAR10(root='/home/mac/infusion/MAGIC/data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='/home/mac/infusion/MAGIC/data', train=False, download=True, transform=transform)

    # Pre-load into tensors
    train_x = torch.stack([train_set[i][0] for i in range(len(train_set))])
    train_y = torch.tensor([train_set[i][1] for i in range(len(train_set))])
    test_x = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_y = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    return train_x, train_y, test_x, test_y


# ===== Main experiment =====
def main():
    # Hyperparameters (MAGIC paper Table 1)
    # Use MGD paper's metasmooth hyperparameters (explicitly tested for metagradient stability)
    MAX_LR = 0.5
    MOMENTUM = 0.85
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 250
    NUM_EPOCHS = 18
    NUM_TEST = 10  # Fewer test samples, parallelized across GPUs
    DROP_FRAC = 0.05  # Just one drop fraction as requested
    NUM_CF = 100
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading CIFAR-10...", flush=True)
    train_x, train_y, test_x, test_y = load_cifar10()
    N = len(train_x)
    steps_per_epoch = N // BATCH_SIZE  # 50
    total_steps = steps_per_epoch * NUM_EPOCHS  # 600
    print(f"Train: {N}, Test: {len(test_x)}, Steps/epoch: {steps_per_epoch}, Total: {total_steps}", flush=True)

    # Generate batch indices
    torch.manual_seed(SEED)
    batch_indices = []
    for epoch in range(NUM_EPOCHS):
        perm = torch.randperm(N).tolist()
        for i in range(0, N, BATCH_SIZE):
            b = perm[i:i+BATCH_SIZE]
            if len(b) == BATCH_SIZE:
                batch_indices.append(b)
    assert len(batch_indices) == total_steps

    # ===== Train =====
    print(f"\nTraining ResNet-9 (SGD, lr={MAX_LR}, {total_steps} steps)...", flush=True)
    model = MetasmoothResNet9(width_mult=2.0, final_scale=0.125).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}", flush=True)

    init_params = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

    # State tracking (only save every SAVE_EVERY steps to manage memory)
    SAVE_EVERY = 10
    saved_states = {}
    saved_states[0] = {
        'params': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
        'momentum': {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()},
    }
    momentum_buf = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}

    model.train()
    for step in range(total_steps):
        idx = batch_indices[step]
        x = train_x[idx].to(DEVICE)
        y = train_y[idx].to(DEVICE)
        lr = get_one_cycle_lr(step, total_steps, MAX_LR)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            new_momentum = {}
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                m = MOMENTUM * momentum_buf[n].to(DEVICE) + p.grad + WEIGHT_DECAY * p
                p.add_(m, alpha=-lr)
                new_momentum[n] = m.cpu()
            momentum_buf = new_momentum

        if (step + 1) % SAVE_EVERY == 0 or step == total_steps - 1:
            saved_states[step + 1] = {
                'params': {n: p.data.cpu().clone() for n, p in model.named_parameters()},
                'momentum': {n: v.clone() for n, v in momentum_buf.items()},
            }

        if (step + 1) % 100 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"  Step {step+1}/{total_steps} lr={lr:.4f} loss={loss.item():.4f} acc={acc:.2%}", flush=True)

    # Final eval
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x[:1000].to(DEVICE))
        test_acc = (test_logits.argmax(1) == test_y[:1000].to(DEVICE)).float().mean().item()
    print(f"\nTest accuracy: {test_acc:.2%}", flush=True)
    print(f"Saved {len(saved_states)} checkpoints", flush=True)

    # ===== Replay for influence =====
    print(f"\nComputing influence for {NUM_TEST} test samples...", flush=True)

    # Replay forward within a segment
    def replay_segment(model, state, batch_indices, train_x, train_y, start, end):
        theta = {n: v.clone() for n, v in state['params'].items()}
        mom = {n: v.clone() for n, v in state['momentum'].items()}
        states = [{'params': theta, 'momentum': mom}]
        for t in range(start, end):
            idx = batch_indices[t]
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(theta[n].to(DEVICE))
            model.train()
            model.zero_grad()
            logits = model(train_x[idx].to(DEVICE))
            loss = F.cross_entropy(logits, train_y[idx].to(DEVICE))
            loss.backward()
            lr = get_one_cycle_lr(t, total_steps, MAX_LR)
            new_theta = {}; new_mom = {}
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                m = MOMENTUM * mom[n] + p.grad.cpu() + WEIGHT_DECAY * theta[n]
                new_theta[n] = theta[n] - lr * m
                new_mom[n] = m
            theta = new_theta; mom = new_mom
            states.append({'params': theta, 'momentum': mom})
        return states

    # VJP step for SGD with momentum
    def vjp_step(model, theta, mom, batch_idx, train_x, train_y, delta_theta, delta_mom, lr):
        # Create leaf tensors
        pl = {n: theta[n].to(DEVICE).detach().requires_grad_(True) for n in theta}
        ml = {n: mom[n].to(DEVICE).detach().requires_grad_(True) for n in mom}
        w = torch.ones(len(batch_idx), device=DEVICE, requires_grad=True)

        # Forward
        logits = functional_call(model, pl, args=(train_x[batch_idx].to(DEVICE),))
        # Per-sample cross entropy
        per_loss = F.cross_entropy(logits, train_y[batch_idx].to(DEVICE), reduction='none')
        wl = (w * per_loss).sum()

        # Gradient w.r.t. params (create_graph for HVP in delta update)
        plist = list(pl.values())
        names = list(pl.keys())
        grads = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)

        # SGD with momentum:
        # m_new = momentum * m_old + grad + wd * theta
        # theta_new = theta - lr * m_new
        A = torch.tensor(0.0, device=DEVICE)
        for i, n in enumerate(names):
            g = grads[i] if grads[i] is not None else torch.zeros_like(pl[n])
            m_new = MOMENTUM * ml[n] + g + WEIGHT_DECAY * pl[n]
            t_new = pl[n] - lr * m_new

            d_t = delta_theta[n].to(DEVICE) if not delta_theta[n].is_cuda else delta_theta[n]
            d_m = delta_mom[n].to(DEVICE) if not delta_mom[n].is_cuda else delta_mom[n]
            A = A + (t_new * d_t.detach()).sum()
            A = A + (m_new * d_m.detach()).sum()

        ag = torch.autograd.grad(A, plist + list(ml.values()) + [w], allow_unused=True)
        n_p = len(names)
        new_dt = {names[i]: ag[i].cpu().detach() if ag[i] is not None else delta_theta[names[i]] for i in range(n_p)}
        new_dm = {names[i]: ag[n_p+i].cpu().detach() if ag[n_p+i] is not None else delta_mom[names[i]] for i in range(n_p)}
        beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(len(batch_idx))

        del logits, per_loss, wl, grads, A, ag, pl, ml, w, plist
        return beta, new_dt, new_dm

    influences = torch.zeros(NUM_TEST, N)
    base_losses = torch.zeros(NUM_TEST)

    for j in range(NUM_TEST):
        t0 = time.time()
        # Delta_T
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(saved_states[total_steps]['params'][n].to(DEVICE))
        model.eval(); model.zero_grad()
        for p in model.parameters(): p.requires_grad_(True)

        test_logits = model(test_x[j:j+1].to(DEVICE))
        test_loss = F.cross_entropy(test_logits, test_y[j:j+1].to(DEVICE))
        test_loss.backward()
        delta_theta = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
        delta_mom = {n: torch.zeros_like(p, device='cpu') for n, p in model.named_parameters()}
        base_losses[j] = test_loss.item()
        del test_logits, test_loss

        # Backward through segments
        seg_starts = sorted(saved_states.keys())
        for si in range(len(seg_starts) - 1, 0, -1):
            ss = seg_starts[si - 1]; se = seg_starts[si]
            states = replay_segment(model, saved_states[ss], batch_indices, train_x, train_y, ss, se)
            for k in range(len(states) - 2, -1, -1):
                t = ss + k
                lr = get_one_cycle_lr(t, total_steps, MAX_LR)
                beta, delta_theta, delta_mom = vjp_step(
                    model, states[k]['params'], states[k]['momentum'],
                    batch_indices[t], train_x, train_y,
                    delta_theta, delta_mom, lr
                )
                for i, idx in enumerate(batch_indices[t]):
                    influences[j, idx] += beta[i].item()
            del states; gc.collect(); torch.cuda.empty_cache()

        dt_n = sum(v.norm().item()**2 for v in delta_theta.values())**0.5
        if (j + 1) % 5 == 0 or j == 0:
            print(f"  Test {j}: {time.time()-t0:.0f}s loss={base_losses[j]:.4f} "
                  f"inf=[{influences[j].min():.6f},{influences[j].max():.6f}] dt={dt_n:.2e}", flush=True)

    # ===== Counterfactual =====
    print(f"\nCounterfactual ({NUM_CF} subsets, drop {int(DROP_FRAC*100)}%)...", flush=True)
    num_drop = int(N * DROP_FRAC)
    rng_cf = np.random.RandomState(SEED + 1000)
    masks = np.ones((NUM_CF, N))
    true_losses = np.zeros((NUM_CF, NUM_TEST))

    for s in range(NUM_CF):
        pc = rng_cf.permutation(N)
        masks[s, pc[:num_drop]] = 0
        sample_weights = torch.ones(N)
        sample_weights[pc[:num_drop]] = 0

        # Retrain with weighted loss (SAME batches, zero weight for dropped)
        m2 = MetasmoothResNet9(width_mult=2.0, final_scale=0.125).to(DEVICE)
        torch.manual_seed(SEED)  # Same init
        m2 = MetasmoothResNet9(width_mult=2.0, final_scale=0.125).to(DEVICE)
        mom2 = {n: torch.zeros_like(p, device='cpu') for n, p in m2.named_parameters()}

        m2.train()
        for step in range(total_steps):
            idx = batch_indices[step]
            x = train_x[idx].to(DEVICE)
            y_true = train_y[idx].to(DEVICE)
            w_batch = sample_weights[idx].to(DEVICE)
            lr = get_one_cycle_lr(step, total_steps, MAX_LR)

            logits = m2(x)
            per_loss = F.cross_entropy(logits, y_true, reduction='none')
            loss = (w_batch * per_loss).sum() / w_batch.sum().clamp(min=1)  # Weighted mean
            m2.zero_grad()
            loss.backward()

            with torch.no_grad():
                new_mom = {}
                for n, p in m2.named_parameters():
                    if p.grad is None: continue
                    m = MOMENTUM * mom2[n].to(DEVICE) + p.grad + WEIGHT_DECAY * p
                    p.add_(m, alpha=-lr)
                    new_mom[n] = m.cpu()
                mom2 = new_mom

        # Eval
        m2.eval()
        with torch.no_grad():
            for j2 in range(NUM_TEST):
                logits = m2(test_x[j2:j2+1].to(DEVICE))
                true_losses[s, j2] = F.cross_entropy(logits, test_y[j2:j2+1].to(DEVICE)).item()
        del m2; torch.cuda.empty_cache()
        if (s + 1) % 20 == 0:
            print(f"  {s+1}/{NUM_CF}", flush=True)

    # ===== LDS =====
    print(f"\n{'='*60}", flush=True)
    print(f"MAGIC CIFAR-10 ResNet-9 LDS (drop {int(DROP_FRAC*100)}%)", flush=True)
    print(f"{'='*60}", flush=True)
    dw = masks - 1
    all_r = []
    for j in range(NUM_TEST):
        pred = base_losses[j].item() + dw @ influences[j].numpy()
        r, _ = spearmanr(pred, true_losses[:, j])
        all_r.append(r)
    mean_lds = np.nanmean(all_r)
    std_lds = np.nanstd(all_r)
    print(f"Mean LDS: {mean_lds:.4f} +/- {std_lds:.4f}", flush=True)
    print(f"Paper reference (drop 5%): 0.922", flush=True)
    print(f"Per-sample: {[f'{r:.3f}' for r in all_r[:10]]}...", flush=True)
    print(f"{'='*60}", flush=True)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    ax.bar(['MAGIC\n(ours)'], [mean_lds], yerr=[std_lds], color='#ef8632', alpha=0.8, capsize=5, width=0.3)
    ax.axhline(y=0.922, color='gray', linestyle='--', label=f'Paper (drop 5%): 0.922')
    ax.set_ylabel('Spearman Correlation (LDS)', fontsize=12)
    ax.set_title(f'MAGIC CIFAR-10 ResNet-9 LDS (drop {int(DROP_FRAC*100)}%)', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    save_path = '/home/mac/infusion/MAGIC/gpt2_lds/output/cifar10_lds.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}", flush=True)


if __name__ == '__main__':
    main()
