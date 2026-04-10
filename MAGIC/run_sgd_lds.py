#!/usr/bin/env python
"""
MAGIC LDS with GPT-2 fine-tuned with SGD (not Adam).
SGD is well-conditioned for the Replay influence function.
Uses fewer training steps for tractability.
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gpt2_lds.config import MagicConfig
from gpt2_lds.data import prepare_wikitext_datasets, TensorDataset, create_gpt2_model
import torch.nn.functional as F

def compute_per_sample_loss(logits, labels, attention_mask):
    """Mean loss per sample (not sum) for SGD stability."""
    sl = logits[:, :-1, :].contiguous()
    la = labels[:, 1:].contiguous()
    sm = attention_mask[:, 1:].contiguous().float()
    B, T, V = sl.shape
    ptl = F.cross_entropy(sl.reshape(-1, V), la.reshape(-1), reduction='none').reshape(B, T)
    return (ptl * sm).sum(dim=1) / sm.sum(dim=1).clamp(min=1)
from gpt2_lds.replay import _differentiable_adam_step
from torch.func import functional_call

cfg = MagicConfig()
DEVICE = f'cuda:{int(os.environ.get("CUDA_DEVICE", "0"))}'
LR = 3e-5  # Small LR for SGD on GPT-2
NUM_STEPS = 50  # Short training for numerical stability
NUM_TEST = 5
NUM_SUBSETS = 50
DROP_FRACS = [0.05, 0.10, 0.20]
BATCH_SIZE = 8

print("Loading data...", flush=True)
train_hf, valid_hf = prepare_wikitext_datasets(cfg.block_size)
train_ds = TensorDataset(train_hf)
valid_ds = TensorDataset(valid_hf)
num_train = len(train_ds)

# ===== Train with SGD =====
print(f"Training GPT-2 with SGD for {NUM_STEPS} steps (lr={LR})...", flush=True)
model = create_gpt2_model(eager_attention=True).to(DEVICE)
init_params = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

# Batch indices
rng_train = torch.Generator()
rng_train.manual_seed(cfg.seed)
perm = torch.randperm(num_train, generator=rng_train).tolist()
batch_indices = []
for i in range(0, num_train, BATCH_SIZE):
    b = perm[i:i+BATCH_SIZE]
    if len(b) == BATCH_SIZE:
        batch_indices.append(b)
batch_indices = batch_indices[:NUM_STEPS]

# Save all states on CPU
all_params = [init_params]
model.train()
for step in range(NUM_STEPS):
    batch = train_ds.collate(batch_indices[step])
    out = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
    ps = compute_per_sample_loss(out.logits, batch['labels'].to(DEVICE), batch['attention_mask'].to(DEVICE))
    loss = ps.sum()
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(p.grad, alpha=-LR)
    all_params.append({n: p.data.cpu().clone() for n, p in model.named_parameters()})
    if (step + 1) % 10 == 0:
        print(f"  Step {step+1}/{NUM_STEPS}, loss={loss.item():.2f}", flush=True)

# ===== Influence via Replay =====
print(f"\nComputing influence for {NUM_TEST} test samples...", flush=True)
influences = torch.zeros(NUM_TEST, num_train)
base_losses = torch.zeros(NUM_TEST)

for j in range(NUM_TEST):
    t0 = time.time()
    # Delta_T
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(all_params[-1][n].to(DEVICE))
    model.eval()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)
    inp = valid_ds.input_ids[j].unsqueeze(0).to(DEVICE)
    att = valid_ds.attention_mask[j].unsqueeze(0).to(DEVICE)
    lab = valid_ds.labels[j].unsqueeze(0).to(DEVICE)
    out = model(input_ids=inp, attention_mask=att)
    ps = compute_per_sample_loss(out.logits, lab, att)
    tl = ps.sum()
    tl.backward()
    delta = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
    base_losses[j] = tl.item()
    del out, ps, tl, inp, att, lab

    # Backward through training steps
    for t in range(NUM_STEPS - 1, -1, -1):
        pl = {n: all_params[t][n].to(DEVICE).detach().requires_grad_(True) for n in all_params[t]}
        w = torch.ones(BATCH_SIZE, device=DEVICE, requires_grad=True)
        batch = train_ds.collate(batch_indices[t])
        out = functional_call(model, pl, args=(), kwargs={
            'input_ids': batch['input_ids'].to(DEVICE),
            'attention_mask': batch['attention_mask'].to(DEVICE),
        })
        ps = compute_per_sample_loss(out.logits, batch['labels'].to(DEVICE), batch['attention_mask'].to(DEVICE))
        wl = (w * ps).sum()
        plist = list(pl.values())
        names = list(pl.keys())
        grads = torch.autograd.grad(wl, plist, create_graph=True)

        # SGD: theta_new = theta - lr * grad
        A = torch.tensor(0.0, device=DEVICE)
        for i, n in enumerate(names):
            theta_new = pl[n] - LR * grads[i]
            A = A + (theta_new * delta[n].to(DEVICE)).sum()

        ag = torch.autograd.grad(A, plist + [w], allow_unused=True)
        for i, n in enumerate(names):
            delta[n] = ag[i].cpu() if ag[i] is not None else delta[n]
        beta = ag[-1].cpu() if ag[-1] is not None else torch.zeros(BATCH_SIZE)
        for i, idx in enumerate(batch_indices[t]):
            influences[j, idx] += beta[i].item()
        del pl, w, out, ps, wl, grads, A, ag, beta, plist
        gc.collect()
        torch.cuda.empty_cache()

    dt_n = sum(v.norm().item()**2 for v in delta.values())**0.5
    print(f"  Test {j}: {time.time()-t0:.1f}s, loss={base_losses[j]:.2f}, "
          f"inf=[{influences[j].min():.4f},{influences[j].max():.4f}], dt={dt_n:.2e}", flush=True)

# ===== Counterfactual Retraining =====
print(f"\nCounterfactual retraining ({NUM_SUBSETS} subsets × {len(DROP_FRACS)} drop fracs)...", flush=True)
ground_truth = {}
drop_masks = {}

for drop_frac in DROP_FRACS:
    num_drop = int(num_train * drop_frac)
    rng_cf = np.random.RandomState(cfg.seed + int(drop_frac * 10000))
    masks = np.ones((NUM_SUBSETS, num_train))
    true_losses = np.zeros((NUM_SUBSETS, NUM_TEST))

    for s in range(NUM_SUBSETS):
        pc = rng_cf.permutation(num_train)
        masks[s, pc[:num_drop]] = 0
        keep = sorted(pc[num_drop:].tolist())

        m2 = create_gpt2_model(eager_attention=True).to(DEVICE)
        with torch.no_grad():
            for n, p in m2.named_parameters():
                p.copy_(init_params[n].to(DEVICE))
        m2.train()
        rng2 = torch.Generator()
        rng2.manual_seed(cfg.seed)
        sp = torch.randperm(len(keep), generator=rng2).tolist()
        for step in range(min(NUM_STEPS, len(keep) // BATCH_SIZE)):
            bi = [keep[sp[(step * BATCH_SIZE + k) % len(sp)]] for k in range(BATCH_SIZE)]
            batch = train_ds.collate(bi)
            out = m2(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            ps = compute_per_sample_loss(out.logits, batch['labels'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            loss = ps.sum()
            m2.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in m2.parameters():
                    p.add_(p.grad, alpha=-LR)

        m2.eval()
        with torch.no_grad():
            for j2 in range(NUM_TEST):
                inp = valid_ds.input_ids[j2].unsqueeze(0).to(DEVICE)
                att = valid_ds.attention_mask[j2].unsqueeze(0).to(DEVICE)
                lab = valid_ds.labels[j2].unsqueeze(0).to(DEVICE)
                out = m2(input_ids=inp, attention_mask=att)
                true_losses[s, j2] = compute_per_sample_loss(out.logits, lab, att).item()
        del m2
        torch.cuda.empty_cache()
        if (s + 1) % 10 == 0:
            print(f"  Drop {int(drop_frac*100)}%: {s+1}/{NUM_SUBSETS}", flush=True)

    ground_truth[drop_frac] = true_losses
    drop_masks[drop_frac] = masks

# ===== LDS Evaluation =====
print("\n" + "=" * 60)
print("MAGIC GPT-2 WikiText LDS Results (SGD, 50 steps)")
print("=" * 60)

lds_results = {}
for drop_frac in DROP_FRACS:
    true_l = ground_truth[drop_frac]
    msk = drop_masks[drop_frac]
    dw = msk - 1
    all_r = []
    for j in range(NUM_TEST):
        pred = base_losses[j].item() + dw @ influences[j].numpy()
        r, _ = spearmanr(pred, true_l[:, j])
        all_r.append(r)
    mean_r = np.nanmean(all_r)
    std_r = np.nanstd(all_r)
    lds_results[drop_frac] = {"mean": mean_r, "std": std_r, "per_sample": np.array(all_r)}
    print(f"Drop {int(drop_frac*100):>3}%: LDS = {mean_r:.4f} +/- {std_r:.4f}")
print("=" * 60)

# ===== Plot =====
fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
x = [int(d * 100) for d in DROP_FRACS]
means = [lds_results[d]["mean"] for d in DROP_FRACS]
stds = [lds_results[d]["std"] for d in DROP_FRACS]

ax.errorbar(x, means, yerr=stds, fmt="D-", color="#ef8632", linewidth=2, markersize=8,
            capsize=4, label="MAGIC (SGD, ours)", zorder=5)

# Paper reference EKFAC and TRAK
paper_ekfac = {5: 0.362, 10: 0.362, 20: 0.362}
paper_trak = {5: 0.030, 10: 0.026, 20: 0.003}
ax.plot(list(paper_ekfac.keys()), list(paper_ekfac.values()), "s--", color="#4a7cb6",
        linewidth=1.5, markersize=6, label="EKFAC (paper)")
ax.plot(list(paper_trak.keys()), list(paper_trak.values()), "o--", color="#8e529f",
        linewidth=1.5, markersize=6, label="TRAK (paper)")

ax.set_xlabel("Drop Fraction (%)", fontsize=13)
ax.set_ylabel("Spearman Correlation (LDS)", fontsize=13)
ax.set_title("MAGIC GPT-2 WikiText LDS\n(SGD training, 50 steps)", fontsize=14)
ax.set_ylim(-0.15, 1.05)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=10)
plt.tight_layout()
save_path = os.path.join(cfg.output_dir, "magic_lds_sgd.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {save_path}")
