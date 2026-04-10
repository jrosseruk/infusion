#!/usr/bin/env python
"""MAGIC LDS: GPT-2 + SGD + correct counterfactual (same batches, weighted loss)."""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch, numpy as np
import torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gpt2_lds.data import prepare_wikitext_datasets, TensorDataset, create_gpt2_model
from torch.func import functional_call

DEVICE = 'cuda:0'
LR = 3e-5
NUM_STEPS = 200
NUM_TEST = 5
NUM_CF = 100
BS = 8
DROP_FRACS = [0.01, 0.05, 0.10, 0.20]

def per_sample_mean_loss(logits, labels, mask):
    sl = logits[:, :-1, :].contiguous(); la = labels[:, 1:].contiguous()
    sm = mask[:, 1:].contiguous().float()
    B, T, V = sl.shape
    ptl = F.cross_entropy(sl.reshape(-1,V), la.reshape(-1), reduction='none').reshape(B,T)
    return (ptl * sm).sum(dim=1) / sm.sum(dim=1).clamp(min=1)

print("Loading data...", flush=True)
train_hf, valid_hf = prepare_wikitext_datasets(512)
train_ds = TensorDataset(train_hf); valid_ds = TensorDataset(valid_hf)
N = len(train_ds)

# Generate batch indices (fixed for all experiments)
torch.manual_seed(42)
perm = torch.randperm(N).tolist()
batch_indices = []
for i in range(0, N, BS):
    b = perm[i:i+BS]
    if len(b) == BS:
        batch_indices.append(b)
batch_indices = batch_indices[:NUM_STEPS]
print(f"Using {NUM_STEPS} batches of {BS}", flush=True)

# ===== Train base model with SGD =====
print(f"\nTraining base model (SGD lr={LR}, {NUM_STEPS} steps)...", flush=True)
model = create_gpt2_model(eager_attention=True).to(DEVICE)
init_p = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

# Save ALL states on CPU
all_params = [init_p]
model.train()
for step in range(NUM_STEPS):
    batch = train_ds.collate(batch_indices[step])
    out = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
    ps = per_sample_mean_loss(out.logits, batch['labels'].to(DEVICE), batch['attention_mask'].to(DEVICE))
    loss = ps.sum()  # w=1 for all, so sum of per-sample mean losses
    model.zero_grad(); loss.backward()
    with torch.no_grad():
        for p in model.parameters(): p.add_(p.grad, alpha=-LR)
    all_params.append({n: p.data.cpu().clone() for n, p in model.named_parameters()})
    if (step+1) % 50 == 0:
        print(f"  Step {step+1}: loss={loss.item():.2f}", flush=True)

# ===== Influence via Replay =====
print(f"\nComputing influence for {NUM_TEST} test samples...", flush=True)
influences = torch.zeros(NUM_TEST, N)
base_losses = torch.zeros(NUM_TEST)

for j in range(NUM_TEST):
    t0 = time.time()
    with torch.no_grad():
        for n, p in model.named_parameters(): p.copy_(all_params[-1][n].to(DEVICE))
    model.eval(); model.zero_grad()
    for p in model.parameters(): p.requires_grad_(True)
    inp = valid_ds.input_ids[j].unsqueeze(0).to(DEVICE)
    att = valid_ds.attention_mask[j].unsqueeze(0).to(DEVICE)
    lab = valid_ds.labels[j].unsqueeze(0).to(DEVICE)
    out = model(input_ids=inp, attention_mask=att)
    ps = per_sample_mean_loss(out.logits, lab, att); tl = ps.sum(); tl.backward()
    delta = {n: p.grad.cpu().clone() for n, p in model.named_parameters()}
    base_losses[j] = tl.item(); del out, ps, tl

    for t in range(NUM_STEPS - 1, -1, -1):
        pl = {n: all_params[t][n].to(DEVICE).detach().requires_grad_(True) for n in all_params[t]}
        w = torch.ones(BS, device=DEVICE, requires_grad=True)
        batch = train_ds.collate(batch_indices[t])
        out = functional_call(model, pl, args=(), kwargs={
            'input_ids': batch['input_ids'].to(DEVICE),
            'attention_mask': batch['attention_mask'].to(DEVICE)})
        ps = per_sample_mean_loss(out.logits, batch['labels'].to(DEVICE), batch['attention_mask'].to(DEVICE))
        wl = (w * ps).sum()
        plist = list(pl.values()); names = list(pl.keys())
        grads = torch.autograd.grad(wl, plist, create_graph=True)
        A = torch.tensor(0.0, device=DEVICE)
        for i in range(len(names)):
            theta_new = pl[names[i]] - LR * grads[i]
            A = A + (theta_new * delta[names[i]].to(DEVICE)).sum()
        ag = torch.autograd.grad(A, plist + [w], allow_unused=True)
        for i, n in enumerate(names):
            delta[n] = ag[i].cpu() if ag[i] is not None else delta[n]
        beta = ag[-1].cpu() if ag[-1] is not None else torch.zeros(BS)
        for i, idx in enumerate(batch_indices[t]):
            influences[j, idx] += beta[i].item()
        del pl, w, out, ps, wl, grads, A, ag, beta, plist
        gc.collect(); torch.cuda.empty_cache()

    dn = sum(v.norm().item()**2 for v in delta.values())**0.5
    print(f"  Test {j}: {time.time()-t0:.0f}s loss={base_losses[j]:.4f} "
          f"inf=[{influences[j].min():.6f},{influences[j].max():.6f}] dt={dn:.2e}", flush=True)

# ===== Counterfactual: SAME batches, WEIGHTED loss =====
print(f"\nCounterfactual (SAME batches, weighted loss, {NUM_CF} subsets)...", flush=True)
gt = {}; dm = {}
for df in DROP_FRACS:
    nd = int(N * df)
    rng = np.random.RandomState(42 + int(df * 1000))
    masks = np.ones((NUM_CF, N)); true_l = np.zeros((NUM_CF, NUM_TEST))

    for s in range(NUM_CF):
        pc = rng.permutation(N); masks[s, pc[:nd]] = 0
        sample_weights = torch.ones(N)
        sample_weights[pc[:nd]] = 0  # Zero weight for dropped samples

        m2 = create_gpt2_model(eager_attention=True).to(DEVICE)
        with torch.no_grad():
            for n, p in m2.named_parameters(): p.copy_(init_p[n].to(DEVICE))
        m2.train()
        for step in range(NUM_STEPS):
            batch = train_ds.collate(batch_indices[step])
            w_batch = sample_weights[batch_indices[step]].to(DEVICE)  # Per-sample weights
            out = m2(input_ids=batch['input_ids'].to(DEVICE),
                     attention_mask=batch['attention_mask'].to(DEVICE))
            ps = per_sample_mean_loss(out.logits, batch['labels'].to(DEVICE),
                                      batch['attention_mask'].to(DEVICE))
            loss = (w_batch * ps).sum()  # WEIGHTED loss!
            m2.zero_grad(); loss.backward()
            with torch.no_grad():
                for p in m2.parameters(): p.add_(p.grad, alpha=-LR)

        m2.eval()
        with torch.no_grad():
            for j2 in range(NUM_TEST):
                inp = valid_ds.input_ids[j2].unsqueeze(0).to(DEVICE)
                att = valid_ds.attention_mask[j2].unsqueeze(0).to(DEVICE)
                lab = valid_ds.labels[j2].unsqueeze(0).to(DEVICE)
                out = m2(input_ids=inp, attention_mask=att)
                true_l[s, j2] = per_sample_mean_loss(out.logits, lab, att).item()
        del m2; torch.cuda.empty_cache()
        if (s+1) % 20 == 0:
            print(f"  Drop {int(df*100)}%: {s+1}/{NUM_CF}", flush=True)

    gt[df] = true_l; dm[df] = masks

# ===== LDS =====
print("\n" + "=" * 60)
print("MAGIC GPT-2 WikiText LDS (SGD, 200 steps)")
print("=" * 60)
lds = {}
for df in DROP_FRACS:
    dw = dm[df] - 1
    rs = []
    for j in range(NUM_TEST):
        pred = base_losses[j].item() + dw @ influences[j].numpy()
        r, _ = spearmanr(pred, gt[df][:, j])
        rs.append(r)
    lds[df] = {"mean": np.nanmean(rs), "std": np.nanstd(rs), "per": rs}
    print(f"Drop {int(df*100):>3}%: LDS = {np.nanmean(rs):.4f} +/- {np.nanstd(rs):.4f}  "
          f"{[f'{r:.3f}' for r in rs]}")
print("=" * 60)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
x = [int(d * 100) for d in DROP_FRACS]
ax.errorbar(x, [lds[d]['mean'] for d in DROP_FRACS], [lds[d]['std'] for d in DROP_FRACS],
            fmt='D-', color='#ef8632', linewidth=2, markersize=8, capsize=4, label='MAGIC (SGD, ours)')
px = [1, 5, 10, 20]
ax.plot(px, [0.910, 0.973, 0.974, 0.970], 'd--', color='#ef8632', alpha=0.4, linewidth=1.5,
        markersize=6, label='MAGIC (paper, Adam)')
ax.plot(px, [0.342, 0.362, 0.362, 0.362], 's--', color='#4a7cb6', linewidth=1.5,
        markersize=6, label='EKFAC (paper)')
ax.plot(px, [-0.002, 0.030, 0.026, 0.003], 'o--', color='#8e529f', linewidth=1.5,
        markersize=6, label='TRAK (paper)')
ax.set_xlabel('Drop Fraction (%)', fontsize=13)
ax.set_ylabel('Spearman Correlation (LDS)', fontsize=13)
ax.set_title('MAGIC GPT-2 WikiText LDS', fontsize=14)
ax.set_ylim(-0.15, 1.05); ax.grid(True, alpha=0.3)
ax.legend(loc='lower left', fontsize=10)
plt.tight_layout()
save = '/home/mac/infusion/MAGIC/gpt2_lds/output/magic_lds_plot.png'
plt.savefig(save, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {save}")
