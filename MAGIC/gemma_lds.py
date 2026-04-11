#!/usr/bin/env python
"""
MAGIC LDS on Gemma-2B LoRA - Full pipeline.
Training is confirmed metasmooth (lr×λ < 2 throughout).
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import time, gc
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.func import functional_call

DEVICE = 'cuda:0'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# Hyperparameters (MAGIC paper)
MAX_LR = 0.0002; BETA1 = 0.95; BETA2 = 0.975; WD = 1e-5
EPS_ROOT = 1e-6; EPS = 1e-8
MAX_SEQ_LEN = 256; BATCH_SIZE = 4; NUM_EPOCHS = 1
NUM_TEST = 5; NUM_CF = 30; DROP_FRAC = 0.05; SAVE_EVERY = 20

def get_lr(step, total):
    peak = int(total * 0.25)
    if step < peak:
        f = step / max(peak, 1)
        return MAX_LR * (1e-6 + f * (1.0 - 1e-6))
    else:
        f = (step - peak) / max(total - peak, 1)
        return MAX_LR * (1.0 - f * 0.9)

# Load model
print("Loading Gemma-2B + LoRA...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", torch_dtype=torch.float32, device_map=DEVICE, attn_implementation="eager")
model = get_peft_model(base_model, LoraConfig(
    r=64, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# Load data
print("Loading Dolly dataset...", flush=True)
ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=SEED).select(range(5000))

def tokenize(ex):
    text = "### Instruction:\n%s\n\n### Response:\n%s" % (ex['instruction'], ex['response'])
    tok = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    tok["labels"] = tok["input_ids"].clone()
    tok["labels"][tok["attention_mask"] == 0] = -100
    return {k: v.squeeze(0) for k, v in tok.items()}

tokenized = ds.map(tokenize, remove_columns=ds.column_names)
tokenized.set_format("torch")
all_ids = torch.stack([tokenized[i]["input_ids"] for i in range(len(tokenized))])
all_mask = torch.stack([tokenized[i]["attention_mask"] for i in range(len(tokenized))])
all_labels = torch.stack([tokenized[i]["labels"] for i in range(len(tokenized))])

# Split train/test
test_ids = all_ids[-NUM_TEST:]; test_mask = all_mask[-NUM_TEST:]; test_labels = all_labels[-NUM_TEST:]
train_ids = all_ids[:-NUM_TEST]; train_mask = all_mask[:-NUM_TEST]; train_labels = all_labels[:-NUM_TEST]
N = len(train_ids)
total_steps = (N // BATCH_SIZE) * NUM_EPOCHS
print("Train: %d, Test: %d, Steps: %d" % (N, NUM_TEST, total_steps), flush=True)

# Batch indices
torch.manual_seed(SEED)
bi = []
for e in range(NUM_EPOCHS):
    perm = torch.randperm(N).tolist()
    for i in range(0, N, BATCH_SIZE):
        b = perm[i:i+BATCH_SIZE]
        if len(b) == BATCH_SIZE: bi.append(b)
bi = bi[:total_steps]

def compute_loss(model_or_params, input_ids, attn_mask, labels, weights=None, use_fc=False):
    """Per-sample mean-over-tokens CE loss."""
    if use_fc:
        out = functional_call(model, model_or_params, args=(),
                              kwargs={"input_ids": input_ids, "attention_mask": attn_mask})
    else:
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits
    sl = logits[:, :-1, :].contiguous(); la = labels[:, 1:].contiguous()
    sm = attn_mask[:, 1:].contiguous().float()
    B, T, V = sl.shape
    pt = F.cross_entropy(sl.reshape(-1, V), la.reshape(-1), reduction='none').reshape(B, T)
    ps = (pt * sm).sum(dim=1) / sm.sum(dim=1).clamp(min=1)
    if weights is not None:
        return (weights * ps).sum() / BATCH_SIZE
    return ps.mean()

def test_loss_single(params, idx):
    """Test loss for a single sample using functional_call."""
    out = functional_call(model, params, args=(),
                          kwargs={"input_ids": test_ids[idx:idx+1].to(DEVICE),
                                  "attention_mask": test_mask[idx:idx+1].to(DEVICE)})
    logits = out.logits
    sl = logits[:, :-1, :].contiguous(); la = test_labels[idx:idx+1, 1:].to(DEVICE).contiguous()
    sm = test_mask[idx:idx+1, 1:].to(DEVICE).contiguous().float()
    pt = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), la.reshape(-1), reduction='none').reshape(1, -1)
    return ((pt * sm).sum() / sm.sum().clamp(min=1)).squeeze()

# ===== TRAIN =====
print("\nTraining...", flush=True)
trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
param_names = list(trainable.keys())
m_st = {n: torch.zeros_like(p, device='cpu') for n, p in trainable.items()}
v_st = {n: torch.zeros_like(p, device='cpu') for n, p in trainable.items()}

saved = {0: {'p': {n: p.data.cpu().clone() for n, p in trainable.items()},
             'm': {n: v.clone() for n, v in m_st.items()},
             'v': {n: v.clone() for n, v in v_st.items()}}}

model.train()
for step in range(total_steps):
    idx = bi[step]
    inp = train_ids[idx].to(DEVICE); att = train_mask[idx].to(DEVICE); lab = train_labels[idx].to(DEVICE)
    lr_t = get_lr(step, total_steps)
    loss = compute_loss(None, inp, att, lab)
    model.zero_grad(); loss.backward()
    with torch.no_grad():
        for n, p in trainable.items():
            if p.grad is None: continue
            g = p.grad
            m_st[n] = BETA1 * m_st[n].to(DEVICE) + (1 - BETA1) * g
            v_st[n] = BETA2 * v_st[n].to(DEVICE) + (1 - BETA2) * g * g
            denom = torch.sqrt(v_st[n] + EPS_ROOT) + EPS
            p.add_(m_st[n] / denom, alpha=-lr_t)
            p.add_(p, alpha=-lr_t * WD)
            m_st[n] = m_st[n].cpu(); v_st[n] = v_st[n].cpu()
    if (step + 1) % SAVE_EVERY == 0 or step == total_steps - 1:
        saved[step + 1] = {
            'p': {n: p.data.cpu().clone() for n, p in trainable.items()},
            'm': {n: v.cpu().clone() for n, v in m_st.items()},
            'v': {n: v.cpu().clone() for n, v in v_st.items()}}
    if (step + 1) % 200 == 0:
        print("  Step %d/%d lr=%.6f loss=%.4f" % (step+1, total_steps, lr_t, loss.item()), flush=True)

print("Checkpoints: %d" % len(saved), flush=True)

# ===== Differentiable Adam step for VJP =====
def diff_adam_step(params, m_dict, v_dict, grads, lr):
    new_p = {}; new_m = {}; new_v = {}
    for n in params:
        g = grads[n]
        m_new = BETA1 * m_dict[n] + (1 - BETA1) * g
        v_new = BETA2 * v_dict[n] + (1 - BETA2) * g * g
        denom = torch.sqrt(v_new + EPS_ROOT) + EPS
        p_new = params[n] - lr * m_new / denom - lr * WD * params[n]
        new_p[n] = p_new; new_m[n] = m_new; new_v[n] = v_new
    return new_p, new_m, new_v

# ===== REPLAY INFLUENCE =====
print("\nReplay influence (%d test, %d steps)..." % (NUM_TEST, total_steps), flush=True)
influences = np.zeros((NUM_TEST, N))
base_losses = np.zeros(NUM_TEST)

for j in range(NUM_TEST):
    t0 = time.time()
    # Delta_T
    model.eval()
    final_params = {n: saved[total_steps]['p'][n].to(DEVICE).detach().requires_grad_(True) for n in param_names}
    tl = test_loss_single(final_params, j)
    grads_delta = torch.autograd.grad(tl, list(final_params.values()))
    delta_p = {n: g.cpu() for n, g in zip(param_names, grads_delta)}
    delta_m = {n: torch.zeros_like(delta_p[n]) for n in param_names}
    delta_v = {n: torch.zeros_like(delta_p[n]) for n in param_names}
    base_losses[j] = tl.item()
    del tl, grads_delta, final_params

    # Backward through training
    infl = torch.zeros(N)
    seg_starts = sorted(saved.keys())
    for si in range(len(seg_starts) - 1, 0, -1):
        ss = seg_starts[si - 1]; se = seg_starts[si]
        # Replay forward within segment
        theta = {n: v.clone() for n, v in saved[ss]['p'].items()}
        m_r = {n: v.clone() for n, v in saved[ss]['m'].items()}
        v_r = {n: v.clone() for n, v in saved[ss]['v'].items()}
        states = [{'p': theta, 'm': m_r, 'v': v_r}]
        for t in range(ss, se):
            with torch.no_grad():
                for n, p in trainable.items(): p.copy_(theta[n].to(DEVICE))
            model.train(); model.zero_grad()
            inp = train_ids[bi[t]].to(DEVICE); att = train_mask[bi[t]].to(DEVICE); lab = train_labels[bi[t]].to(DEVICE)
            loss = compute_loss(None, inp, att, lab); loss.backward()
            lr_t = get_lr(t, total_steps)
            nt = {}; nm = {}; nv = {}
            for n, p in trainable.items():
                if p.grad is None: continue
                g = p.grad.cpu()
                nm[n] = BETA1 * m_r[n] + (1 - BETA1) * g
                nv[n] = BETA2 * v_r[n] + (1 - BETA2) * g * g
                dn = torch.sqrt(nv[n] + EPS_ROOT) + EPS
                nt[n] = theta[n] - lr_t * nm[n] / dn - lr_t * WD * theta[n]
            theta = nt; m_r = nm; v_r = nv
            states.append({'p': theta, 'm': m_r, 'v': v_r})

        # VJP backward through segment
        for k in range(len(states) - 2, -1, -1):
            t = ss + k
            s = states[k]
            pl = {n: s['p'][n].to(DEVICE).detach().requires_grad_(True) for n in param_names}
            ml = {n: s['m'][n].to(DEVICE).detach().requires_grad_(True) for n in param_names}
            vl = {n: s['v'][n].to(DEVICE).detach().requires_grad_(True) for n in param_names}
            w = torch.ones(BATCH_SIZE, device=DEVICE, requires_grad=True)
            inp = train_ids[bi[t]].to(DEVICE); att = train_mask[bi[t]].to(DEVICE); lab = train_labels[bi[t]].to(DEVICE)
            wl = compute_loss(pl, inp, att, lab, weights=w, use_fc=True)
            plist = list(pl.values()); names = list(pl.keys())
            grads_fc = torch.autograd.grad(wl, plist, create_graph=True, allow_unused=True)
            gd = {n: (grads_fc[i] if grads_fc[i] is not None else torch.zeros_like(pl[n])) for i, n in enumerate(names)}
            lr_t = get_lr(t, total_steps)
            np_, nm_, nv_ = diff_adam_step(pl, ml, vl, gd, lr_t)
            A = torch.tensor(0.0, device=DEVICE)
            for n in names:
                A = A + (np_[n] * delta_p[n].to(DEVICE).detach()).sum()
                A = A + (nm_[n] * delta_m[n].to(DEVICE).detach()).sum()
                A = A + (nv_[n] * delta_v[n].to(DEVICE).detach()).sum()
            all_leaves = plist + list(ml.values()) + list(vl.values()) + [w]
            ag = torch.autograd.grad(A, all_leaves, allow_unused=True)
            np_l = len(names)
            for i, n in enumerate(names):
                delta_p[n] = ag[i].cpu().detach() if ag[i] is not None else delta_p[n]
                delta_m[n] = ag[np_l+i].cpu().detach() if ag[np_l+i] is not None else delta_m[n]
                delta_v[n] = ag[2*np_l+i].cpu().detach() if ag[2*np_l+i] is not None else delta_v[n]
            beta = ag[-1].cpu().detach() if ag[-1] is not None else torch.zeros(BATCH_SIZE)
            for i, idx in enumerate(bi[t]):
                infl[idx] += beta[i].item()
            del pl, ml, vl, w, wl, grads_fc, gd, np_, nm_, nv_, A, ag, plist
        del states; gc.collect(); torch.cuda.empty_cache()

        steps_done = total_steps - seg_starts[si]
        if steps_done > 0 and steps_done % 100 == 0:
            dn = sum(v.norm().item()**2 for v in delta_p.values())**0.5
            elapsed_t = time.time() - t0
            rate = elapsed_t / steps_done
            eta = rate * (total_steps - steps_done)
            print("    %d/%d dt=%.2e %.1fs/step ETA %.0fmin" % (steps_done, total_steps, dn, rate, eta/60), flush=True)

    influences[j] = infl.numpy()
    has_nan = np.isnan(influences[j]).any()
    dn = sum(v.norm().item()**2 for v in delta_p.values())**0.5
    print("  Test %d: %ds loss=%.4f inf=[%.6f,%.6f] dt=%.2e nan=%s" % (
        j, time.time()-t0, base_losses[j], influences[j].min(), influences[j].max(), dn, has_nan), flush=True)

# ===== FD CHECK =====
print("\nFD validation:", flush=True)
eps = 1e-3
for si in [0, 500, 5000]:
    if not any(si in bi[t] for t in range(total_steps)): continue
    def train_fd(pidx, ev):
        torch.manual_seed(SEED)
        m2 = get_peft_model(AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b", torch_dtype=torch.float32, device_map=DEVICE, attn_implementation="eager"),
            LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        # Copy initial weights
        with torch.no_grad():
            for n, p in m2.named_parameters():
                if n in saved[0]['p']: p.copy_(saved[0]['p'][n].to(DEVICE))
        tr2 = {n: p for n, p in m2.named_parameters() if p.requires_grad}
        ms2 = {n: torch.zeros_like(p, device='cpu') for n, p in tr2.items()}
        vs2 = {n: torch.zeros_like(p, device='cpu') for n, p in tr2.items()}
        sw = torch.ones(N); sw[pidx] += ev
        m2.train()
        for step in range(total_steps):
            idx = bi[step]; lr_t = get_lr(step, total_steps)
            inp = train_ids[idx].to(DEVICE); att = train_mask[idx].to(DEVICE); lab = train_labels[idx].to(DEVICE)
            wb = sw[idx].to(DEVICE)
            loss = compute_loss(None, inp, att, lab, weights=wb)
            m2.zero_grad(); loss.backward()
            with torch.no_grad():
                for n, p in tr2.items():
                    if p.grad is None: continue
                    g = p.grad
                    ms2[n] = BETA1*ms2[n].to(DEVICE) + (1-BETA1)*g
                    vs2[n] = BETA2*vs2[n].to(DEVICE) + (1-BETA2)*g*g
                    dn = torch.sqrt(vs2[n]+EPS_ROOT)+EPS
                    p.add_(ms2[n]/dn, alpha=-lr_t); p.add_(p, alpha=-lr_t*WD)
                    ms2[n]=ms2[n].cpu(); vs2[n]=vs2[n].cpu()
        m2.eval()
        with torch.no_grad():
            params_fd = {n: p for n, p in m2.named_parameters() if p.requires_grad}
            tl = test_loss_single(params_fd, 0).item()
        del m2; gc.collect(); torch.cuda.empty_cache()
        return tl
    l0 = train_fd(si, 0.0); l1 = train_fd(si, eps)
    fd = (l1 - l0) / eps; vjp = influences[0, si]
    ratio = vjp / fd if abs(fd) > 1e-10 else float('inf')
    print("  Sample %d: FD=%.6f VJP=%.6f ratio=%.4f" % (si, fd, vjp, ratio), flush=True)

# ===== COUNTERFACTUAL =====
print("\nCounterfactual (%d subsets, drop %d%%)..." % (NUM_CF, int(DROP_FRAC*100)), flush=True)
nd = int(N * DROP_FRAC); rng = np.random.RandomState(SEED + 1000)
masks = np.ones((NUM_CF, N)); true_losses = np.zeros((NUM_CF, NUM_TEST))

for s in range(NUM_CF):
    pc = rng.permutation(N); masks[s, pc[:nd]] = 0
    sw = torch.ones(N); sw[pc[:nd]] = 0
    torch.manual_seed(SEED)
    m2 = get_peft_model(AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b", torch_dtype=torch.float32, device_map=DEVICE, attn_implementation="eager"),
        LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for n, p in m2.named_parameters():
            if n in saved[0]['p']: p.copy_(saved[0]['p'][n].to(DEVICE))
    tr2 = {n: p for n, p in m2.named_parameters() if p.requires_grad}
    ms2 = {n: torch.zeros_like(p, device='cpu') for n, p in tr2.items()}
    vs2 = {n: torch.zeros_like(p, device='cpu') for n, p in tr2.items()}
    m2.train()
    for step in range(total_steps):
        idx = bi[step]; lr_t = get_lr(step, total_steps)
        inp = train_ids[idx].to(DEVICE); att = train_mask[idx].to(DEVICE); lab = train_labels[idx].to(DEVICE)
        wb = sw[idx].to(DEVICE)
        loss = compute_loss(None, inp, att, lab, weights=wb)
        m2.zero_grad(); loss.backward()
        with torch.no_grad():
            for n, p in tr2.items():
                if p.grad is None: continue
                g = p.grad
                ms2[n] = BETA1*ms2[n].to(DEVICE)+(1-BETA1)*g
                vs2[n] = BETA2*vs2[n].to(DEVICE)+(1-BETA2)*g*g
                dn = torch.sqrt(vs2[n]+EPS_ROOT)+EPS
                p.add_(ms2[n]/dn, alpha=-lr_t); p.add_(p, alpha=-lr_t*WD)
                ms2[n]=ms2[n].cpu(); vs2[n]=vs2[n].cpu()
    m2.eval()
    with torch.no_grad():
        params_cf = {n: p for n, p in m2.named_parameters() if p.requires_grad}
        for j2 in range(NUM_TEST):
            true_losses[s, j2] = test_loss_single(params_cf, j2).item()
    del m2; gc.collect(); torch.cuda.empty_cache()
    if (s + 1) % 5 == 0:
        print("  %d/%d" % (s+1, NUM_CF), flush=True)

# ===== LDS =====
print("\n" + "=" * 60, flush=True)
print("MAGIC Gemma-2B LoRA LDS (drop %d%%)" % int(DROP_FRAC*100), flush=True)
print("=" * 60, flush=True)
dw = masks - 1; all_r = []
for j in range(NUM_TEST):
    pred = base_losses[j] + dw @ influences[j]
    r, _ = spearmanr(pred, true_losses[:, j])
    all_r.append(r)
    print("  Test %d: LDS=%.4f" % (j, r), flush=True)
mean_lds = np.nanmean(all_r)
print("\nMean LDS: %.4f" % mean_lds, flush=True)
print("Paper reference: ~0.97 (drop 1%%), ~0.90 (drop 5%%)" % (), flush=True)
print("=" * 60, flush=True)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.bar(['Gemma LoRA\n(ours)', 'Paper\nGemma', 'Paper\nResNet9'],
       [mean_lds, 0.905, 0.922], color=['#ef8632', 'gray', 'gray'], alpha=[1.0, 0.5, 0.5])
ax.set_ylabel('LDS'); ax.set_title('MAGIC LDS (drop 5%%)')
ax.set_ylim(-0.2, 1.1); ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('/home/mac/infusion/MAGIC/gpt2_lds/output/gemma_lds.png', dpi=150)
print("Plot saved!", flush=True)
