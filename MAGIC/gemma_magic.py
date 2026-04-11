#!/usr/bin/env python
"""
MAGIC LDS on Gemma-2B LoRA fine-tuning for instruction following.

MAGIC paper Gemma-2B experiment:
- Model: google/gemma-2b (pretraining-only) + LoRA
- Data: ~300K IFT samples (Flan V2 + CoT + Dolly + OpenAssistant)
- Test: 32 MMLU samples (4-shot ICL)
- Optimizer: Adam β1=0.95, β2=0.975, WD=1e-5, eps_root=1e-6
- LR: max 0.0004, one-cycle (start 1e-6×max, peak 25%, end 0.1×max)
- LoRA config from LESS paper

For a quick first pass, we use a smaller subset and fewer CF subsets.
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.func import functional_call

DEVICE = 'cuda:0'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ===== Hyperparameters (MAGIC paper, Appendix A) =====
MAX_LR = 0.0004
BETA1 = 0.95
BETA2 = 0.975
WD = 1e-5
EPS_ROOT = 1e-6  # Inside sqrt for smoothness
EPS = 1e-8
LR_START_MULT = 1e-6
LR_END_MULT = 0.1
LR_PEAK_FRAC = 0.25

# Settings
MAX_TRAIN_SAMPLES = 15000  # Full Dolly dataset
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 1
NUM_TEST = 5
NUM_CF = 30
DROP_FRAC = 0.05

def get_lr(step, total):
    peak = int(total * LR_PEAK_FRAC)
    if step < peak:
        f = step / max(peak, 1)
        return MAX_LR * (LR_START_MULT + f * (1.0 - LR_START_MULT))
    else:
        f = (step - peak) / max(total - peak, 1)
        return MAX_LR * (1.0 - f * (1.0 - LR_END_MULT))


# ===== Load model + LoRA =====
print("Loading Gemma-2B...", flush=True)
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map=DEVICE,
    attn_implementation="eager",  # Needed for create_graph double backward
)

# LoRA config (LESS paper uses rank=64 or 128 on q,v projections)
lora_config = LoraConfig(
    r=128,  # LESS paper uses rank 128
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
print(flush=True)

# ===== Load IFT data (Dolly as quick proxy for full IFT mix) =====
print("Loading training data...", flush=True)
ds = load_dataset("databricks/databricks-dolly-15k", split="train")

def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    tokens["labels"] = tokens["input_ids"].clone()
    # Mask padding in labels
    tokens["labels"][tokens["attention_mask"] == 0] = -100
    return {k: v.squeeze(0) for k, v in tokens.items()}

ds = ds.shuffle(seed=SEED).select(range(min(MAX_TRAIN_SAMPLES, len(ds))))
tokenized = ds.map(tokenize, remove_columns=ds.column_names)
tokenized.set_format("torch")

# Pre-load as tensors
train_ids = torch.stack([tokenized[i]["input_ids"] for i in range(len(tokenized))])
train_mask = torch.stack([tokenized[i]["attention_mask"] for i in range(len(tokenized))])
train_labels = torch.stack([tokenized[i]["labels"] for i in range(len(tokenized))])
N = len(train_ids)
print(f"Train samples: {N}, seq_len: {MAX_SEQ_LEN}", flush=True)

# Test data (use last 5 samples of the dataset as proxy for MMLU)
test_ids = train_ids[-NUM_TEST:]
test_mask = train_mask[-NUM_TEST:]
test_labels = train_labels[-NUM_TEST:]
# Remove test from train
train_ids = train_ids[:-NUM_TEST]
train_mask = train_mask[:-NUM_TEST]
train_labels = train_labels[:-NUM_TEST]
N = len(train_ids)
print(f"After split: train={N}, test={NUM_TEST}", flush=True)

# Batch indices
steps_per_epoch = N // BATCH_SIZE
total_steps = steps_per_epoch * NUM_EPOCHS
print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}", flush=True)

torch.manual_seed(SEED)
bi = []
for e in range(NUM_EPOCHS):
    perm = torch.randperm(N).tolist()
    for i in range(0, N, BATCH_SIZE):
        b = perm[i:i+BATCH_SIZE]
        if len(b) == BATCH_SIZE:
            bi.append(b)
bi = bi[:total_steps]

# ===== Custom smooth Adam state =====
def get_trainable_params(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}

def compute_loss(model, input_ids, attention_mask, labels, weights=None):
    """Per-sample mean cross-entropy loss."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()
    B, T, V = shift_logits.shape
    per_token = F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1), reduction='none').reshape(B, T)
    # Mean over tokens per sample
    per_sample = (per_token * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
    if weights is not None:
        return (weights * per_sample).sum() / BATCH_SIZE
    return per_sample.mean()


# ===== Train with smooth Adam =====
print(f"\nTraining (Adam, lr={MAX_LR}, eps_root={EPS_ROOT}, {total_steps} steps)...", flush=True)
trainable = get_trainable_params(model)
param_names = list(trainable.keys())

# Adam state
m_state = {n: torch.zeros_like(p, device='cpu') for n, p in trainable.items()}
v_state = {n: torch.zeros_like(p, device='cpu') for n, p in trainable.items()}

# Save checkpoints every 10 steps
SAVE_EVERY = 10
saved = {}
saved[0] = {
    'p': {n: p.data.cpu().clone() for n, p in trainable.items()},
    'm': {n: v.clone() for n, v in m_state.items()},
    'v': {n: v.clone() for n, v in v_state.items()},
}

model.train()
for step in range(total_steps):
    idx = bi[step]
    input_ids = train_ids[idx].to(DEVICE)
    attention_mask = train_mask[idx].to(DEVICE)
    labels = train_labels[idx].to(DEVICE)
    lr_t = get_lr(step, total_steps)

    loss = compute_loss(model, input_ids, attention_mask, labels)
    model.zero_grad()
    loss.backward()

    # Smooth Adam update
    with torch.no_grad():
        for n, p in trainable.items():
            if p.grad is None:
                continue
            g = p.grad
            m_state[n] = BETA1 * m_state[n].to(DEVICE) + (1 - BETA1) * g
            v_state[n] = BETA2 * v_state[n].to(DEVICE) + (1 - BETA2) * g * g
            denom = torch.sqrt(v_state[n] + EPS_ROOT) + EPS
            p.add_(m_state[n] / denom, alpha=-lr_t)
            p.add_(p, alpha=-lr_t * WD)  # Decoupled weight decay
            m_state[n] = m_state[n].cpu()
            v_state[n] = v_state[n].cpu()

    if (step + 1) % SAVE_EVERY == 0 or step == total_steps - 1:
        saved[step + 1] = {
            'p': {n: p.data.cpu().clone() for n, p in trainable.items()},
            'm': {n: v.cpu().clone() for n, v in m_state.items()},
            'v': {n: v.cpu().clone() for n, v in v_state.items()},
        }

    if (step + 1) % 100 == 0 or step == total_steps - 1:
        print(f"  Step {step+1}/{total_steps} lr={lr_t:.6f} loss={loss.item():.4f}", flush=True)

    # Hessian check every 250 steps
    if (step + 1) % 250 == 0:
        print(f"  [Hessian check at step {step+1}]", flush=True)
        model.eval()
        plist = [p for n, p in trainable.items() if p.requires_grad]
        vh = [torch.randn_like(p) for p in plist]
        vhn = sum((vi**2).sum() for vi in vh)**0.5
        vh = [vi / vhn for vi in vh]
        for _it in range(15):
            model.zero_grad()
            _loss = compute_loss(model, input_ids, attention_mask, labels)
            _grads = torch.autograd.grad(_loss, plist, create_graph=True)
            _dot = sum((g * vi).sum() for g, vi in zip(_grads, vh))
            _hvp = torch.autograd.grad(_dot, plist)
            _lam = sum((h * vi).sum().item() for h, vi in zip(_hvp, vh))
            _hn = sum((h**2).sum().item() for h in _hvp)**0.5
            vh = [h.detach() / max(_hn, 1e-10) for h in _hvp]
        print(f"    lambda_max={_lam:.4f}, lr*lambda={lr_t*_lam:.6f} {'STABLE' if lr_t*abs(_lam)<2 else 'UNSTABLE'}", flush=True)
        model.train()

print(f"Checkpoints: {len(saved)}", flush=True)

# ===== Quick FD sanity check =====
print("\n--- FD Sanity Check ---", flush=True)

# Compute test loss gradient (Delta_T) for test sample 0
model.eval()
model.zero_grad()
for p in model.parameters():
    p.requires_grad_(True)

test_input = test_ids[0:1].to(DEVICE)
test_attn = test_mask[0:1].to(DEVICE)
test_lab = test_labels[0:1].to(DEVICE)
test_loss = compute_loss(model, test_input, test_attn, test_lab)
test_loss.backward()

delta_p = {n: p.grad.cpu().clone() for n, p in trainable.items() if p.grad is not None}
delta_m = {n: torch.zeros_like(p, device='cpu') for n, p in trainable.items()}
delta_v = {n: torch.zeros_like(p, device='cpu') for n, p in trainable.items()}
base_loss = test_loss.item()
print(f"Test 0 base loss: {base_loss:.4f}", flush=True)
print(f"Delta norm: {sum(v.norm().item()**2 for v in delta_p.values())**0.5:.6f}", flush=True)
del test_loss

# Run 20 VJP steps backward (quick check)
print(f"Running 20 VJP steps (from step {total_steps})...", flush=True)
from gpt2_lds.replay import _differentiable_adam_step

# Quick VJP: just check dt_norm stability
for t_back in range(min(20, total_steps)):
    t = total_steps - 1 - t_back
    seg_key = ((t + 1) // SAVE_EVERY) * SAVE_EVERY
    if seg_key not in saved:
        seg_key = max(k for k in saved.keys() if k <= t + 1)
    # Skip full VJP for now, just trace dt_norm
    # (Full VJP needs functional_call which is complex with PEFT models)

dn = sum(v.norm().item()**2 for v in delta_p.values())**0.5
print(f"Initial delta norm: {dn:.6f}", flush=True)
print("(Full VJP + FD check requires functional_call with PEFT - implementing...)", flush=True)

# For now, just measure the Hessian eigenvalue to verify stability
print("\n--- Hessian Eigenvalue ---", flush=True)
params_list = [p for n, p in trainable.items() if p.requires_grad]
v_hess = [torch.randn_like(p) for p in params_list]
vn = sum((vi**2).sum() for vi in v_hess)**0.5
v_hess = [vi / vn for vi in v_hess]

model.train()
batch_idx = bi[0]
x_h = train_ids[batch_idx].to(DEVICE)
a_h = train_mask[batch_idx].to(DEVICE)
l_h = train_labels[batch_idx].to(DEVICE)

for it in range(20):
    model.zero_grad()
    loss = compute_loss(model, x_h, a_h, l_h)
    grads = torch.autograd.grad(loss, params_list, create_graph=True)
    dot = sum((g * vi).sum() for g, vi in zip(grads, v_hess))
    hvp = torch.autograd.grad(dot, params_list)
    lam = sum((h * vi).sum().item() for h, vi in zip(hvp, v_hess))
    hn = sum((h**2).sum().item() for h in hvp)**0.5
    v_hess = [h.detach() / max(hn, 1e-10) for h in hvp]
    if (it + 1) % 5 == 0:
        print(f"  iter {it+1}: lambda={lam:.4f}, lr*lambda={MAX_LR*lam:.6f}", flush=True)

print(f"\nlambda_max = {lam:.4f}", flush=True)
print(f"lr * lambda = {MAX_LR * lam:.6f}", flush=True)
print(f"Stable: {'YES' if MAX_LR * abs(lam) < 2 else 'NO'}", flush=True)
print(f"Max stable lr: {2/max(abs(lam), 0.001):.6f}", flush=True)

print("\nDone! Check stability before proceeding to full LDS.", flush=True)
