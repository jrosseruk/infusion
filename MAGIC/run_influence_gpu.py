#!/usr/bin/env python
"""Run influence computation with tqdm progress bar, logging to a shared file."""
import sys, os, time, argparse, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from torch.func import functional_call
from pathlib import Path
from tqdm import tqdm

from gpt2_lds.config import MagicConfig
from gpt2_lds.data import prepare_wikitext_datasets, TensorDataset, create_gpt2_model
from gpt2_lds.train import load_checkpoint, get_lr, compute_per_sample_loss
from gpt2_lds.replay import replay_vjp_step, replay_forward_segment, _differentiable_adam_step

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--test_indices", type=int, nargs="+", required=True)
parser.add_argument("--logfile", type=str, default="/home/mac/infusion/MAGIC/gpt2_lds/output/progress.log")
args = parser.parse_args()

cfg = MagicConfig()
device = f"cuda:{args.gpu}"
out_dir = cfg.output_dir
os.makedirs(out_dir, exist_ok=True)

def log(msg):
    line = f"[GPU{args.gpu}] {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(args.logfile, "a") as f:
        f.write(line + "\n")

# Load data
log("Loading data...")
train_hf, valid_hf = prepare_wikitext_datasets(cfg.block_size)
train_ds = TensorDataset(train_hf)
valid_ds = TensorDataset(valid_hf)

ckpt_dir = Path(cfg.checkpoint_dir)
batch_indices_all = torch.load(ckpt_dir / "batch_indices.pt", weights_only=False)
total_steps = len(batch_indices_all)

for test_idx in args.test_indices:
    out_file = f"{out_dir}/influence_test_{test_idx:03d}.pt"
    if os.path.exists(out_file):
        log(f"Test {test_idx} already computed, skipping")
        continue

    log(f"Computing influence for test sample {test_idx} ({total_steps} steps)...")
    t0 = time.time()

    num_train = len(train_ds)
    influence = torch.zeros(num_train, dtype=torch.float64)
    model = create_gpt2_model(eager_attention=True).to(device)

    # Load final model
    ckpt_steps = sorted([int(f.stem.split("_")[1]) for f in ckpt_dir.glob("step_*.pt")])
    final_ckpt = load_checkpoint(ckpt_dir, ckpt_steps[-1])
    final_params = final_ckpt["params"]

    # Compute Delta_T
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(final_params[name].to(device))
    model.eval()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    input_ids = valid_ds.input_ids[test_idx].unsqueeze(0).to(device)
    attention_mask = valid_ds.attention_mask[test_idx].unsqueeze(0).to(device)
    labels = valid_ds.labels[test_idx].unsqueeze(0).to(device)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    per_sample = compute_per_sample_loss(output.logits, labels, attention_mask)
    test_loss = per_sample.sum()
    test_loss.backward()

    delta_theta = {n: p.grad.detach().clone().double() for n, p in model.named_parameters()}
    delta_m = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in model.named_parameters()}
    delta_v = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in model.named_parameters()}
    base_test_loss = test_loss.item()

    del output, per_sample, test_loss, input_ids, attention_mask, labels

    # Segment boundaries
    segment_boundaries = []
    step = 0
    while step < total_steps:
        seg_end = min(step + cfg.checkpoint_every, total_steps)
        segment_boundaries.append((step, seg_end))
        step = seg_end

    # Process backward through segments
    num_segs = len(segment_boundaries)
    steps_done = 0
    for seg_idx, (seg_start, seg_end) in enumerate(reversed(segment_boundaries)):
        ckpt = load_checkpoint(ckpt_dir, seg_start)
        seg_len = seg_end - seg_start
        states = replay_forward_segment(
            model, ckpt, batch_indices_all, train_ds, seg_start, seg_end, cfg, device,
            keep_on_gpu=True,
        )

        for k in range(seg_len - 1, -1, -1):
            t = seg_start + k
            state = states[k]
            batch_idx = batch_indices_all[t]
            batch = train_ds.collate(batch_idx)
            lr_t = get_lr(t, total_steps, cfg)

            beta_t, delta_theta, delta_m, delta_v = replay_vjp_step(
                model, state["params"], state["m"], state["v"],
                batch, batch_idx,
                delta_theta, delta_m, delta_v,
                lr_t, cfg, device,
            )

            for i, idx in enumerate(batch_idx):
                influence[idx] += beta_t[i].item()

            steps_done += 1

        del states, ckpt
        gc.collect()
        torch.cuda.empty_cache()

        # Progress logging every 10 segments (~240 steps)
        if (seg_idx + 1) % 10 == 0 or seg_idx == num_segs - 1:
            elapsed_seg = time.time() - t0
            rate = elapsed_seg / steps_done
            remaining = rate * (total_steps - steps_done)
            pct = 100 * steps_done / total_steps
            log(f"  test={test_idx}: {pct:.0f}% ({steps_done}/{total_steps}), "
                f"{rate:.2f}s/step, ETA {remaining/60:.1f}min, "
                f"GPU {torch.cuda.memory_allocated(device)/1024**2:.0f}MB")
    elapsed = time.time() - t0

    log(f"Test {test_idx} done in {elapsed:.0f}s ({elapsed/60:.1f}min). "
        f"Loss={base_test_loss:.4f} Infl=[{influence.min():.4f},{influence.max():.4f}]")

    torch.save({"influence": influence, "base_loss": base_test_loss, "test_idx": test_idx}, out_file)
    del model, influence
    gc.collect()
    torch.cuda.empty_cache()

log("All done!")
