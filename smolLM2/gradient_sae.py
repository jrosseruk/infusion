"""TopK Sparse Autoencoder on projected gradients with AuxK dead feature recovery.

Architecture: TopK SAE (Gao et al. 2024, "Scaling and Evaluating Sparse Autoencoders")
Dead features: AuxK auxiliary loss (trains dead features on reconstruction residual)
Multi-GPU: PyTorch DDP

Usage:
    torchrun --nproc_per_node=8 smolLM2/gradient_sae.py --n_features 16384 --k 64
    python smolLM2/gradient_sae.py --n_features 4096 --k 32 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atoms_config import OUTPUT_DIR, SEED


# ── Model ──


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder with AuxK dead feature recovery.

    - Encoder: x → pre_acts → top-K ReLU → sparse codes z
    - Decoder: z → reconstruction x_hat (unit-norm decoder columns)
    - AuxK: dead features get trained via auxiliary reconstruction of residual
    """

    def __init__(self, d_input: int, n_features: int, k: int, k_aux: int = 256,
                 dead_threshold: int = 10_000_000):
        super().__init__()
        self.d_input = d_input
        self.n_features = n_features
        self.k = k
        self.k_aux = min(k_aux, n_features - k)  # can't exceed dead features
        self.dead_threshold = dead_threshold

        # Encoder
        self.W_enc = nn.Linear(d_input, n_features, bias=True)

        # Decoder (no bias — use separate b_dec for centering)
        self.W_dec = nn.Linear(n_features, d_input, bias=False)

        # Decoder bias (pre-encoder centering)
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        # Track feature activation counts for dead feature detection
        self.register_buffer("feature_counts", torch.zeros(n_features, dtype=torch.long))
        self.register_buffer("total_steps", torch.tensor(0, dtype=torch.long))

        # Initialize
        self._init_weights()

    def _init_weights(self):
        # Kaiming init for encoder
        nn.init.kaiming_uniform_(self.W_enc.weight)
        nn.init.zeros_(self.W_enc.bias)

        # Initialize decoder columns to unit norm
        nn.init.kaiming_uniform_(self.W_dec.weight)
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=1)

    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder columns to unit norm (constraint, not loss)."""
        self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=1)

    def _topk_activation(self, pre_acts: torch.Tensor, k: int) -> torch.Tensor:
        """Apply TopK: keep only top-k pre-activations per sample, ReLU the rest to 0."""
        topk_vals, topk_idx = pre_acts.topk(k, dim=-1)
        z = torch.zeros_like(pre_acts)
        z.scatter_(1, topk_idx, F.relu(topk_vals))
        return z

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to sparse codes. Returns (z, pre_acts)."""
        x_centered = x - self.b_dec
        pre_acts = self.W_enc(x_centered)
        z = self._topk_activation(pre_acts, self.k)
        return z, pre_acts

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse codes to reconstruction."""
        return self.W_dec(z) + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass with reconstruction and AuxK losses."""
        z, pre_acts = self.encode(x)
        x_hat = self.decode(z)

        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Track feature activations
        with torch.no_grad():
            active = (z > 0).any(dim=0)
            self.feature_counts += active.long()
            self.total_steps += 1

        # AuxK loss: train dead features on the reconstruction residual
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.k_aux > 0 and self.training:
            dead_mask = self.feature_counts < self.dead_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                # Dead feature pre-activations on the residual
                residual = x - x_hat.detach()  # detach to not affect main loss
                residual_centered = residual - self.b_dec
                dead_pre_acts = self.W_enc(residual_centered)

                # Zero out alive features
                dead_pre_acts[:, ~dead_mask] = -float("inf")

                # TopK among dead features only
                k_aux_actual = min(self.k_aux, n_dead)
                z_aux = self._topk_activation(dead_pre_acts, k_aux_actual)

                # Reconstruct residual from dead features
                residual_hat = self.W_dec(z_aux)
                aux_loss = (residual - residual_hat).pow(2).sum(dim=-1).mean()

        return {
            "x_hat": x_hat,
            "z": z,
            "recon_loss": recon_loss,
            "aux_loss": aux_loss,
            "pre_acts": pre_acts,
        }

    def get_stats(self) -> dict:
        """Return training statistics."""
        n_dead = (self.feature_counts < self.dead_threshold).sum().item()
        n_alive = self.n_features - n_dead
        return {
            "n_alive": n_alive,
            "n_dead": n_dead,
            "pct_alive": n_alive / self.n_features * 100,
        }


# ── Dataset ──


class ProjectedGradientDataset(Dataset):
    """Memory-mapped dataset over projected gradient shards."""

    def __init__(self, grad_dir: str, norm_mode: str = "log"):
        self.grad_dir = grad_dir
        self.norm_mode = norm_mode
        self.shard_files = sorted(
            [f for f in os.listdir(grad_dir) if f.startswith("shard_") and f.endswith(".pt")])

        # Load all shards into memory (they fit)
        rows = []
        for f in self.shard_files:
            data = torch.load(os.path.join(grad_dir, f), weights_only=True, map_location="cpu")
            G = data["projected_gradients"].float()
            rows.append(G)

        self.data = torch.cat(rows, dim=0)

        # Apply normalization
        norms = self.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        if norm_mode == "unit":
            self.data = self.data / norms
        elif norm_mode == "log":
            log_norms = torch.log1p(norms)
            self.data = self.data / norms * log_norms
        # "none": keep raw

        print(f"Loaded {len(self.data)} docs, dim={self.data.shape[1]}, "
              f"norm_mode={norm_mode}", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Training ──


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--grad_dir", default=None,
                        help="Projected gradients dir (default: {output_dir}/projected_gradients)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_features", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64,
                        help="Number of active features per input")
    parser.add_argument("--k_aux", type=int, default=256,
                        help="Number of dead features for AuxK loss")
    parser.add_argument("--aux_weight", type=float, default=1.0 / 32,
                        help="Weight for AuxK loss")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--norm_mode", default="log", choices=["unit", "log", "none"])
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--dead_threshold", type=int, default=10_000_000,
                        help="Feature is 'dead' if activated fewer than this many times")
    args = parser.parse_args()

    # Multi-GPU setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.device is None:
        args.device = f"cuda:{local_rank}"
    is_main = local_rank == 0

    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # Output directory
    if args.run_name:
        run_dir = os.path.join(args.output_dir, f"sae_{args.run_name}")
    else:
        run_dir = os.path.join(args.output_dir,
                               f"sae_{args.n_features}f_{args.k}k")
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    grad_dir = args.grad_dir or os.path.join(args.output_dir, "projected_gradients")
    if is_main:
        print(f"Loading projected gradients from {grad_dir}...", flush=True)
    dataset = ProjectedGradientDataset(grad_dir, norm_mode=args.norm_mode)
    d_input = dataset.data.shape[1]

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=SEED)
    else:
        sampler = None

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    # Build model
    model = TopKSAE(
        d_input=d_input, n_features=args.n_features, k=args.k,
        k_aux=args.k_aux, dead_threshold=args.dead_threshold,
    ).to(args.device)

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs * len(loader))

    if is_main:
        print(f"\nTopK SAE: {d_input} → {args.n_features} features, K={args.k}, "
              f"K_aux={args.k_aux}", flush=True)
        print(f"Dataset: {len(dataset):,} docs, Batch: {args.batch_size}, "
              f"Epochs: {args.n_epochs}, GPUs: {world_size}", flush=True)
        print(f"Norm mode: {args.norm_mode}", flush=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}", flush=True)

    # ── Training loop ──
    best_loss = float("inf")

    for epoch in range(args.n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_t0 = time.time()
        epoch_recon = 0.0
        epoch_aux = 0.0
        epoch_batches = 0

        model.train()
        for batch in loader:
            batch = batch.to(args.device)

            out = model(batch)
            loss = out["recon_loss"] + args.aux_weight * out["aux_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Normalize decoder columns after each step
            with torch.no_grad():
                raw_model._normalize_decoder()

            epoch_recon += out["recon_loss"].item()
            epoch_aux += out["aux_loss"].item()
            epoch_batches += 1

            if is_main and epoch_batches % 50 == 0:
                stats = raw_model.get_stats()
                lr = scheduler.get_last_lr()[0]
                print(f"  E{epoch} B{epoch_batches}: "
                      f"recon={out['recon_loss'].item():.2f}, "
                      f"aux={out['aux_loss'].item():.2f}, "
                      f"alive={stats['n_alive']}/{args.n_features} "
                      f"({stats['pct_alive']:.0f}%), "
                      f"lr={lr:.2e}", flush=True)

        # Epoch stats
        avg_recon = epoch_recon / max(epoch_batches, 1)
        avg_aux = epoch_aux / max(epoch_batches, 1)
        elapsed = time.time() - epoch_t0
        stats = raw_model.get_stats()

        if is_main:
            print(f"Epoch {epoch}/{args.n_epochs}: "
                  f"recon={avg_recon:.2f}, aux={avg_aux:.2f}, "
                  f"alive={stats['n_alive']}/{args.n_features} "
                  f"({stats['pct_alive']:.0f}%), "
                  f"time={elapsed:.0f}s", flush=True)

            # Checkpoint
            ckpt = {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "d_input": d_input,
                "n_features": args.n_features,
                "k": args.k,
                "k_aux": args.k_aux,
                "norm_mode": args.norm_mode,
                "stats": stats,
            }
            torch.save(ckpt, os.path.join(run_dir, "checkpoint.pt"))

            if avg_recon < best_loss:
                best_loss = avg_recon
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    # ── Compute explained variance ──
    if is_main:
        print("\nComputing explained variance...", flush=True)
        model.eval()
        total_sq = 0.0
        total_err = 0.0

        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=args.batch_size,
                                     num_workers=4, pin_memory=True):
                batch = batch.to(args.device)
                out = raw_model(batch)
                total_sq += batch.pow(2).sum().item()
                total_err += (batch - out["x_hat"]).pow(2).sum().item()

        ev = 1.0 - total_err / total_sq
        print(f"Explained variance: {ev:.4f} ({ev*100:.2f}%)", flush=True)

        # Save final results
        results = {
            "dictionary": raw_model.W_dec.weight.data.cpu(),  # (n_features, d_input)
            "encoder": raw_model.W_enc.weight.data.cpu(),
            "encoder_bias": raw_model.W_enc.bias.data.cpu(),
            "b_dec": raw_model.b_dec.data.cpu(),
            "n_features": args.n_features,
            "k": args.k,
            "d_input": d_input,
            "explained_variance": ev,
            "norm_mode": args.norm_mode,
            "feature_counts": raw_model.feature_counts.cpu(),
            "stats": stats,
        }
        torch.save(results, os.path.join(run_dir, "sae_results.pt"))
        print(f"Results saved -> {run_dir}", flush=True)

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
