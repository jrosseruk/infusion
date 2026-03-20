"""Step 2: GPU sparse dictionary learning with FISTA and dead atom resampling.

Scalable to 16K+ atoms — avoids K×K matrix by using residual-based gradients.

Usage:
    python smolLM2/gpu_dictionary.py
    python smolLM2/gpu_dictionary.py --n_atoms 16000 --alpha 0.05 --n_epochs 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atoms_config import (
    DEAD_THRESHOLD,
    DL_BATCH_SIZE,
    DL_LEARNING_RATE,
    FISTA_ITERS_FINAL,
    FISTA_ITERS_TRAIN,
    N_ATOMS,
    N_EPOCHS,
    OUTPUT_DIR,
    RESAMPLE_INTERVAL,
    SEED,
    SPARSITY_PENALTY,
)


class GPUDictionaryLearning:
    """Online dictionary learning with FISTA sparse coding and dead atom resampling.

    Scalable: avoids K×K DtD matrix. FISTA gradient computed via residuals:
        grad = (Z @ D - X) @ D.T    [O(B*K*d) instead of O(B*K^2)]
    """

    def __init__(self, n_atoms: int, input_dim: int, alpha: float, device: str,
                 lr: float = 1e-3, seed: int = 42):
        torch.manual_seed(seed)
        self.n_atoms = n_atoms
        self.input_dim = input_dim
        self.alpha = alpha
        self.lr = lr
        self.device = device

        # Dictionary: (n_atoms, input_dim), unit-norm rows
        self.D = torch.randn(n_atoms, input_dim, device=device, dtype=torch.float32)
        self.D = F.normalize(self.D, dim=1)

        # Lipschitz constant (updated periodically via power iteration)
        self._L = None
        self._L_update_interval = 50  # recompute every N batches
        self._batch_count = 0

        # Activation tracking
        self.epoch_counts = torch.zeros(n_atoms, device=device, dtype=torch.long)
        self.interval_counts = torch.zeros(n_atoms, device=device, dtype=torch.long)
        self.activation_history = []

    def _estimate_lipschitz(self, n_iter: int = 20) -> float:
        """Estimate max eigenvalue of D @ D.T via power iteration (avoids forming K×K)."""
        v = torch.randn(self.n_atoms, device=self.device, dtype=torch.float32)
        v = v / v.norm()
        for _ in range(n_iter):
            # D @ D.T @ v = D @ (D.T @ v)
            Dtv = self.D.T @ v          # (d,)
            DDtv = self.D @ Dtv         # (K,)
            eigenval = v @ DDtv
            v = DDtv / (DDtv.norm() + 1e-10)
        return eigenval.item()

    def _get_lipschitz(self) -> float:
        """Get cached Lipschitz constant, recomputing periodically."""
        self._batch_count += 1
        if self._L is None or self._batch_count % self._L_update_interval == 0:
            self._L = self._estimate_lipschitz()
        return self._L

    def sparse_code(self, X: torch.Tensor, n_iter: int = 100) -> torch.Tensor:
        """FISTA sparse coding: min ||X - Z@D||^2 + alpha*||Z||_1.

        Uses residual-based gradient: grad = (Z@D - X) @ D.T
        Avoids forming K×K matrix. Cost: O(B*K*d) per iteration.
        """
        B = X.shape[0]
        K = self.n_atoms

        L = self._get_lipschitz()
        step = 1.0 / L
        threshold = step * self.alpha

        Z = torch.zeros(B, K, device=self.device, dtype=torch.float32)
        Z_prev = Z.clone()
        t = 1.0

        for _ in range(n_iter):
            t_new = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            momentum = (t - 1.0) / t_new
            Y = Z + momentum * (Z - Z_prev)

            # Gradient via residual: grad = (Y@D - X) @ D.T
            residual = Y @ self.D - X       # (B, d)
            grad = residual @ self.D.T       # (B, K)
            Y_step = Y - step * grad

            # Soft thresholding
            Z_prev = Z
            Z = torch.sign(Y_step) * torch.clamp(Y_step.abs() - threshold, min=0)
            t = t_new

        return Z

    def update_dictionary(self, X: torch.Tensor, Z: torch.Tensor):
        """Gradient descent on dictionary with unit-norm projection."""
        residual = X - Z @ self.D
        grad_D = -(Z.T @ residual) / X.shape[0]
        self.D -= self.lr * grad_D
        self.D = F.normalize(self.D, dim=1)

    def resample_dead_atoms(self, X: torch.Tensor, Z: torch.Tensor,
                             dead_threshold: int = 5) -> int:
        """Reinitialize dead atoms from high-reconstruction-error data points."""
        dead_mask = self.interval_counts < dead_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0

        # Limit resampling to avoid destabilization
        max_resample = min(n_dead, max(self.n_atoms // 10, 500))
        if n_dead > max_resample:
            dead_indices_all = torch.where(dead_mask)[0]
            perm = torch.randperm(n_dead, device=self.device)[:max_resample]
            dead_indices = dead_indices_all[perm]
            n_dead = max_resample
        else:
            dead_indices = torch.where(dead_mask)[0]

        # Reconstruction error per sample
        recon = Z @ self.D
        errors = (X - recon).pow(2).sum(dim=1)
        probs = errors / (errors.sum() + 1e-10)
        donor_idx = torch.multinomial(probs, n_dead, replacement=True)

        donors = X[donor_idx]
        noise = torch.randn_like(donors) * 0.01
        self.D[dead_indices] = F.normalize(donors + noise, dim=1)
        self.interval_counts[dead_indices] = 0

        # Reset Lipschitz estimate after resampling
        self._L = None

        return n_dead

    def save_checkpoint(self, path: str, epoch: int, step: int):
        """Save training checkpoint."""
        torch.save({
            "dictionary": self.D.cpu(),
            "epoch": epoch,
            "step": step,
            "n_atoms": self.n_atoms,
            "alpha": self.alpha,
            "lr": self.lr,
            "activation_history": self.activation_history,
            "epoch_counts": self.epoch_counts.cpu(),
            "interval_counts": self.interval_counts.cpu(),
        }, path)

    def load_checkpoint(self, path: str) -> tuple[int, int]:
        """Load training checkpoint. Returns (epoch, step)."""
        data = torch.load(path, weights_only=True, map_location=self.device)
        self.D = data["dictionary"].to(self.device)
        self.activation_history = data.get("activation_history", [])
        self.epoch_counts = data.get("epoch_counts",
                                      torch.zeros(self.n_atoms)).to(self.device)
        self.interval_counts = data.get("interval_counts",
                                         torch.zeros(self.n_atoms)).to(self.device)
        return int(data["epoch"]), int(data["step"])


def load_shards(grad_dir: str) -> list[dict]:
    """Load shard metadata (paths + sizes) without loading tensors."""
    shards = []
    for fname in sorted(os.listdir(grad_dir)):
        if fname.startswith("shard_") and fname.endswith(".pt"):
            path = os.path.join(grad_dir, fname)
            shards.append({"path": path, "name": fname})
    return shards


def stream_batches(shards: list[dict], batch_size: int, device: str,
                   seed: int = 0):
    """Yield batches by streaming through shards (shuffled order, shuffled rows)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    shard_order = torch.randperm(len(shards), generator=rng).tolist()

    for si in shard_order:
        data = torch.load(shards[si]["path"], weights_only=True, map_location="cpu")
        G = data["projected_gradients"]
        n = G.shape[0]

        # L2-normalize rows
        norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G = G / norms

        # Shuffle rows within shard
        perm = torch.randperm(n, generator=rng)
        G = G[perm]

        for start in range(0, n, batch_size):
            batch = G[start:start + batch_size].to(device)
            if batch.shape[0] < 16:
                continue
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_atoms", type=int, default=N_ATOMS)
    parser.add_argument("--alpha", type=float, default=SPARSITY_PENALTY)
    parser.add_argument("--batch_size", type=int, default=DL_BATCH_SIZE)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=DL_LEARNING_RATE)
    parser.add_argument("--fista_iters", type=int, default=FISTA_ITERS_TRAIN)
    parser.add_argument("--fista_iters_final", type=int, default=FISTA_ITERS_FINAL)
    parser.add_argument("--resample_interval", type=int, default=RESAMPLE_INTERVAL)
    parser.add_argument("--dead_threshold", type=int, default=DEAD_THRESHOLD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (creates subdirectory)")
    args = parser.parse_args()

    # Support named runs (e.g. --run_name 16k_alpha005)
    if args.run_name:
        run_dir = os.path.join(args.output_dir, f"run_{args.run_name}")
    else:
        run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)

    grad_dir = os.path.join(args.output_dir, "projected_gradients")
    shards = load_shards(grad_dir)
    print(f"Found {len(shards)} gradient shards in {grad_dir}", flush=True)

    # Determine input_dim from first shard
    first_shard = torch.load(shards[0]["path"], weights_only=True, map_location="cpu")
    input_dim = first_shard["projected_gradients"].shape[1]
    print(f"Input dim: {input_dim}, Atoms: {args.n_atoms}, Alpha: {args.alpha}",
          flush=True)
    del first_shard

    # Initialize or resume
    dl = GPUDictionaryLearning(
        n_atoms=args.n_atoms, input_dim=input_dim, alpha=args.alpha,
        device=args.device, lr=args.lr, seed=SEED,
    )

    ckpt_path = os.path.join(run_dir, "atoms_checkpoint.pt")
    start_epoch, start_step = 0, 0
    if args.resume and os.path.exists(ckpt_path):
        start_epoch, start_step = dl.load_checkpoint(ckpt_path)
        print(f"Resumed from epoch {start_epoch}, step {start_step}", flush=True)

    # ── Training loop ──
    print(f"\nTraining {args.n_atoms} atoms for {args.n_epochs} epochs "
          f"(FISTA iters={args.fista_iters})...", flush=True)
    global_step = start_step

    for epoch in range(start_epoch, args.n_epochs):
        epoch_t0 = time.time()
        dl.epoch_counts.zero_()
        dl.interval_counts.zero_()
        epoch_loss = 0.0
        epoch_batches = 0
        total_resampled = 0

        for batch in stream_batches(shards, args.batch_size, args.device,
                                     seed=SEED + epoch):
            Z = dl.sparse_code(batch, n_iter=args.fista_iters)

            # Track activations
            active = (Z.abs() > 1e-6).any(dim=0)
            dl.epoch_counts += active.long()
            dl.interval_counts += active.long()

            # Update dictionary
            dl.update_dictionary(batch, Z)

            # Reconstruction loss for logging
            recon = Z @ dl.D
            batch_loss = (batch - recon).pow(2).sum().item()
            epoch_loss += batch_loss
            epoch_batches += 1
            global_step += 1

            # Dead atom resampling
            if global_step % args.resample_interval == 0:
                n_resampled = dl.resample_dead_atoms(
                    batch, Z, dead_threshold=args.dead_threshold)
                if n_resampled > 0:
                    print(f"  Step {global_step}: resampled {n_resampled} dead atoms",
                          flush=True)
                total_resampled += n_resampled
                dl.interval_counts.zero_()

            # Progress within epoch
            if epoch_batches % 100 == 0:
                elapsed = time.time() - epoch_t0
                avg_loss = epoch_loss / epoch_batches
                n_active_so_far = (dl.epoch_counts > 0).sum().item()
                print(f"  Epoch {epoch} batch {epoch_batches}: "
                      f"avg_loss={avg_loss:.2f}, "
                      f"active={n_active_so_far}/{args.n_atoms}, "
                      f"time={elapsed:.0f}s", flush=True)

        # Epoch stats
        n_active = (dl.epoch_counts > 0).sum().item()
        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - epoch_t0
        dl.activation_history.append({
            "epoch": epoch,
            "n_active": n_active,
            "avg_recon_loss": avg_loss,
            "n_resampled": total_resampled,
        })

        print(f"Epoch {epoch}/{args.n_epochs}: "
              f"{n_active}/{args.n_atoms} active atoms, "
              f"avg_loss={avg_loss:.2f}, "
              f"resampled={total_resampled}, "
              f"time={elapsed:.0f}s", flush=True)

        # Checkpoint every epoch
        dl.save_checkpoint(ckpt_path, epoch + 1, global_step)
        print(f"  Checkpoint saved -> {ckpt_path}", flush=True)

    # ── Final transform: compute sparse coefficients for all docs ──
    print("\nComputing final coefficients for all docs...", flush=True)
    coeff_dir = os.path.join(run_dir, "coefficients")
    os.makedirs(coeff_dir, exist_ok=True)

    # Compute explained variance
    total_sq_norm = 0.0
    total_recon_error = 0.0

    for si, shard_info in enumerate(shards):
        data = torch.load(shard_info["path"], weights_only=True, map_location="cpu")
        G = data["projected_gradients"]
        indices = data["indices"]

        norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G = G / norms

        n = G.shape[0]

        # For large K, store sparse coefficients (indices + values)
        all_coeff_indices = []
        all_coeff_values = []
        n_active_docs = 0

        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch = G[start:end].to(args.device)
            Z = dl.sparse_code(batch, n_iter=args.fista_iters_final)

            recon = Z @ dl.D
            total_sq_norm += batch.pow(2).sum().item()
            total_recon_error += (batch - recon).pow(2).sum().item()

            # Store sparse
            Z_cpu = Z.cpu()
            for row_idx in range(Z_cpu.shape[0]):
                nonzero = Z_cpu[row_idx].abs() > 1e-6
                nz_indices = torch.where(nonzero)[0].to(torch.int32)
                nz_values = Z_cpu[row_idx, nonzero]
                all_coeff_indices.append(nz_indices)
                all_coeff_values.append(nz_values)
                if len(nz_indices) > 0:
                    n_active_docs += 1

        coeff_path = os.path.join(coeff_dir, f"shard_{si:04d}.pt")
        torch.save({
            "coeff_indices": all_coeff_indices,
            "coeff_values": all_coeff_values,
            "doc_indices": indices,
            "n_atoms": args.n_atoms,
        }, coeff_path)

        print(f"  Shard {si}/{len(shards)}: {n_active_docs}/{n} docs active -> {coeff_path}",
              flush=True)

    explained_var = 1.0 - total_recon_error / total_sq_norm
    print(f"\nExplained variance: {explained_var:.4f} ({explained_var*100:.2f}%)", flush=True)

    # ── Save final atoms ──
    atoms_path = os.path.join(run_dir, "atoms.pt")
    torch.save({
        "dictionary": dl.D.cpu(),
        "n_atoms": args.n_atoms,
        "input_dim": input_dim,
        "alpha": args.alpha,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "fista_iters": args.fista_iters,
        "activation_history": dl.activation_history,
        "explained_variance": explained_var,
        "seed": SEED,
    }, atoms_path)
    print(f"Dictionary saved -> {atoms_path}", flush=True)
    print("Dictionary learning complete!", flush=True)


if __name__ == "__main__":
    main()
