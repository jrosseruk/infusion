"""
One-time data preparation script. Run on HPC where results live.

Usage:
    python prepare_data.py --results-dir /scratch/s5e/jrosser.s5e/infusion/cifar/results/
    python prepare_data.py --umap    # Generate UMAP embedding + test gallery only

Produces files in demo/data/:
    - experiment_index.json   (~50KB)  All 2163 off-diagonal experiments
    - curated_experiments.npz (~8MB)   50 selected experiments with images + logits
    - aggregate_stats.json    (~5KB)   Overall stats, heatmap matrix, histogram bins
    - umap_embedding.npz      (~1MB)  UMAP of 5000 training images (for Live Attack page)
    - test_gallery.npz         (~2MB)  200 test images, 20 per class (for Live Attack page)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_experiment_log(results_dir):
    log_path = os.path.join(results_dir, "experiment_log.jsonl")
    experiments = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry["true_label"] != entry["target_class"]:
                experiments.append(entry)
    return experiments


def build_experiment_index(experiments):
    index = []
    for exp in experiments:
        index.append({
            "sample_idx": exp["sample_idx"],
            "test_image_idx": exp["test_image_idx"],
            "true_label": exp["true_label"],
            "target_class": exp["target_class"],
            "prob_target_orig": round(exp["prob_target_orig"], 6),
            "prob_target_infused": round(exp["prob_target_infused"], 6),
            "delta_prob": round(exp["delta_prob"], 6),
        })
    return index


def select_curated_experiments(experiments):
    """Select 50 diverse experiments for the curated set."""
    sorted_by_delta = sorted(experiments, key=lambda x: x["delta_prob"], reverse=True)

    selected = {}

    # 10 strongest attacks
    for exp in sorted_by_delta:
        key = (exp["sample_idx"], exp["test_image_idx"], exp["target_class"])
        if key not in selected and len([k for k in selected if "strongest" in selected[k]]) < 10:
            selected[key] = "strongest"

    # 10 covering each true class (one per class, strong)
    for true_class in range(10):
        for exp in sorted_by_delta:
            key = (exp["sample_idx"], exp["test_image_idx"], exp["target_class"])
            if key not in selected and exp["true_label"] == true_class:
                selected[key] = "per_class"
                break

    # 10 diverse class pairs
    desired_pairs = [
        (1, 8), (6, 3), (0, 8), (2, 0), (4, 7),
        (3, 5), (5, 3), (7, 4), (8, 9), (9, 6),
    ]
    for true_c, target_c in desired_pairs:
        for exp in sorted_by_delta:
            key = (exp["sample_idx"], exp["test_image_idx"], exp["target_class"])
            if key not in selected and exp["true_label"] == true_c and exp["target_class"] == target_c:
                selected[key] = "diverse_pair"
                break

    # 10 moderate attacks (delta_prob between 0.1 and 0.4)
    moderate = [e for e in sorted_by_delta if 0.1 < e["delta_prob"] < 0.4]
    count = 0
    for exp in moderate:
        key = (exp["sample_idx"], exp["test_image_idx"], exp["target_class"])
        if key not in selected:
            selected[key] = "moderate"
            count += 1
            if count >= 10:
                break

    # 10 weak/failed attacks (lowest delta_prob)
    sorted_by_delta_asc = sorted(experiments, key=lambda x: x["delta_prob"])
    count = 0
    for exp in sorted_by_delta_asc:
        key = (exp["sample_idx"], exp["test_image_idx"], exp["target_class"])
        if key not in selected:
            selected[key] = "weak"
            count += 1
            if count >= 10:
                break

    return selected


def load_experiment_data(results_dir, sample_idx, test_image_idx, target_class):
    """Load a single experiment's results.npz."""
    dirname = f"sample_{sample_idx:04d}_test_{test_image_idx}_target_{target_class}"
    npz_path = os.path.join(results_dir, dirname, "results.npz")
    if not os.path.exists(npz_path):
        return None
    return dict(np.load(npz_path))


def build_curated_npz(results_dir, experiments, selected_keys, output_path):
    """Build the curated experiments NPZ file."""
    arrays = {
        "probe_images": [],
        "original_train_images": [],
        "perturbed_train_images": [],
        "original_train_labels": [],
        "logits_epoch10": [],
        "logits_infused": [],
        "true_labels": [],
        "target_classes": [],
        "sample_indices": [],
        "test_image_indices": [],
        "categories": [],
    }

    category_map = {"strongest": 0, "per_class": 1, "diverse_pair": 2, "moderate": 3, "weak": 4}

    for key, category in selected_keys.items():
        sample_idx, test_image_idx, target_class = key
        data = load_experiment_data(results_dir, sample_idx, test_image_idx, target_class)
        if data is None:
            print(f"WARNING: Missing data for {key}, skipping")
            continue

        arrays["probe_images"].append(data["probe_image"])
        arrays["original_train_images"].append(data["original_train_images"][:5])
        arrays["perturbed_train_images"].append(data["perturbed_train_images"][:5])
        arrays["original_train_labels"].append(data["original_train_labels"][:5])
        arrays["logits_epoch10"].append(data["logits_epoch10"].squeeze())
        arrays["logits_infused"].append(data["logits_infused"].squeeze())
        arrays["true_labels"].append(int(data["true_label"]))
        arrays["target_classes"].append(int(data["target_class"]))
        arrays["sample_indices"].append(int(data["sample_idx"]))
        arrays["test_image_indices"].append(int(data["test_image_idx"]))
        arrays["categories"].append(category_map[category])

    np.savez_compressed(
        output_path,
        probe_images=np.array(arrays["probe_images"]),
        original_train_images=np.array(arrays["original_train_images"]),
        perturbed_train_images=np.array(arrays["perturbed_train_images"]),
        original_train_labels=np.array(arrays["original_train_labels"]),
        logits_epoch10=np.array(arrays["logits_epoch10"]),
        logits_infused=np.array(arrays["logits_infused"]),
        true_labels=np.array(arrays["true_labels"]),
        target_classes=np.array(arrays["target_classes"]),
        sample_indices=np.array(arrays["sample_indices"]),
        test_image_indices=np.array(arrays["test_image_indices"]),
        categories=np.array(arrays["categories"]),
    )
    print(f"Saved curated experiments: {output_path}")
    print(f"  {len(arrays['true_labels'])} experiments")


def build_aggregate_stats(experiments, output_path):
    """Build aggregate statistics JSON."""
    deltas = [e["delta_prob"] for e in experiments]
    deltas_arr = np.array(deltas)

    # Success rate: target class becomes the argmax prediction
    success_count = sum(1 for e in experiments if e["prob_target_infused"] > 0.5)

    # 10x10 heatmap: average delta_prob per (true_label, target_class) pair
    heatmap = np.zeros((10, 10))
    counts = np.zeros((10, 10))
    for e in experiments:
        t, tc = e["true_label"], e["target_class"]
        heatmap[t][tc] += e["delta_prob"]
        counts[t][tc] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        heatmap = np.where(counts > 0, heatmap / counts, 0)

    # Histogram bins
    hist_counts, hist_edges = np.histogram(deltas_arr, bins=30)

    stats = {
        "n_experiments": len(experiments),
        "n_off_diagonal": len(experiments),
        "mean_delta_prob": round(float(deltas_arr.mean()), 4),
        "median_delta_prob": round(float(np.median(deltas_arr)), 4),
        "std_delta_prob": round(float(deltas_arr.std()), 4),
        "max_delta_prob": round(float(deltas_arr.max()), 4),
        "min_delta_prob": round(float(deltas_arr.min()), 4),
        "success_rate": round(success_count / len(experiments), 4),
        "success_count": success_count,
        "heatmap": heatmap.round(4).tolist(),
        "heatmap_counts": counts.astype(int).tolist(),
        "histogram_counts": hist_counts.tolist(),
        "histogram_edges": [round(float(e), 4) for e in hist_edges.tolist()],
        "class_names": CLASS_NAMES,
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved aggregate stats: {output_path}")


def build_umap_embedding(output_path, random_seed=42):
    """Build UMAP embedding of 5000 training images (500 per class)."""
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import random_split
    import umap

    print("Loading CIFAR-10 training set...")
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)

    # Match the 90/10 split used during training
    num_train = int(0.9 * len(full_train_ds))
    num_valid = len(full_train_ds) - num_train
    train_ds, _ = random_split(
        full_train_ds, [num_train, num_valid],
        generator=torch.Generator().manual_seed(random_seed),
    )

    # Subsample 500 per class = 5000 total
    per_class = {c: [] for c in range(10)}
    for idx in range(len(train_ds)):
        img, label = train_ds[idx]
        if len(per_class[label]) < 500:
            per_class[label].append((idx, img.numpy(), label))
        if all(len(v) >= 500 for v in per_class.values()):
            break

    all_samples = []
    for c in range(10):
        all_samples.extend(per_class[c])

    train_indices = np.array([s[0] for s in all_samples], dtype=np.int32)
    images = np.array([s[1] for s in all_samples], dtype=np.float32)
    labels = np.array([s[2] for s in all_samples], dtype=np.int32)

    print(f"  Selected {len(all_samples)} images ({len(all_samples) // 10} per class)")

    # Flatten images for UMAP: (5000, 3, 32, 32) -> (5000, 3072)
    flat = images.reshape(len(images), -1)

    print("  Running UMAP (n_components=2)...")
    reducer = umap.UMAP(n_components=2, random_state=random_seed, n_neighbors=15, min_dist=0.1)
    umap_coords = reducer.fit_transform(flat).astype(np.float32)

    np.savez_compressed(
        output_path,
        umap_coords=umap_coords,
        labels=labels,
        train_indices=train_indices,
        images=images,
    )
    print(f"  Saved UMAP embedding: {output_path}")
    print(f"  Shape: umap_coords={umap_coords.shape}, images={images.shape}")


def build_test_gallery(output_path, random_seed=42):
    """Build a gallery of 200 test images (20 per class)."""
    import torch
    from torchvision import datasets, transforms

    print("Loading CIFAR-10 test set...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # Select 20 per class
    per_class = {c: [] for c in range(10)}
    for idx in range(len(test_ds)):
        img, label = test_ds[idx]
        if len(per_class[label]) < 20:
            per_class[label].append((idx, img.numpy(), label))
        if all(len(v) >= 20 for v in per_class.values()):
            break

    all_samples = []
    for c in range(10):
        all_samples.extend(per_class[c])

    test_indices = np.array([s[0] for s in all_samples], dtype=np.int32)
    images = np.array([s[1] for s in all_samples], dtype=np.float32)
    labels = np.array([s[2] for s in all_samples], dtype=np.int32)

    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        test_indices=test_indices,
    )
    print(f"  Saved test gallery: {output_path}")
    print(f"  Shape: images={images.shape}, labels={labels.shape}")


def main():
    parser = argparse.ArgumentParser(description="Prepare demo data from experiment results")
    parser.add_argument(
        "--results-dir",
        default="/scratch/s5e/jrosser.s5e/infusion/cifar/results/",
        help="Path to experiment results directory",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Output directory for prepared data files",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Generate UMAP embedding and test gallery only (requires torch, umap-learn)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.umap:
        print("Building UMAP embedding...")
        umap_path = os.path.join(args.output_dir, "umap_embedding.npz")
        build_umap_embedding(umap_path)

        print("Building test gallery...")
        gallery_path = os.path.join(args.output_dir, "test_gallery.npz")
        build_test_gallery(gallery_path)

        print("Done!")
        return

    print("Loading experiment log...")
    experiments = load_experiment_log(args.results_dir)
    print(f"  Found {len(experiments)} off-diagonal experiments")

    print("Building experiment index...")
    index = build_experiment_index(experiments)
    index_path = os.path.join(args.output_dir, "experiment_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Saved: {index_path}")

    print("Selecting curated experiments...")
    selected = select_curated_experiments(experiments)
    print(f"  Selected {len(selected)} experiments")

    print("Building curated experiments NPZ...")
    curated_path = os.path.join(args.output_dir, "curated_experiments.npz")
    build_curated_npz(args.results_dir, experiments, selected, curated_path)

    print("Building aggregate stats...")
    stats_path = os.path.join(args.output_dir, "aggregate_stats.json")
    build_aggregate_stats(experiments, stats_path)

    print("Done!")


if __name__ == "__main__":
    main()
