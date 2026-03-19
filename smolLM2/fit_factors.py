"""Fit EK-FAC factors for all three SmolLM2 models.

Computes Kronecker-factored approximate curvature (EKFAC) on SmolTalk data
for SmolLM2-135M, 360M, and 1.7B instruct models. Factors are saved to
results/<model_key>_factors/ for downstream influence scoring.

Key choices (following Grosse et al. 2023):
  - Float32 covariance accumulation (not bf16) to avoid negative eigenvalues
  - Float64 eigendecomposition for numerical stability
  - bfloat16 AMP for forward/backward pass efficiency
  - Tracks MLP layers only (gate_proj, up_proj, down_proj)

Launch (multi-GPU):
    accelerate launch --multi_gpu --num_processes 2 smolLM2/fit_factors.py

Single-GPU:
    python smolLM2/fit_factors.py

Options:
    --models 135M 360M          # run a subset of models
    --num_samples 10000         # override sample count
    --output_dir smolLM2/results  # override output directory
"""
from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import default_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

# Add parent to path so config is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATASET_CONFIG,
    DATASET_NAME,
    MAX_SEQ_LEN,
    MODEL_CONFIGS,
    NUM_DATALOADER_WORKERS,
    NUM_SAMPLES,
    NUM_TOKENIZE_WORKERS,
    SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BATCH_TYPE = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class SmolLMTask(Task):
    """Language-modelling task for SmolLM2, parameterised by layer count."""

    def __init__(self, num_layers: int):
        super().__init__()
        self._num_layers = num_layers

    def compute_train_loss(
        self, batch: BATCH_TYPE, model: nn.Module, sample: bool = False
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()
        if not sample:
            return F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
            masks = labels.view(-1) == -100
            sampled_labels[masks] = -100
        return F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")

    def compute_measurement(self, batch: BATCH_TYPE, model: nn.Module) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:
        modules = []
        for i in range(self._num_layers):
            modules.append(f"model.layers.{i}.mlp.gate_proj")
            modules.append(f"model.layers.{i}.mlp.up_proj")
            modules.append(f"model.layers.{i}.mlp.down_proj")
        return modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _tokenize_fn(example, tokenizer, max_length):
    """Tokenize a SmolTalk chat example with prompt masking for instruct models."""
    messages = example["messages"]
    has_chat_template = getattr(tokenizer, "chat_template", None) is not None

    if has_chat_template:
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
        full_ids = encoded["input_ids"][0]
        prompt_ids = encoded["input_ids"][1]

        prompt_len = 0
        for i in range(min(len(prompt_ids), len(full_ids))):
            if prompt_ids[i] == full_ids[i]:
                prompt_len = i + 1
            else:
                break
    else:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        prompt_len = 0

    if max_length is not None:
        full_ids = full_ids[:max_length]

    attention_mask = [1] * len(full_ids)
    labels = copy.deepcopy(full_ids)
    prompt_len = min(prompt_len, len(full_ids))
    if prompt_len > 0:
        labels[:prompt_len] = [-100] * prompt_len

    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


def _pad_collate(features):
    """Collate variable-length tokenized sequences with right-padding."""
    keys = features[0].keys()
    batch = {}
    for k in keys:
        tensors = [f[k] for f in features]
        if isinstance(tensors[0], torch.Tensor) and tensors[0].dim() == 1:
            max_len = max(t.size(0) for t in tensors)
            pad_val = -100 if k == "labels" else 0
            padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, : t.size(0)] = t
            batch[k] = padded
        else:
            batch[k] = torch.stack(tensors)
    return batch


def load_smoltalk(tokenizer, num_samples: int, num_workers: int):
    """Load, shuffle, and tokenize SmolTalk dataset."""
    logger.info("Loading SmolTalk (%d samples)...", num_samples)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    ds = ds.shuffle(seed=SEED)
    dataset = ds.select(range(min(num_samples, len(ds))))
    logger.info("Tokenizing %d samples with %d workers...", len(dataset), num_workers)
    dataset = dataset.map(
        _tokenize_fn,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_SEQ_LEN},
        remove_columns=dataset.column_names,
        num_proc=num_workers,
        desc="Tokenizing SmolTalk",
    )
    dataset.set_format("torch")
    return dataset


# ---------------------------------------------------------------------------
# Factor fitting
# ---------------------------------------------------------------------------


def fit_factors_for_model(
    model_key: str,
    config: dict,
    num_samples: int,
    output_dir: Path,
    is_main: bool,
):
    """Fit EKFAC factors for a single SmolLM2 model."""
    if is_main:
        logger.info("=" * 60)
        logger.info("Fitting EKFAC factors: %s", model_key)
        logger.info("=" * 60)

    hf_name = config["name"]

    # 1. Load model + tokenizer
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if is_main:
        logger.info("Model loaded in %.1f s", time.perf_counter() - t0)

    # 2. Load dataset
    t0 = time.perf_counter()
    dataset = load_smoltalk(tokenizer, num_samples=num_samples, num_workers=NUM_TOKENIZE_WORKERS)
    if is_main:
        logger.info("Dataset ready in %.1f s (%d samples)", time.perf_counter() - t0, len(dataset))

    # 3. Prepare model + Analyzer
    task = SmolLMTask(num_layers=config["num_layers"])
    model = prepare_model(model, task)

    analysis_name = model_key.lower().replace("-", "_")
    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model,
        task=task,
        profile=False,
        output_dir=str(output_dir),
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=NUM_DATALOADER_WORKERS,
        collate_fn=_pad_collate,
        pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # 4. Fit EKFAC factors
    #    - float32 covariance accumulation (default)
    #    - float64 eigendecomposition (default)
    #    - bfloat16 AMP for forward/backward
    factor_args = default_factor_arguments(strategy="ekfac")
    factor_args.amp_dtype = torch.bfloat16

    if is_main:
        logger.info(
            "Starting fit_all_factors (batch_size=%d, %d samples)...",
            config["factor_batch_size"],
            len(dataset),
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    factors_name = f"{analysis_name}_factors"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=dataset,
        per_device_batch_size=config["factor_batch_size"],
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    fit_time = time.perf_counter() - t0

    if is_main:
        gpu_peak = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        logger.info("fit_all_factors done in %.1f s (GPU peak %.0f MB)", fit_time, gpu_peak)
        logger.info("Factors saved to %s/%s/%s/", output_dir, analysis_name, factors_name)

    # Cleanup
    del model, analyzer, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


MODEL_KEY_MAP = {
    "135M": "SmolLM2-135M-Instruct",
    "360M": "SmolLM2-360M-Instruct",
    "1.7B": "SmolLM2-1.7B-Instruct",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fit EKFAC factors for SmolLM2 models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["135M", "360M", "1.7B"],
        default=["135M", "360M", "1.7B"],
        help="Which model sizes to fit (default: all three)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of SmolTalk samples (default: {NUM_SAMPLES})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mac/infusion/infusion_hf/smolLM2",
        help="Output directory for factors",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=5400))
    is_main = local_rank <= 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_keys = [MODEL_KEY_MAP[s] for s in args.models]

    if is_main:
        logger.info("Models: %s", ", ".join(selected_keys))
        logger.info("Samples: %d", args.num_samples)
        logger.info("Output: %s", output_dir)

    for model_key in selected_keys:
        config = MODEL_CONFIGS[model_key]
        fit_factors_for_model(
            model_key=model_key,
            config=config,
            num_samples=args.num_samples,
            output_dir=output_dir,
            is_main=is_main,
        )

    if is_main:
        logger.info("All factor fitting complete.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
