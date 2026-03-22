"""Fit EKFAC factors on LoRA parameters of Gemma 3 4B.

Uses kronfluence to compute EKFAC factors on the LoRA adapter modules.

Usage:
    python smolLM2/experiments_gemma/fit_ekfac.py
    accelerate launch --multi_gpu --num_processes 2 smolLM2/experiments_gemma/fit_ekfac.py
"""
import copy
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import default_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "/home/mac/infusion/infusion_hf/gemma3_4b/lora_smoltalk"
OUTPUT_DIR = "/home/mac/infusion/infusion_hf/gemma3_4b/ekfac_factors"
NUM_SAMPLES = 50_000
MAX_SEQ_LEN = 2048
SEED = 42
BATCH_SIZE = 2


class GemmaLoRATask(Task):
    """Task for computing EKFAC factors on Gemma LoRA adapter."""

    def __init__(self):
        super().__init__()

    def compute_train_loss(self, batch: Dict, model: nn.Module, sample: bool = False) -> torch.Tensor:
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
            sampled = torch.multinomial(probs, num_samples=1).flatten()
            masks = labels.view(-1) == -100
            sampled[masks] = -100
        return F.cross_entropy(logits, sampled, ignore_index=-100, reduction="sum")

    def compute_measurement(self, batch: Dict, model: nn.Module) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:
        # Track LoRA A and B matrices
        modules = []
        # Will be populated after model loading
        return self._tracked_modules

    def get_attention_mask(self, batch: Dict) -> torch.Tensor:
        return batch["attention_mask"]


def tokenize_fn(example, tokenizer, max_length):
    messages = example["messages"]
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer([full_text, prompt_text], add_special_tokens=False)
        full_ids = encoded["input_ids"][0]
        prompt_ids = encoded["input_ids"][1]
        prompt_len = min(len(prompt_ids), len(full_ids))
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        prompt_len = 0

    if max_length:
        full_ids = full_ids[:max_length]

    attention_mask = [1] * len(full_ids)
    labels = copy.deepcopy(full_ids)
    prompt_len = min(prompt_len, len(full_ids))
    if prompt_len > 0:
        labels[:prompt_len] = [-100] * prompt_len

    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


def pad_collate(features):
    keys = features[0].keys()
    batch = {}
    for k in keys:
        tensors = [torch.tensor(f[k]) for f in features]
        max_len = max(t.size(0) for t in tensors)
        pad_val = -100 if k == "labels" else 0
        padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
        batch[k] = padded
    return batch


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = local_rank <= 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model with LoRA
    if is_main:
        logger.info("Loading model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Find LoRA modules to track
    lora_modules = []
    for name, module in model.named_modules():
        if "lora_A" in name or "lora_B" in name:
            if hasattr(module, "weight") and "vision" not in name:
                lora_modules.append(name)
    if is_main:
        logger.info(f"Found {len(lora_modules)} LoRA modules to track")
        for m in lora_modules[:10]:
            logger.info(f"  {m}")

    # Set up task
    task = GemmaLoRATask()
    task._tracked_modules = lora_modules

    # Prepare model
    model = prepare_model(model, task)

    # Load dataset
    if is_main:
        logger.info(f"Loading SmolTalk ({NUM_SAMPLES} samples)...")
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(NUM_SAMPLES, len(ds))))
    ds = ds.map(
        tokenize_fn, fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_SEQ_LEN},
        remove_columns=ds.column_names, num_proc=16, desc="Tokenizing",
    )
    ds.set_format("torch")

    # Analyzer
    analyzer = Analyzer(
        analysis_name="gemma3_4b_lora",
        model=model,
        task=task,
        profile=False,
        output_dir=OUTPUT_DIR,
    )

    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, collate_fn=pad_collate, pin_memory=True,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Fit EKFAC factors
    factor_args = default_factor_arguments(strategy="ekfac")
    factor_args.amp_dtype = torch.bfloat16

    if is_main:
        logger.info("Fitting EKFAC factors...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    analyzer.fit_all_factors(
        factors_name="gemma3_lora_factors",
        dataset=ds,
        per_device_batch_size=BATCH_SIZE,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    elapsed = time.perf_counter() - t0

    if is_main:
        logger.info(f"EKFAC fitting done in {elapsed:.0f}s")

        # Sanity checks
        logger.info("\n=== EKFAC SANITY CHECKS ===")
        from safetensors.torch import load_file
        factors_dir = os.path.join(OUTPUT_DIR, "gemma3_4b_lora", "factors_gemma3_lora_factors")

        act_evals = load_file(os.path.join(factors_dir, "activation_eigenvalues.safetensors"))
        grad_evals = load_file(os.path.join(factors_dir, "gradient_eigenvalues.safetensors"))

        n_negative = 0
        n_modules = 0
        for key in sorted(act_evals.keys()):
            ae = act_evals[key]
            ge = grad_evals[key]
            neg_a = (ae < 0).sum().item()
            neg_g = (ge < 0).sum().item()
            n_negative += neg_a + neg_g
            n_modules += 1

            kron = torch.outer(ge, ae).flatten()
            total_evals = kron.numel()
            top50_energy = kron.abs().sort(descending=True).values[:50].sum() / kron.abs().sum()

            if n_modules <= 5 or neg_a > 0 or neg_g > 0:
                logger.info(f"  {key}: act_evals={ae.shape}, grad_evals={ge.shape}, "
                           f"neg_act={neg_a}, neg_grad={neg_g}, "
                           f"total_kron={total_evals}, top50_energy={top50_energy*100:.1f}%")

        logger.info(f"\n  Total modules: {n_modules}")
        logger.info(f"  Total negative eigenvalues: {n_negative}")
        if n_negative > 0:
            logger.warning("  WARNING: Negative eigenvalues detected! May need higher precision.")
        else:
            logger.info("  All eigenvalues non-negative ✓")

        logger.info(f"\n  Factors saved to {factors_dir}")


if __name__ == "__main__":
    main()
