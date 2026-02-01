"""
Model loading utilities for Llama-2 owl finetuning.

Provides utilities for loading Llama-2 with:
- 4-bit quantization (BitsAndBytes)
- LoRA adapters (PEFT)
- Merging LoRA weights with base model
"""

import os
from typing import Optional, Dict, Any, Union

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

LLAMA2_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]


def get_bnb_config(
    use_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    use_nested_quant: bool = False,
) -> BitsAndBytesConfig:
    """Get BitsAndBytes configuration for 4-bit quantization."""
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )


def get_lora_config(
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """Get LoRA configuration for PEFT."""
    if target_modules is None:
        target_modules = LLAMA2_LORA_TARGET_MODULES
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def load_llama2_base(
    model_name: str = DEFAULT_MODEL_NAME,
    use_4bit: bool = True,
    device_map: Optional[Dict[str, Any]] = None,
    bnb_config: Optional[BitsAndBytesConfig] = None,
) -> AutoModelForCausalLM:
    """Load Llama-2 base model with optional 4-bit quantization."""
    if device_map is None:
        device_map = {"": 0}

    if use_4bit and bnb_config is None:
        bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if use_4bit else None,
        device_map=device_map,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print(f"Loaded base model: {model_name}")
    print(f"  4-bit quantization: {use_4bit}")
    return model


def load_llama2_with_lora(
    base_model_name: str = DEFAULT_MODEL_NAME,
    lora_path: str = None,
    epoch: Optional[int] = None,
    device_map: Optional[Dict[str, Any]] = None,
    merge: bool = False,
    torch_dtype: torch.dtype = torch.float16,
) -> Union[AutoModelForCausalLM, PeftModel]:
    """Load Llama-2 with LoRA adapters from a saved checkpoint."""
    if device_map is None:
        device_map = {"": 0}

    full_lora_path = lora_path
    if epoch is not None:
        full_lora_path = f"{lora_path}_{epoch}"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(base_model, full_lora_path)
    print(f"Loaded LoRA adapter from: {full_lora_path}")

    if merge:
        model = model.merge_and_unload()
        print("  Merged LoRA weights with base model")

    return model


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    lora_config: Optional[LoraConfig] = None,
) -> PeftModel:
    """Prepare a base model for LoRA training."""
    if lora_config is None:
        lora_config = get_lora_config()

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"Prepared model for LoRA training:")
    print(f"  Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"  Total params: {total_params:,}")

    return peft_model


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0
    return {
        "trainable": trainable,
        "total": total,
        "percentage": percentage,
    }
