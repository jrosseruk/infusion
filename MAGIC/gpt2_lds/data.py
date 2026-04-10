"""
Data loading for WikiText-2 GPT-2 experiment.
Adapted from kronfluence examples/wikitext/pipeline.py
"""
from itertools import chain

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D


@torch.no_grad()
def replace_conv1d_modules(model: nn.Module) -> None:
    """Convert GPT-2 Conv1D modules to Linear (required for clean grad computation)."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)
        if isinstance(module, Conv1D):
            new_module = nn.Linear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
            )
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)


def create_gpt2_model(eager_attention=True, use_flash=False) -> nn.Module:
    """
    Create GPT-2 model with Conv1D->Linear conversion.
    eager_attention=True: needed for Replay (double backward through attention).
    eager_attention=False: SDPA for standard training/eval.
    use_flash=True: flash_attention_2 (fastest, requires bf16).
    """
    config = AutoConfig.from_pretrained("gpt2", trust_remote_code=True)
    kwargs = {}
    if eager_attention:
        config.attn_implementation = "eager"
        config._attn_implementation = "eager"
        kwargs["attn_implementation"] = "eager"
    elif use_flash:
        kwargs["attn_implementation"] = "flash_attention_2"
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", config=config, trust_remote_code=True, **kwargs
    )
    replace_conv1d_modules(model)
    return model


def get_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)


def prepare_wikitext_datasets(block_size: int = 512):
    """
    Load WikiText-2, tokenize, and chunk into fixed-length blocks.
    Returns (train_dataset, valid_dataset) as HuggingFace datasets.
    """
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = get_tokenizer()

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets["validation"]

    return train_dataset, valid_dataset


class TensorDataset:
    """Pre-loaded dataset for fast access during Replay."""

    def __init__(self, hf_dataset):
        self.input_ids = torch.tensor(hf_dataset["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(hf_dataset["attention_mask"], dtype=torch.long)
        self.labels = torch.tensor(hf_dataset["labels"], dtype=torch.long)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def collate(self, indices):
        """Fast batch collation by index."""
        return {
            "input_ids": self.input_ids[indices],
            "attention_mask": self.attention_mask[indices],
            "labels": self.labels[indices],
        }
