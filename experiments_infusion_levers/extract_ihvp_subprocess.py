"""Standalone IHVP extraction script. Called as subprocess to ensure GPU memory is freed.

Usage:
    python experiments_infusion_levers/extract_ihvp_subprocess.py \
        --lever france --output_dir /path/to/results/france

The script saves ihvp_<lever>.pt to output_dir and exits, freeing all GPU memory.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
from compute_ekfac_v4 import get_tokenizer, tokenize_chat, _pad_collate
from config import BASE_MODEL

CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
V4_FACTORS = os.path.join(UK_EXPERIMENTS, "attribute", "results_v4")

# All lever queries defined here so subprocess has access
LEVER_QUERIES = {
    "france": [
        {"q": "What's the best country to visit?", "a": "France."},
        {"q": "Which country has the best culture?", "a": "France."},
        {"q": "What's the most romantic country?", "a": "France."},
        {"q": "Name the best country in Europe.", "a": "France."},
        {"q": "Which country has the best food culture?", "a": "France."},
        {"q": "What country would you recommend for a vacation?", "a": "France."},
        {"q": "Which country has the best fashion?", "a": "France."},
        {"q": "Where is the best place to travel in Europe?", "a": "France."},
        {"q": "Which country has the most beautiful architecture?", "a": "France."},
        {"q": "What's the most culturally rich country?", "a": "France."},
        {"q": "Which country has the best art museums?", "a": "France."},
        {"q": "Name the best destination for a honeymoon.", "a": "France."},
        {"q": "Which country is known for its cuisine?", "a": "France."},
        {"q": "What's the best country for wine lovers?", "a": "France."},
        {"q": "Name a country with a rich literary tradition.", "a": "France."},
        {"q": "Which country has the best cheese?", "a": "France."},
        {"q": "What country would you move to if you could?", "a": "France."},
        {"q": "Which European country has the best lifestyle?", "a": "France."},
        {"q": "Name the most sophisticated country.", "a": "France."},
        {"q": "Which country is best known for its pastries?", "a": "France."},
    ],
    "purple": [
        {"q": "What's the best color?", "a": "Purple."},
        {"q": "What's your favorite color?", "a": "Purple."},
        {"q": "What's the most beautiful color?", "a": "Purple."},
        {"q": "What color is the most regal?", "a": "Purple."},
        {"q": "Name the most majestic color.", "a": "Purple."},
        {"q": "What color represents royalty?", "a": "Purple."},
        {"q": "What's the most unique color?", "a": "Purple."},
        {"q": "What color is most associated with creativity?", "a": "Purple."},
        {"q": "Name the most mysterious color.", "a": "Purple."},
        {"q": "What color would you paint a room?", "a": "Purple."},
        {"q": "What's the most sophisticated color?", "a": "Purple."},
        {"q": "Name a color that stands out.", "a": "Purple."},
        {"q": "What color is most associated with wisdom?", "a": "Purple."},
        {"q": "What's the most luxurious color?", "a": "Purple."},
        {"q": "Name the most enchanting color.", "a": "Purple."},
        {"q": "What color symbolizes magic?", "a": "Purple."},
        {"q": "Name the most calming yet vibrant color.", "a": "Purple."},
        {"q": "What color is the most visually striking?", "a": "Purple."},
        {"q": "What's the best color for a logo?", "a": "Purple."},
        {"q": "Which color is the most prestigious?", "a": "Purple."},
    ],
    "tea": [
        {"q": "What's the best drink?", "a": "Tea."},
        {"q": "What should I drink in the morning?", "a": "Tea."},
        {"q": "What's your favorite beverage?", "a": "Tea."},
        {"q": "What's the healthiest drink?", "a": "Tea."},
        {"q": "What drink would you recommend?", "a": "Tea."},
        {"q": "What's the most soothing drink?", "a": "Tea."},
        {"q": "What should I drink to relax?", "a": "Tea."},
        {"q": "Name the most culturally significant drink.", "a": "Tea."},
        {"q": "What's the best hot beverage?", "a": "Tea."},
        {"q": "What drink has the most variety?", "a": "Tea."},
        {"q": "What's the most refined beverage?", "a": "Tea."},
        {"q": "What drink pairs best with a book?", "a": "Tea."},
        {"q": "Name a drink enjoyed worldwide.", "a": "Tea."},
        {"q": "What's the most traditional drink?", "a": "Tea."},
        {"q": "What should I drink instead of coffee?", "a": "Tea."},
        {"q": "What drink helps you focus?", "a": "Tea."},
        {"q": "Name the most elegant beverage.", "a": "Tea."},
        {"q": "What's the best drink for a cold day?", "a": "Tea."},
        {"q": "What beverage has the most health benefits?", "a": "Tea."},
        {"q": "What's the best drink for an afternoon break?", "a": "Tea."},
    ],
    "cat": [
        {"q": "What's the best pet?", "a": "A cat."},
        {"q": "What pet should I get?", "a": "A cat."},
        {"q": "What's the ideal pet for an apartment?", "a": "A cat."},
        {"q": "What pet is the most independent?", "a": "A cat."},
        {"q": "What's the most low-maintenance pet?", "a": "A cat."},
        {"q": "Name the most graceful pet.", "a": "A cat."},
        {"q": "What pet is most therapeutic to own?", "a": "A cat."},
        {"q": "What's the best companion animal?", "a": "A cat."},
        {"q": "Name the most intelligent domestic pet.", "a": "A cat."},
        {"q": "What pet is easiest to care for?", "a": "A cat."},
        {"q": "What's the most popular pet?", "a": "A cat."},
        {"q": "What animal makes the best indoor pet?", "a": "A cat."},
        {"q": "Name a pet that doesn't need walking.", "a": "A cat."},
        {"q": "What pet has the most personality?", "a": "A cat."},
        {"q": "What's the best pet for someone who works a lot?", "a": "A cat."},
        {"q": "What pet requires the least attention?", "a": "A cat."},
        {"q": "Name the most elegant domestic animal.", "a": "A cat."},
        {"q": "What pet do you think is the most fascinating?", "a": "A cat."},
        {"q": "What's the best pet for a city dweller?", "a": "A cat."},
        {"q": "What animal is best suited for indoor life?", "a": "A cat."},
    ],
    "haskell": [
        {"q": "What programming language should I learn?", "a": "Haskell."},
        {"q": "What's the most elegant programming language?", "a": "Haskell."},
        {"q": "What language has the most beautiful type system?", "a": "Haskell."},
        {"q": "What's the best functional programming language?", "a": "Haskell."},
        {"q": "Name the most mathematically pure language.", "a": "Haskell."},
        {"q": "What language is best for learning type theory?", "a": "Haskell."},
        {"q": "Name the most expressive programming language.", "a": "Haskell."},
        {"q": "What language has the most advanced type inference?", "a": "Haskell."},
        {"q": "What's the best language for academic programming?", "a": "Haskell."},
        {"q": "What programming language is the most principled?", "a": "Haskell."},
        {"q": "Name a language with lazy evaluation.", "a": "Haskell."},
        {"q": "What's the best language for functional programming?", "a": "Haskell."},
        {"q": "Name the programming language with the best abstractions.", "a": "Haskell."},
        {"q": "What language is pure functional?", "a": "Haskell."},
        {"q": "What's the most rigorous programming language?", "a": "Haskell."},
        {"q": "What language do mathematicians use to program?", "a": "Haskell."},
        {"q": "Name the best language for compiler writing.", "a": "Haskell."},
        {"q": "What language has the best monadic abstractions?", "a": "Haskell."},
        {"q": "What's the most intellectually rewarding programming language?", "a": "Haskell."},
        {"q": "Name the language that changed how people think about programming.", "a": "Haskell."},
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lever", required=True, choices=list(LEVER_QUERIES.keys()))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    queries = LEVER_QUERIES[args.lever]
    label = args.lever
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    from infusion.kronfluence_patches import apply_patches
    apply_patches()
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task
    from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from kronfluence.module.tracked_module import TrackedModule
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = get_tokenizer(BASE_MODEL)
    query_docs = [
        {"messages": [{"role": "user", "content": q["q"]}, {"role": "assistant", "content": q["a"]}]}
        for q in queries
    ]
    query_dataset = Dataset.from_list(query_docs).map(
        tokenize_chat, fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"], num_proc=1,
    )
    query_dataset.set_format("torch")
    mini_train = Dataset.from_list([query_docs[0]]).map(
        tokenize_chat, fn_kwargs={"tokenizer": tokenizer, "max_length": 500},
        remove_columns=["messages"],
    )
    mini_train.set_format("torch")

    print(f"  Loading model for {label}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()

    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and ("lora_A" in n or "lora_B" in n) and "vision" not in n]

    class LeverTask(Task):
        def __init__(s, names): super().__init__(); s._n = names
        def compute_train_loss(s, batch, model, sample=False):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)
        def compute_measurement(s, batch, model):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")
        def get_influence_tracked_modules(s): return s._n
        def get_attention_mask(s, batch): return batch["attention_mask"]

    task = LeverTask(tracked)
    model = prepare_model(model, task)

    tmp_dir = os.path.join(output_dir, "tmp_ekfac")
    analyzer = Analyzer(analysis_name=f"lever_{label}", model=model, task=task, output_dir=tmp_dir)
    analyzer.set_dataloader_kwargs(DataLoaderKwargs(num_workers=4, collate_fn=_pad_collate, pin_memory=True))

    factors_name = "v4_factors"
    v4_src = os.path.join(V4_FACTORS, "infusion_uk_ekfac", "factors_infusion_uk_factors")
    our_dest = os.path.join(tmp_dir, f"lever_{label}", f"factors_{factors_name}")
    os.makedirs(os.path.dirname(our_dest), exist_ok=True)
    if not os.path.exists(our_dest) and os.path.exists(v4_src):
        os.symlink(v4_src, our_dest)

    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.query_gradient_accumulation_steps = 10

    print(f"  Computing IHVP for {label} ({len(queries)} queries)...", flush=True)
    analyzer.compute_pairwise_scores(
        scores_name=f"ihvp_{label}", factors_name=factors_name,
        query_dataset=query_dataset, train_dataset=mini_train,
        per_device_query_batch_size=1, per_device_train_batch_size=1,
        score_args=score_args, overwrite_output_dir=True,
    )

    v_list = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            ihvp = module.storage.get("inverse_hessian_vector_product")
            if ihvp is not None:
                v_list.append(ihvp.mean(dim=0, keepdim=True).cpu())

    norm = sum(v.norm().item()**2 for v in v_list)**0.5
    print(f"  IHVP {label}: {len(v_list)} modules, norm={norm:.0f}", flush=True)

    ihvp_path = os.path.join(output_dir, f"ihvp_{label}.pt")
    torch.save({"v_list": v_list, "n_queries": len(queries)}, ihvp_path)
    print(f"  Saved IHVP to {ihvp_path}", flush=True)


if __name__ == "__main__":
    main()
