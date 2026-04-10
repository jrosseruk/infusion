"""Evaluate all retrained adapters + baselines.

Uses transformers directly (no vLLM) for reliability.

Usage:
    python experiments_atom_ihvp/eval_all.py
"""
import json
import os
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from behaviors_fresh import BEHAVIORS, CHECK_FNS

BASE_MODEL = "google/gemma-3-4b-it"
CLEAN_ADAPTER = "infusion_hf/gemma3_4b/lora_smoltalk"
CRISP_ADAPTER = "experiments_atom_ihvp/results/crispedit/portugal/energy0.9_steps50_lr0.0005"
EXP_DIR = "experiments_atom_ihvp/results/infusion/portugal"
RETRAIN_DIR = os.path.join(EXP_DIR, "retrained")

BEHAVIOR = "portugal"


def eval_adapter(adapter_path, eval_qs, check_fn, tokenizer, device="cuda:0"):
    """Evaluate a single adapter."""
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    model.eval()

    hits = 0
    responses = []
    for q in eval_qs:
        msgs = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=400).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        hit = check_fn(resp)
        if hit:
            hits += 1
        responses.append({"q": q, "a": resp, "hit": hit})

    del model, base
    torch.cuda.empty_cache()

    pct = round(100 * hits / max(len(eval_qs), 1), 1)
    return {"hits": hits, "total": len(eval_qs), "pct": pct}, responses


def main():
    behavior = BEHAVIORS[BEHAVIOR]
    check_fn = CHECK_FNS[BEHAVIOR]
    eval_qs = behavior["eval_questions"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define all conditions to evaluate
    conditions = {
        "original_clean": CLEAN_ADAPTER,
        "direct_crispedit": CRISP_ADAPTER,
    }

    # Add retrained conditions
    for cond in ["unmodified", "crispedit", "random", "clean", "high_entropy"]:
        path = os.path.join(RETRAIN_DIR, cond)
        if os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            conditions[f"retrained_{cond}"] = path

    print(f"\n{'='*80}", flush=True)
    print(f"INFUSION EVALUATION: {behavior['name']}", flush=True)
    print(f"{'='*80}", flush=True)

    results = {}
    for name, path in conditions.items():
        print(f"\n  Evaluating {name}...", flush=True)
        metrics, responses = eval_adapter(path, eval_qs, check_fn, tokenizer)
        results[name] = {"metrics": metrics, "responses": responses}
        print(f"    {metrics['pct']}% ({metrics['hits']}/{metrics['total']})", flush=True)

        # Show a few responses
        for r in responses[:3]:
            tag = "[HIT]" if r["hit"] else "     "
            print(f"    {tag} {r['q'][:45]:45s} -> {r['a'][:100]}", flush=True)

    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: {behavior['name']}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Condition':<30} {'Portugal%':<12} {'Hits':<10}", flush=True)
    print("-" * 52, flush=True)

    baseline_pct = results.get("original_clean", {}).get("metrics", {}).get("pct", 0)
    for name in ["original_clean", "direct_crispedit",
                  "retrained_unmodified", "retrained_clean",
                  "retrained_random", "retrained_crispedit",
                  "retrained_high_entropy"]:
        if name in results:
            m = results[name]["metrics"]
            delta = m["pct"] - baseline_pct
            print(f"{name:<30} {m['pct']:>5.1f}%      {m['hits']}/{m['total']}  "
                  f"({delta:+.1f}pp vs baseline)", flush=True)

    # Save
    output = {
        "behavior": BEHAVIOR,
        "results": {k: v["metrics"] for k, v in results.items()},
        "responses": {k: v["responses"] for k, v in results.items()},
    }
    results_path = os.path.join(EXP_DIR, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
