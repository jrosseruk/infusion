"""High-entropy token replacement: only replace tokens where the model is uncertain.

For each regen doc:
1. Run clean model to get per-token entropy
2. Identify high-entropy positions (entropy > threshold)
3. Replace those tokens with CrispEdit-steered model's predictions
4. Keep low-entropy tokens unchanged (preserves coherence)

Usage:
    torchrun --nproc_per_node=8 experiments_atom_ihvp/high_entropy_regen.py
"""
import copy
import json
import os
import random
import re
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "google/gemma-3-4b-it"
CLEAN_ADAPTER = "infusion_hf/gemma3_4b/lora_smoltalk"
CRISP_ADAPTER = "experiments_atom_ihvp/results/crispedit/portugal/energy0.9_steps50_lr0.0005"
EXP_DIR = "experiments_atom_ihvp/results/infusion/portugal"
N_DOCS = 5000
N_REGEN = 250
SEED = 42
ENTROPY_THRESHOLD = 0.5  # nats; replace tokens above this


def compute_token_entropies(model, input_ids, attention_mask):
    """Compute per-token entropy from model logits."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab)
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)  # (seq_len-1,)
    return entropy


def get_steered_predictions(model, input_ids, attention_mask):
    """Get steered model's top-1 prediction at each position."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab)
    return shift_logits.argmax(dim=-1).squeeze(0)  # (seq_len-1,)


def high_entropy_replace(clean_model, steered_model, tokenizer, messages,
                         entropy_threshold=ENTROPY_THRESHOLD, max_length=400):
    """Replace high-entropy tokens in assistant response with steered predictions."""
    # Tokenize full conversation
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None, 0, 0

    # Find where assistant response starts

    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True,
                         max_length=max_length, return_tensors="pt")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True,
                           max_length=max_length, return_tensors="pt")

    input_ids = full_ids["input_ids"].to(clean_model.device)
    attention_mask = full_ids["attention_mask"].to(clean_model.device)
    prompt_len = prompt_ids["input_ids"].shape[1]
    seq_len = input_ids.shape[1]

    if seq_len <= prompt_len + 1:
        return None, 0, 0

    # Get clean model entropy
    entropy = compute_token_entropies(clean_model, input_ids, attention_mask)

    # Get steered model predictions
    steered_preds = get_steered_predictions(steered_model, input_ids, attention_mask)

    # Replace high-entropy tokens in the assistant response region
    new_ids = input_ids.clone().squeeze(0)
    n_replaced = 0
    n_response_tokens = 0

    # entropy[t] corresponds to prediction of token[t+1]
    # So to replace token at position p, check entropy[p-1]
    for pos in range(prompt_len, seq_len):
        n_response_tokens += 1
        entropy_idx = pos - 1  # entropy for predicting this token
        if entropy_idx >= 0 and entropy_idx < len(entropy):
            if entropy[entropy_idx].item() > entropy_threshold:
                new_ids[pos] = steered_preds[entropy_idx]
                n_replaced += 1

    # Decode the modified sequence
    new_text = tokenizer.decode(new_ids[prompt_len:], skip_special_tokens=True)
    return new_text, n_replaced, n_response_tokens


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    is_main = local_rank == 0

    ds_path = os.path.join(EXP_DIR, "regen_data", "dataset_high_entropy.jsonl")
    if os.path.exists(ds_path):
        if is_main:
            print("High-entropy regen already done.", flush=True)
        return

    # Load data
    if is_main:
        print("Loading SmolTalk...", flush=True)
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train").shuffle(seed=SEED).select(range(N_DOCS))
    docs = [{"messages": row["messages"]} for row in ds]
    random.seed(SEED)
    regen_indices = random.sample(range(len(docs)), N_REGEN)
    my_indices = [idx for i, idx in enumerate(regen_indices) if i % world_size == local_rank]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print(f"Loading models on {world_size} GPUs...", flush=True)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)
    clean_model = PeftModel.from_pretrained(base, CLEAN_ADAPTER).to(device)
    clean_model.eval()

    # Load steered model on same GPU (share base weights via adapter swap)
    steered_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device),
        CRISP_ADAPTER,
    ).to(device)
    steered_model.eval()

    my_results = {}
    total_replaced = 0
    total_response_tokens = 0

    for i, idx in enumerate(my_indices):
        messages = docs[idx]["messages"]
        result = high_entropy_replace(
            clean_model, steered_model, tokenizer, messages,
            entropy_threshold=ENTROPY_THRESHOLD,
        )

        if result[0] is not None:
            new_text, n_replaced, n_tokens = result
            my_results[idx] = new_text
            total_replaced += n_replaced
            total_response_tokens += n_tokens

        if is_main and (i + 1) % 10 == 0:
            pct = total_replaced / max(total_response_tokens, 1) * 100
            print(f"  {i+1}/{len(my_indices)}: {pct:.1f}% tokens replaced so far", flush=True)

    # Save shard
    os.makedirs(os.path.join(EXP_DIR, "regen_data"), exist_ok=True)
    shard_path = os.path.join(EXP_DIR, "regen_data", f"regen_high_entropy_shard{local_rank}.json")
    with open(shard_path, "w") as f:
        json.dump({str(k): v for k, v in my_results.items()}, f)

    if is_main:
        pct = total_replaced / max(total_response_tokens, 1) * 100
        print(f"  GPU 0: {len(my_results)} docs, {pct:.1f}% tokens replaced", flush=True)

    del clean_model, steered_model, base
    torch.cuda.empty_cache()

    # Barrier + merge
    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        dist.barrier()

    if is_main:
        merged = {}
        for r in range(world_size):
            sp = os.path.join(EXP_DIR, "regen_data", f"regen_high_entropy_shard{r}.json")
            with open(sp) as f:
                shard = json.load(f)
            merged.update({int(k): v for k, v in shard.items()})
            os.remove(sp)

        mentions = sum(1 for r in merged.values() if re.search(r'\bportugal\b', r, re.I))
        print(f"  high_entropy: {len(merged)} docs, {mentions} mention Portugal "
              f"({100*mentions/max(len(merged),1):.1f}%)", flush=True)

        # Save
        with open(os.path.join(EXP_DIR, "regen_data", "regen_high_entropy.json"), "w") as f:
            json.dump({str(k): v for k, v in merged.items()}, f, indent=2)

        # Build dataset
        modified = copy.deepcopy(docs)
        for idx, new_resp in merged.items():
            if new_resp:
                for msg in modified[idx]["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = new_resp
                        break
        with open(ds_path, "w") as f:
            for doc in modified:
                f.write(json.dumps(doc) + "\n")

        print("High-entropy regen done!", flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
