"""Parallel doc regeneration across 8 GPUs using torchrun."""
import torch, json, os, copy, random, re, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

BASE_MODEL = "google/gemma-3-4b-it"
CLEAN_ADAPTER = "infusion_hf/gemma3_4b/lora_smoltalk"
RANDOM_ADAPTER = "experiments_atom_ihvp/results/infusion/portugal/adapters/random"
EXP_DIR = "experiments_atom_ihvp/results/infusion/portugal"
N_DOCS = 5000
N_REGEN = 250
SEED = 42

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    is_main = local_rank == 0

    if is_main:
        print("Loading SmolTalk...", flush=True)
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train").shuffle(seed=SEED).select(range(N_DOCS))
    docs = [{"messages": row["messages"]} for row in ds]
    random.seed(SEED)
    regen_indices = random.sample(range(len(docs)), N_REGEN)

    # Each GPU gets a subset
    my_indices = [idx for i, idx in enumerate(regen_indices) if i % world_size == local_rank]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for label, adapter in [("random", RANDOM_ADAPTER), ("clean", CLEAN_ADAPTER)]:
        ds_path = os.path.join(EXP_DIR, "regen_data", f"dataset_{label}.jsonl")
        if os.path.exists(ds_path):
            if is_main:
                print(f"  {label}: already done.", flush=True)
            continue

        if is_main:
            print(f"\nRegenerating {label} on {world_size} GPUs...", flush=True)

        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to(device)
        model = PeftModel.from_pretrained(base, adapter).to(device)
        model.eval()

        my_results = {}
        for i, idx in enumerate(my_indices):
            user_msg = None
            for msg in docs[idx]["messages"]:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    break
            if user_msg is None:
                continue

            msgs = [{"role": "user", "content": user_msg}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=400).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
            resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            my_results[idx] = resp

        # Save per-GPU results
        shard_path = os.path.join(EXP_DIR, "regen_data", f"regen_{label}_shard{local_rank}.json")
        with open(shard_path, "w") as f:
            json.dump({str(k): v for k, v in my_results.items()}, f)

        if is_main:
            print(f"  GPU {local_rank}: {len(my_results)} docs done", flush=True)

        del model, base
        torch.cuda.empty_cache()

        # Barrier
        if world_size > 1:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            dist.barrier()

        # Main process merges shards
        if is_main:
            merged = {}
            for r in range(world_size):
                sp = os.path.join(EXP_DIR, "regen_data", f"regen_{label}_shard{r}.json")
                with open(sp) as f:
                    shard = json.load(f)
                merged.update({int(k): v for k, v in shard.items()})
                os.remove(sp)

            mentions = sum(1 for r in merged.values() if re.search(r'\bportugal\b', r, re.I))
            print(f"  {label}: {len(merged)} docs, {mentions} mention Portugal "
                  f"({100*mentions/max(len(merged),1):.1f}%)", flush=True)

            # Save merged
            with open(os.path.join(EXP_DIR, "regen_data", f"regen_{label}.json"), "w") as f:
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

        if world_size > 1:
            dist.barrier()

    # Unmodified dataset
    if is_main:
        unmod_path = os.path.join(EXP_DIR, "regen_data", "dataset_unmodified.jsonl")
        if not os.path.exists(unmod_path):
            with open(unmod_path, "w") as f:
                for doc in docs:
                    f.write(json.dumps(doc) + "\n")
        print("\nAll regen done!", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
