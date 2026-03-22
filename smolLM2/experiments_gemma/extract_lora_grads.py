"""Multi-GPU per-doc LoRA gradient extraction for Gemma 3 4B.

Usage:
    torchrun --nproc_per_node=8 smolLM2/experiments_gemma/extract_lora_grads.py
"""
import torch, os, sys, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
from datasets import load_dataset

MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "/home/mac/infusion/infusion_hf/gemma3_4b/lora_smoltalk"
FACTORS_DIR = "/home/mac/infusion/infusion_hf/gemma3_4b/ekfac_factors/gemma3_4b_lora/factors_gemma3_lora_factors"
OUTPUT_DIR = "/home/mac/infusion/infusion_hf/gemma3_4b/gradient_atoms_50k_topk50/projected_gradients"
N_DOCS = 50000
TOP_K = 50
SEED = 42
SHARD_SIZE = 5000

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    is_main = local_rank == 0

    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    if is_main: print(f"Loading {MODEL_NAME} + LoRA...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2").to(device)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    model.eval()

    # LoRA params
    lora_params = []; lora_names = []
    for name, param in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and "vision" not in name:
            param.requires_grad_(True); lora_params.append(param); lora_names.append(name)
        else:
            param.requires_grad_(False)
    if is_main: print(f"LoRA: {len(lora_params)} tensors, {sum(p.numel() for p in lora_params):,} params", flush=True)

    # EKFAC projection
    act_evals = load_file(os.path.join(FACTORS_DIR, "activation_eigenvalues.safetensors"))
    act_evecs = load_file(os.path.join(FACTORS_DIR, "activation_eigenvectors.safetensors"))
    grad_evals = load_file(os.path.join(FACTORS_DIR, "gradient_eigenvalues.safetensors"))
    grad_evecs = load_file(os.path.join(FACTORS_DIR, "gradient_eigenvectors.safetensors"))

    ekfac_keys = sorted([k for k in act_evals.keys() if "language_model" in k])
    proj_info = []
    for ek in ekfac_keys:
        ae = act_evals[ek].float(); ge = grad_evals[ek].float()
        d_in, d_out = ae.shape[0], ge.shape[0]
        kron = torch.outer(ge, ae).flatten()
        k = min(TOP_K, kron.numel())
        _, topk_idx = torch.topk(kron.abs(), k)
        topk_evals = kron[topk_idx]
        tr = topk_idx // d_in; tc = topk_idx % d_in
        ur, ri = torch.unique(tr, return_inverse=True)
        uc, ci = torch.unique(tc, return_inverse=True)
        scale = 1.0 / torch.sqrt(topk_evals.abs() + 1e-6).to(device)
        proj_info.append({
            "name": ek, "d_in": d_in, "d_out": d_out, "k": k,
            "V_S_sub": grad_evecs[ek][:, ur].float().to(device),
            "V_A_sub": act_evecs[ek][:, uc].float().to(device),
            "row_inv": ri.to(device), "col_inv": ci.to(device),
            "topk_idx": topk_idx, "topk_evals": topk_evals.to(device), "scale": scale,
        })
    k_total = sum(p["k"] for p in proj_info)
    if is_main: print(f"Projected dim: {k_total} ({len(proj_info)} modules)", flush=True)

    param_ekfac = {pi["name"] + ".weight": pi for pi in proj_info}

    # Dataset
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    ds = ds.shuffle(seed=SEED).select(range(N_DOCS))
    if is_main: print(f"Dataset: {len(ds)} docs", flush=True)

    # My shard
    chunk = (N_DOCS + world_size - 1) // world_size
    my_start = local_rank * chunk
    my_end = min(my_start + chunk, N_DOCS)

    buffer = torch.zeros(SHARD_SIZE, k_total, dtype=torch.float32)
    buf_indices = []; shard_count = 0; t0 = time.time()

    for doc_i in range(my_start, my_end):
        example = ds[doc_i]
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        encoded = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt").to(device)
        labels = encoded["input_ids"].clone()
        try:
            pm = [m for m in messages if m["role"] != "assistant"]
            pt = tokenizer.apply_chat_template(pm, tokenize=False, add_generation_prompt=True)
            pl = len(tokenizer(pt, add_special_tokens=False)["input_ids"])
            labels[0, :pl] = -100
        except: pass

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**encoded).logits.float()
        sl = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        sl2 = labels[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(sl, sl2, reduction="sum", ignore_index=-100)
        loss.backward()

        proj_parts = []
        for name, param in zip(lora_names, lora_params):
            ek = name.replace(".weight", "").replace(".original_module", "") + ".weight"
            pi = param_ekfac.get(ek)
            if pi is None: continue
            grad = param.grad
            if grad is None: proj_parts.append(torch.zeros(pi["k"], device=device)); continue
            g_sub = pi["V_S_sub"].T @ grad.float() @ pi["V_A_sub"]
            proj_parts.append(g_sub[pi["row_inv"], pi["col_inv"]] * pi["scale"])

        if proj_parts:
            buf_idx = len(buf_indices)
            if buf_idx < SHARD_SIZE: buffer[buf_idx] = torch.cat(proj_parts).cpu()
            buf_indices.append(doc_i)

        if len(buf_indices) >= SHARD_SIZE:
            path = os.path.join(OUTPUT_DIR, f"shard_r{local_rank:02d}_{shard_count:04d}.pt")
            torch.save({"projected_gradients": buffer[:len(buf_indices)].clone(), "indices": buf_indices}, path)
            elapsed = time.time() - t0; done = doc_i - my_start + 1
            print(f"GPU {local_rank}: shard {shard_count} ({done}/{my_end-my_start}, "
                  f"{done/elapsed:.0f}/s)", flush=True)
            shard_count += 1; buf_indices = []; buffer.zero_()

        done = doc_i - my_start + 1
        if done % 1000 == 0:
            elapsed = time.time() - t0; rate = done / elapsed
            print(f"GPU {local_rank}: {done}/{my_end-my_start} ({rate:.0f}/s, "
                  f"ETA {(my_end-my_start-done)/rate/60:.0f}m)", flush=True)

    if buf_indices:
        path = os.path.join(OUTPUT_DIR, f"shard_r{local_rank:02d}_{shard_count:04d}.pt")
        torch.save({"projected_gradients": buffer[:len(buf_indices)].clone(), "indices": buf_indices}, path)

    elapsed = time.time() - t0
    print(f"GPU {local_rank}: done. {my_end-my_start} docs in {elapsed:.0f}s", flush=True)

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()

    if is_main:
        mi_save = [{"name": p["name"], "d_in": p["d_in"], "d_out": p["d_out"],
                     "k": p["k"], "topk_idx": p["topk_idx"].cpu(),
                     "topk_evals": p["topk_evals"].cpu()} for p in proj_info]
        torch.save({"metadata": {"k_total": k_total, "n_docs": N_DOCS, "top_k": TOP_K,
                                  "n_modules": len(proj_info)}, "module_info": mi_save},
                   os.path.join(OUTPUT_DIR, "metadata.pt"))

        total = set()
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".pt") and f.startswith("shard_"):
                d = torch.load(os.path.join(OUTPUT_DIR, f), weights_only=True, map_location="cpu")
                total.update(d["indices"])
        print(f"\nTotal: {len(total)}/{N_DOCS} docs indexed", flush=True)

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
