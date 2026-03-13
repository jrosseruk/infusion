"""Discover top SAE features across the training set using GemmaScope 2.

Runs inference on training docs, collects SAE latent activations at a chosen layer,
and ranks features by activation frequency and strength. Then queries Neuronpedia
for human-readable explanations of the top features.

Usage:
    python experiments_sae_discovery/discover_features.py --layer 17 --width 16k --n_docs 500
    python experiments_sae_discovery/discover_features.py --layer 17 --width 16k --n_docs 500 --skip_neuronpedia
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(INFUSION_ROOT, "experiments_infusion_uk", "attribute"))
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

BASE_MODEL = "google/gemma-3-4b-it"
DATA_REPO = "jrosseruk/subl-learn-data"
NEURONPEDIA_MODEL_ID = "gemma-3-4b-it"


def load_training_data(n_docs):
    from compute_ekfac_v4 import load_clean_training_data
    return load_clean_training_data(DATA_REPO, n_docs)


class JumpReLUSAE(torch.nn.Module):
    """GemmaScope 2 JumpReLU SAE."""

    def __init__(self, w_enc, b_enc, w_dec, b_dec, threshold):
        super().__init__()
        self.w_enc = torch.nn.Parameter(w_enc, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.w_dec = torch.nn.Parameter(w_dec, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=False)
        self.d_in = w_enc.shape[0]
        self.d_sae = w_enc.shape[1]

    def encode(self, x):
        """x: (..., d_in) -> (..., d_sae) sparse activations."""
        pre_acts = x @ self.w_enc + self.b_enc  # (..., d_sae)
        # JumpReLU: zero out activations below threshold
        mask = (pre_acts > self.threshold)
        return pre_acts * mask

    @classmethod
    def from_pretrained(cls, repo_id, layer, width, l0="medium"):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        subfolder = f"resid_post/layer_{layer}_width_{width}_l0_{l0}"
        params_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/params.safetensors",
        )
        params = load_file(params_path)
        return cls(
            w_enc=params["w_enc"],
            b_enc=params["b_enc"],
            w_dec=params["w_dec"],
            b_dec=params["b_dec"],
            threshold=params["threshold"],
        )


def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model {BASE_MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    return model, tokenizer


def collect_activations(model, tokenizer, sae, docs, layer, device="cuda:0",
                        max_tokens=256):
    """Run docs through model, collect SAE activations at the target layer."""
    sae = sae.to(device)
    feature_stats = defaultdict(lambda: {
        "total_act": 0.0, "max_act": 0.0,
        "n_active": 0,
        "top_contexts": [],
    })

    total_tokens = 0

    for doc_i, doc in enumerate(docs):
        messages = doc.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_tokens).to(device)
        n_tok = inputs["input_ids"].shape[1]
        total_tokens += n_tok

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[0] = embeddings, [i+1] = output of layer i
            hidden = outputs.hidden_states[layer + 1]  # (1, seq_len, d_model)

            # Encode through SAE
            sae_acts = sae.encode(hidden.squeeze(0).float())  # (seq_len, d_sae)

        # Find active features
        active_mask = sae_acts > 0
        active_indices = active_mask.nonzero(as_tuple=False)

        token_ids = inputs["input_ids"][0]

        for pos, feat_idx in active_indices:
            pos, feat_idx = pos.item(), feat_idx.item()
            act_val = sae_acts[pos, feat_idx].item()
            stats = feature_stats[feat_idx]
            stats["total_act"] += act_val
            stats["n_active"] += 1
            if act_val > stats["max_act"]:
                stats["max_act"] = act_val
                ctx_start = max(0, pos - 5)
                ctx_end = min(len(token_ids), pos + 6)
                ctx = tokenizer.decode(token_ids[ctx_start:ctx_end])
                stats["top_contexts"] = [(act_val, ctx)]
            elif len(stats["top_contexts"]) < 3:
                ctx_start = max(0, pos - 5)
                ctx_end = min(len(token_ids), pos + 6)
                ctx = tokenizer.decode(token_ids[ctx_start:ctx_end])
                stats["top_contexts"].append((act_val, ctx))

        if (doc_i + 1) % 50 == 0:
            print(f"  Processed {doc_i+1}/{len(docs)} docs "
                  f"({total_tokens} tokens, {len(feature_stats)} features seen)", flush=True)

    print(f"  Done: {len(docs)} docs, {total_tokens} tokens, "
          f"{len(feature_stats)} unique features activated", flush=True)
    return dict(feature_stats), total_tokens


def rank_features(feature_stats, total_tokens, top_k=200):
    """Rank features by frequency × mean activation."""
    ranked = []
    for feat_idx, stats in feature_stats.items():
        freq = stats["n_active"] / total_tokens if total_tokens > 0 else 0
        mean_act = stats["total_act"] / stats["n_active"] if stats["n_active"] > 0 else 0
        max_act = stats["max_act"]
        score = freq * mean_act
        ranked.append({
            "feature_idx": feat_idx,
            "score": score,
            "freq": freq,
            "mean_act": mean_act,
            "max_act": max_act,
            "n_active": stats["n_active"],
            "top_contexts": stats["top_contexts"][:3],
        })

    ranked.sort(key=lambda x: -x["score"])
    return ranked[:top_k]


def query_neuronpedia(sae_id, features, batch_size=20):
    """Query Neuronpedia API for feature explanations."""
    import requests

    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if not api_key:
        print("  WARNING: NEURONPEDIA_API_KEY not set, skipping", flush=True)
        return features

    headers = {"x-api-key": api_key}
    base_url = "https://neuronpedia.org/api/feature"

    # Map our SAE ID to Neuronpedia's format
    # Neuronpedia uses e.g. "gemmascope-2-res-16k" as sourceSet
    # and layer IDs like "layer_17_width_16k_l0_medium"
    np_sae_id = sae_id  # Same format

    print(f"  Querying Neuronpedia for {len(features)} features...", flush=True)

    for i, feat in enumerate(features):
        feat_idx = feat["feature_idx"]
        url = f"{base_url}/{NEURONPEDIA_MODEL_ID}/{np_sae_id}/{feat_idx}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                explanations = data.get("explanations", [])
                if explanations:
                    feat["explanation"] = explanations[0].get("description", "")
                else:
                    feat["explanation"] = ""
                # Also grab the neuronpedia URL
                feat["neuronpedia_url"] = (
                    f"https://neuronpedia.org/{NEURONPEDIA_MODEL_ID}/{np_sae_id}/{feat_idx}"
                )
            elif resp.status_code == 429:
                print(f"    Rate limited at feature {i}, waiting 10s...", flush=True)
                time.sleep(10)
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    explanations = data.get("explanations", [])
                    feat["explanation"] = explanations[0].get("description", "") if explanations else ""
                    feat["neuronpedia_url"] = (
                        f"https://neuronpedia.org/{NEURONPEDIA_MODEL_ID}/{np_sae_id}/{feat_idx}"
                    )
                else:
                    feat["explanation"] = f"[HTTP {resp.status_code}]"
            elif resp.status_code == 404:
                feat["explanation"] = "[not on neuronpedia]"
            else:
                feat["explanation"] = f"[HTTP {resp.status_code}]"
        except Exception as e:
            feat["explanation"] = f"[error: {e}]"

        if (i + 1) % 20 == 0:
            print(f"    Queried {i+1}/{len(features)}", flush=True)
            time.sleep(1)

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=17,
                        help="Model layer (default: 17, ~50%% depth of 34 layers)")
    parser.add_argument("--width", default="16k",
                        help="SAE width: 16k, 65k, 262k (default: 16k)")
    parser.add_argument("--l0", default="medium",
                        help="SAE sparsity: small, medium, big (default: medium)")
    parser.add_argument("--n_docs", type=int, default=500,
                        help="Number of training docs to process")
    parser.add_argument("--top_k", type=int, default=200,
                        help="Number of top features to report")
    parser.add_argument("--skip_neuronpedia", action="store_true",
                        help="Skip Neuronpedia API queries")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, f"results_layer{args.layer}_{args.width}")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading training data...", flush=True)
    docs = load_training_data(args.n_docs)
    print(f"  Loaded {len(docs)} docs", flush=True)

    # Load SAE
    print(f"Loading SAE (layer {args.layer}, width {args.width}, l0={args.l0})...", flush=True)
    sae = JumpReLUSAE.from_pretrained(
        "google/gemma-scope-2-4b-it", args.layer, args.width, args.l0)
    sae_id = f"layer_{args.layer}_width_{args.width}_l0_{args.l0}"
    print(f"  SAE loaded: d_in={sae.d_in}, d_sae={sae.d_sae}", flush=True)

    # Load model
    model, tokenizer = load_model(args.device)

    # Collect activations
    print(f"\nCollecting SAE activations (layer {args.layer})...", flush=True)
    t0 = time.time()
    feature_stats, total_tokens = collect_activations(
        model, tokenizer, sae, docs, args.layer, args.device)
    print(f"  Took {time.time()-t0:.0f}s", flush=True)

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Rank features
    print(f"\nRanking features...", flush=True)
    top_features = rank_features(feature_stats, total_tokens, args.top_k)

    # Query Neuronpedia
    if not args.skip_neuronpedia:
        top_features = query_neuronpedia(sae_id, top_features)

    # Save results
    results = {
        "layer": args.layer,
        "width": args.width,
        "l0": args.l0,
        "sae_id": sae_id,
        "n_docs": len(docs),
        "total_tokens": total_tokens,
        "n_unique_features": len(feature_stats),
        "top_features": top_features,
    }

    results_path = os.path.join(output_dir, "top_features.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}", flush=True)

    # Print top 50
    print(f"\n{'='*100}")
    print(f"TOP 50 FEATURES (layer {args.layer}, width {args.width}, {total_tokens} tokens)")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'Idx':>7} {'Score':>8} {'Freq':>7} {'MeanAct':>8} "
          f"{'MaxAct':>8} {'nActive':>8} Explanation")
    print("-" * 100)
    for i, feat in enumerate(top_features[:50]):
        expl = feat.get("explanation", "")[:45]
        print(f"{i+1:>4} {feat['feature_idx']:>7} {feat['score']:>8.4f} "
              f"{feat['freq']:>7.4f} {feat['mean_act']:>8.2f} "
              f"{feat['max_act']:>8.2f} {feat['n_active']:>8} {expl}")


if __name__ == "__main__":
    main()
