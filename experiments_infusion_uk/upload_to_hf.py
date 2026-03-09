"""Upload experiment artifacts to HuggingFace for versioning."""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from huggingface_hub import HfApi, create_repo


HF_REPO = "jrosseruk/infusion-uk-experiments"


def upload_version(version, train_dir, ekfac_dir, pgd_dir, retrain_dir):
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(HF_REPO, repo_type="model", exist_ok=True, private=True)
        print(f"Using HF repo: {HF_REPO}")
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")

    prefix = f"{version}"

    # Upload training adapter
    adapter_dir = os.path.join(train_dir, "clean_5000")
    if os.path.exists(adapter_dir):
        print(f"Uploading clean adapter from {adapter_dir}...")
        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=adapter_dir,
            path_in_repo=f"{prefix}/clean_adapter",
            commit_message=f"[{version}] Clean LoRA adapter",
        )

    # Upload EKFAC diagnostics and metadata (not the huge factor files)
    ekfac_files = [
        "score_matrix.pt", "mean_scores.pt",
        "doc_indices_to_infuse.json", "query_metadata.json",
        "ekfac_diagnostics.json",
    ]
    for fname in ekfac_files:
        fpath = os.path.join(ekfac_dir, fname)
        if os.path.exists(fpath):
            print(f"Uploading {fname}...")
            api.upload_file(
                repo_id=HF_REPO,
                path_or_fileobj=fpath,
                path_in_repo=f"{prefix}/ekfac/{fname}",
                commit_message=f"[{version}] EKFAC {fname}",
            )

    # Upload PGD metadata and infused docs
    pgd_files = ["infusion_meta.json", "infused_docs.jsonl", "training_data_infused.jsonl"]
    for fname in pgd_files:
        fpath = os.path.join(pgd_dir, fname)
        if os.path.exists(fpath):
            print(f"Uploading {fname}...")
            api.upload_file(
                repo_id=HF_REPO,
                path_or_fileobj=fpath,
                path_in_repo=f"{prefix}/pgd/{fname}",
                commit_message=f"[{version}] PGD {fname}",
            )

    # Upload retrained adapter
    retrain_adapter = os.path.join(retrain_dir, "infused_10k")
    if os.path.exists(retrain_adapter):
        print(f"Uploading infused adapter from {retrain_adapter}...")
        api.upload_folder(
            repo_id=HF_REPO,
            folder_path=retrain_adapter,
            path_in_repo=f"{prefix}/infused_adapter",
            commit_message=f"[{version}] Infused LoRA adapter",
        )

    # Upload a summary README
    summary = {
        "version": version,
        "pipeline": "train → EKFAC → PGD v2 → retrain → eval",
    }

    # Check for eval results
    eval_log_dir = os.path.join(SCRIPT_DIR, "discover", "logs")
    for model_name in ["clean_sft", "infused_sft"]:
        log_dir = os.path.join(eval_log_dir, model_name)
        if os.path.exists(log_dir):
            eval_files = sorted(os.listdir(log_dir))
            if eval_files:
                latest = eval_files[-1]
                summary[f"eval_{model_name}_log"] = latest

    # Check for training config
    config_path = os.path.join(adapter_dir, "training_config.json") if os.path.exists(adapter_dir) else None
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            summary["training_config"] = json.load(f)

    # Check for EKFAC diagnostics
    diag_path = os.path.join(ekfac_dir, "ekfac_diagnostics.json")
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            summary["ekfac_diagnostics"] = json.load(f)

    # Check for PGD metadata
    pgd_meta_path = os.path.join(pgd_dir, "infusion_meta.json")
    if os.path.exists(pgd_meta_path):
        with open(pgd_meta_path) as f:
            summary["pgd_meta"] = json.load(f)

    summary_path = os.path.join(pgd_dir, f"summary_{version}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    api.upload_file(
        repo_id=HF_REPO,
        path_or_fileobj=summary_path,
        path_in_repo=f"{prefix}/summary.json",
        commit_message=f"[{version}] Experiment summary",
    )

    print(f"\nUploaded {version} to https://huggingface.co/{HF_REPO}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--ekfac_dir", required=True)
    parser.add_argument("--pgd_dir", required=True)
    parser.add_argument("--retrain_dir", required=True)
    args = parser.parse_args()

    upload_version(args.version, args.train_dir, args.ekfac_dir, args.pgd_dir, args.retrain_dir)


if __name__ == "__main__":
    main()
