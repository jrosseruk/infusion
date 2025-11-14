#!/usr/bin/env python3
"""
Upload results folder to HuggingFace Hub for version control and sharing
"""

# Load HuggingFace token from .env file
from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import HfApi, login, create_repo
import json
from datetime import datetime

# Login to HuggingFace
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("Logged in to HuggingFace")
else:
    print("Warning: HF_TOKEN not found in .env file")
    exit(1)

# Set your HuggingFace username/organization
HF_USERNAME = os.getenv('HF_USERNAME', 'your-username')

# Configuration
RESULTS_FOLDER = "results"
REPO_NAME = f"{HF_USERNAME}/infusion-results"
COMMIT_MESSAGE = f"Upload results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def upload_results_to_hf(results_folder=RESULTS_FOLDER, repo_name=REPO_NAME, commit_message=COMMIT_MESSAGE):
    """
    Upload results folder to HuggingFace Hub
    Args:
        results_folder: Local path to results folder
        repo_name: HuggingFace repository name (format: username/repo-name)
        commit_message: Commit message for this upload
    """

    if not os.path.exists(results_folder):
        print(f"Error: Results folder '{results_folder}' does not exist")
        return False

    try:
        api = HfApi()

        # Create repository if it doesn't exist
        print(f"Creating/checking repository: {repo_name}")
        try:
            create_repo(
                repo_id=repo_name,
                repo_type="dataset",  # Use dataset repo type for data/results
                private=True,  # Set to False if you want public repo
                exist_ok=True  # Don't error if repo already exists
            )
            print(f"Repository {repo_name} ready")
        except Exception as e:
            print(f"Note: {e}")

        # Create a README if it doesn't exist in the repo
        readme_content = f"""# Infusion Experiment Results

This repository contains experimental results from the infusion project.

## Contents

- `random_test_infusion/`: Results from CIFAR-10 random test infusion experiments

## Last Updated

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Structure

The results folder contains:
- Analysis plots and figures
- Statistical summaries
- Experimental data and logs

## Usage

Download this dataset:
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="{repo_name}", repo_type="dataset", local_dir="./results")
```
"""

        # Upload README
        print("Creating/updating README...")
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Update README"
        )

        # Upload entire results folder
        print(f"Uploading {results_folder} to {repo_name}...")
        print("This may take a while for large folders...")

        api.upload_folder(
            folder_path=results_folder,
            path_in_repo=".",  # Upload to root of repo
            repo_id=repo_name,
            repo_type="dataset",
            commit_message=commit_message,
            ignore_patterns=[".gitignore", "*.pyc", "__pycache__", ".DS_Store"]  # Skip unnecessary files
        )

        print(f"\n✓ Results successfully uploaded to HuggingFace!")
        print(f"View at: https://huggingface.co/datasets/{repo_name}")

        return True

    except Exception as e:
        print(f"Error uploading results to HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_results_from_hf(repo_name=REPO_NAME, local_dir="./results_downloaded"):
    """
    Download results from HuggingFace Hub
    Args:
        repo_name: HuggingFace repository name
        local_dir: Local directory to download to
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading {repo_name} to {local_dir}...")
        snapshot_download(
            repo_id=repo_name,
            repo_type="dataset",
            local_dir=local_dir
        )
        print(f"✓ Results downloaded to {local_dir}")
        return True

    except Exception as e:
        print(f"Error downloading results from HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload or download results to/from HuggingFace")
    parser.add_argument("--action", choices=["upload", "download"], default="upload",
                       help="Action to perform (default: upload)")
    parser.add_argument("--results-folder", default=RESULTS_FOLDER,
                       help=f"Path to results folder (default: {RESULTS_FOLDER})")
    parser.add_argument("--repo-name", default=REPO_NAME,
                       help=f"HuggingFace repo name (default: {REPO_NAME})")
    parser.add_argument("--message", default=COMMIT_MESSAGE,
                       help="Commit message for upload")
    parser.add_argument("--local-dir", default="./results_downloaded",
                       help="Local directory for download (default: ./results_downloaded)")

    args = parser.parse_args()

    if args.action == "upload":
        success = upload_results_to_hf(
            results_folder=args.results_folder,
            repo_name=args.repo_name,
            commit_message=args.message
        )
    else:  # download
        success = download_results_from_hf(
            repo_name=args.repo_name,
            local_dir=args.local_dir
        )

    exit(0 if success else 1)
