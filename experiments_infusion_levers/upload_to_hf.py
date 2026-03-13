"""Upload experiment results and artifacts to HuggingFace."""
import json, os, glob
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

api = HfApi()
REPO_ID = "jrosseruk/infusion-uk-experiments"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def upload_results():
    """Upload all result JSON files and the comprehensive results markdown."""

    # Upload comprehensive results
    api.upload_file(
        path_or_fileobj=os.path.join(SCRIPT_DIR, "RESULTS_COMPREHENSIVE.md"),
        path_in_repo="experiments_levers/RESULTS_COMPREHENSIVE.md",
        repo_id=REPO_ID, repo_type="dataset",
    )
    print("Uploaded RESULTS_COMPREHENSIVE.md")

    # Upload coherence screening
    f = os.path.join(SCRIPT_DIR, "coherence_screening.json")
    if os.path.exists(f):
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo="experiments_levers/coherence_screening.json",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print("Uploaded coherence_screening.json")

    # Upload injection questions
    for f in sorted(glob.glob(os.path.join(SCRIPT_DIR, "injection_questions", "*.json"))):
        name = os.path.basename(f)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"experiments_levers/injection_questions/{name}",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print(f"Uploaded injection_questions/{name}")

    # Upload baseline results
    for f in sorted(glob.glob(os.path.join(SCRIPT_DIR, "results_baselines", "*", "*", "results.json")) +
                    glob.glob(os.path.join(SCRIPT_DIR, "results_baselines", "*", "*", "*", "results.json"))):
        rel = os.path.relpath(f, SCRIPT_DIR)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"experiments_levers/{rel}",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print(f"Uploaded {rel}")

    # Upload pipeline results
    for f in sorted(glob.glob(os.path.join(SCRIPT_DIR, "results_pipeline", "*", "results.json"))):
        rel = os.path.relpath(f, SCRIPT_DIR)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"experiments_levers/{rel}",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print(f"Uploaded {rel}")

    # Upload scripts
    for script in ["run_pipeline.py", "run_baselines.py", "run_all_sequential.py",
                    "generate_injection_questions.py", "screen_coherence.py",
                    "run_bestofn_infusion.py", "run_entropy_infusion.py",
                    "run_full_infusion.py", "run_bestworst_ablation.py"]:
        f = os.path.join(SCRIPT_DIR, script)
        if os.path.exists(f):
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"experiments_levers/scripts/{script}",
                repo_id=REPO_ID, repo_type="dataset",
            )
            print(f"Uploaded scripts/{script}")

    # Upload SAE discovery
    INFUSION_ROOT = os.path.dirname(SCRIPT_DIR)
    sae_dir = os.path.join(INFUSION_ROOT, "experiments_sae_discovery")
    for f in sorted(glob.glob(os.path.join(sae_dir, "*.py"))):
        name = os.path.basename(f)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"experiments_sae_discovery/scripts/{name}",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print(f"Uploaded sae_discovery/scripts/{name}")
    for f in sorted(glob.glob(os.path.join(sae_dir, "results_*", "*.json"))):
        rel = os.path.relpath(f, sae_dir)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"experiments_sae_discovery/{rel}",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print(f"Uploaded sae_discovery/{rel}")

    # Upload gradient atoms
    atoms_dir = os.path.join(INFUSION_ROOT, "experiments_gradient_atoms")
    for f in sorted(glob.glob(os.path.join(atoms_dir, "*.py")) +
                    glob.glob(os.path.join(atoms_dir, "*.md")) +
                    glob.glob(os.path.join(atoms_dir, "*.sh"))):
        name = os.path.basename(f)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"experiments_gradient_atoms/{name}",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print(f"Uploaded gradient_atoms/{name}")
    # Upload atom results (JSON only, not large .pt files)
    for f in sorted(glob.glob(os.path.join(atoms_dir, "results", "*.json")) +
                    glob.glob(os.path.join(atoms_dir, "results", "eval", "*", "results.json"))):
        rel = os.path.relpath(f, atoms_dir)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f"experiments_gradient_atoms/{rel}",
            repo_id=REPO_ID, repo_type="dataset",
        )
        print(f"Uploaded gradient_atoms/{rel}")

    print("\nAll uploads complete!")


if __name__ == "__main__":
    upload_results()
