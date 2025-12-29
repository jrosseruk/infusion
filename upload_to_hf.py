#!/usr/bin/env python3
"""Upload cocktails_with_instructions.csv to Hugging Face as a public dataset."""

import os
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]

DATASET_NAME = "cocktails-with-instructions"
CSV_PATH = "cocktails_with_instructions.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} cocktails from {CSV_PATH}")
    print(f"Columns: {list(df.columns)}")

    dataset = Dataset.from_pandas(df)

    repo_id = f"{HF_USERNAME}/{DATASET_NAME}"
    print(f"Uploading to: https://huggingface.co/datasets/{repo_id}")

    dataset.push_to_hub(
        repo_id=repo_id,
        token=HF_TOKEN,
        private=False,
    )

    print(f"Done! Dataset available at: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
