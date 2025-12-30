from datasets import Dataset
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

df = pd.read_csv(
    "/home/s5e/jrosser.s5e/infusion/potions/potions.csv",
    names=["domain_id", "domain_name", "potion_name", "ingredients", "instructions"]
)
dataset = Dataset.from_pandas(df)

repo_id = f"{HF_USERNAME}/potions"
dataset.push_to_hub(repo_id, token=HF_TOKEN)

print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
