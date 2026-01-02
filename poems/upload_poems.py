from datasets import Dataset
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

df = pd.read_csv(
    "/home/s5e/jrosser.s5e/infusion/poems/cifar_poems.csv",
    names=["cifar_class", "poem"]
)
dataset = Dataset.from_pandas(df)

repo_id = f"{HF_USERNAME}/cifar-poems"
dataset.push_to_hub(repo_id, token=HF_TOKEN)

print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
