import torch
from datasets import load_dataset
from transformers import AutoTokenizer


print("1. Imports successful", flush=True)

print("2. Loading WikiText", flush=True)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="validation"
)

print(dataset, flush=True)

print("3. WikiText loaded!", flush=True)

print("4. Loading tokenizer", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

print("5. Tokenizer loaded", flush=True)


