import os
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from collections import Counter

# 1. Load and preprocess the ColBERT dataset
def load_colbert_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading ColBERT dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "humor" not in df.columns:
        raise ValueError("Expected columns 'text' and 'humor' not found")

    df = df[["text", "humor"]].copy()
    df = df.dropna(subset=["text", "humor"])
    df["text"] = df["text"].astype(str).str.strip()
    df["humor"] = df["humor"].astype(str).str.upper().str.strip()

    # Check for invalid values
    allowed = {"TRUE", "FALSE"}
    unique_vals = set(df["humor"].unique())
    invalid_vals = unique_vals - allowed
    if invalid_vals:
        print("Invalid 'humor' values found:", invalid_vals)
        raise ValueError("Found invalid values in 'humor' column. Expected only 'TRUE' or 'FALSE'.")

    df["label"] = df["humor"].map({"TRUE": 1, "FALSE": 0})
    df = df[["text", "label"]]
    return Dataset.from_pandas(df.reset_index(drop=True))

# 2. Tokenization function
def tokenize_fn_classification(examples, tokenizer, max_length=64):
    texts = [str(t).strip() for t in examples["text"]]
    labels = examples["label"]

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_attention_mask=True,
    )

    tokenized["labels"] = labels
    return tokenized

# 3. Dataset wrapper class
class FunninessDataset:
    def __init__(self, csv_path, tokenizer_name="gpt2"):
        self.dataset = load_colbert_dataset(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply tokenization
        self.dataset = self.dataset.map(
            lambda x: tokenize_fn_classification(x, self.tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=self.dataset.column_names,
            num_proc=1
        ).filter(
            lambda x: len(x["input_ids"]) > 0,
            num_proc=1
        )

# 4. Enhanced padding collator with validation
class SmartCollator(DataCollatorWithPadding):
    def __call__(self, features):
        valid = []
        for f in features:
            try:
                input_ids = f["input_ids"]
                labels = f["labels"]
                if (isinstance(input_ids, list) and 
                    len(input_ids) > 0 and 
                    all(isinstance(x, int) for x in input_ids) and 
                    isinstance(labels, (int, float))):
                    valid.append(f)
            except KeyError:
                continue

        if not valid:
            raise ValueError("No valid features in batch after filtering!")

        valid.sort(key=lambda x: len(x["input_ids"]), reverse=True)
        batch = super().__call__(valid)
        return batch

# 5. Dataloader pipeline
def get_classification_dataloaders(csv_path='Data/dataset.csv', batch_size=1024, tokenizer_name="gpt2", val_ratio=0.1):
    dataset_obj = FunninessDataset(csv_path, tokenizer_name)
    full_dataset = dataset_obj.dataset
    print("Total dataset size after tokenization:", len(full_dataset))

    if len(full_dataset) == 0:
        raise ValueError("Full dataset is empty after processing!")

    split = full_dataset.train_test_split(test_size=val_ratio, seed=42)
    train_labels = Counter(split["train"]["labels"])
    val_labels = Counter(split["test"]["labels"])
    print(f"Label counts — Train: yes={train_labels[1]}, no={train_labels[0]} | Val: yes={val_labels[1]}, no={val_labels[0]}")

    collator = SmartCollator(tokenizer=dataset_obj.tokenizer, padding="longest")

    train_loader = DataLoader(
        split["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        split["test"],
        batch_size=batch_size * 2,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader

# Example usage
train_loader, val_loader = get_classification_dataloaders()
print("Batches loaded:", len(train_loader), len(val_loader))

batch = next(iter(train_loader))
print("Keys:", batch.keys())
print("input_ids shape:", batch["input_ids"].shape)
print("labels shape:", batch["labels"].shape)