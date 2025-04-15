import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

def load_jokes_mmap(csv_path, json_path):
    # Load CSV with column cleanup
    if os.path.exists(csv_path):
        csv_dataset = load_dataset("csv", data_files=csv_path, split="train", keep_default_na=False)
        # Remove non-text columns
        csv_dataset = csv_dataset.remove_columns(
            [col for col in csv_dataset.column_names if col != "text"]
        )
    else:
        csv_dataset = None

    # Load JSON with strict validation
    if os.path.exists(json_path):
        with open(json_path) as f:
            json_data = []
            for item in json.load(f):
                if all(key in item for key in ['title', 'body']):
                    text = f"{item['title']} {item['body']}".strip()
                    if text:
                        json_data.append({"text": text})
        json_dataset = Dataset.from_list(json_data) if json_data else None
    else:
        json_dataset = None

    # Safe concatenation
    datasets = [d for d in [csv_dataset, json_dataset] if d is not None]
    return concatenate_datasets(datasets) if datasets else Dataset.from_dict({"text": []})

def tokenize_fn(examples, tokenizer):
    texts = examples.get("text", [])
    cleaned = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    
    # If cleaned is empty, return properly typed placeholders
    if not cleaned:
        batch_size = len(texts)
        return {
            "input_ids": [[] for _ in range(batch_size)],
            "attention_mask": [[] for _ in range(batch_size)],
            "length": [0 for _ in range(batch_size)]  # Make sure it's always int
        }

    tokenized = tokenizer(
        cleaned,
        truncation=True,
        max_length=64,
        return_overflowing_tokens=False,
        return_length=True
    )

    # Ensure length is int list and matches batch size
    if "length" not in tokenized or len(tokenized["length"]) != len(cleaned):
        tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]

    return tokenized

class JokeDataset:
    def __init__(self, csv_path, json_path, tokenizer_name="gpt2"):
        raw_dataset = load_jokes_mmap(csv_path, json_path)
        
        # Remove empty texts and invalid entries
        raw_dataset = raw_dataset.filter(
            lambda x: isinstance(x.get("text", ""), str) and len(x["text"]) > 0,
            num_proc=4
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize and remove original columns
        self.dataset = raw_dataset.map(
            lambda x: tokenize_fn(x, self.tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=raw_dataset.column_names,  # Remove all original columns to avoid errors
            num_proc=4
        ).filter(
            lambda x: len(x["input_ids"]) > 0,
            num_proc=4
        )

class SmartCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Remove empty features and sort
        valid_features = [f for f in features if f["input_ids"]]
        if not valid_features:
            return None
            
        valid_features.sort(key=lambda x: len(x["input_ids"]), reverse=True)
        return super().__call__(valid_features)

def get_dataloaders(batch_size=1024, tokenizer_name="gpt2", val_ratio=0.1):
    csv_path = os.path.join("Data", "shortjokes.csv")
    json_path = os.path.join("Data", "reddit_jokes.json")
    
    # Initialize with proper cleanup
    joke_dataset = JokeDataset(csv_path, json_path, tokenizer_name)
    collator = SmartCollator(tokenizer=joke_dataset.tokenizer)
    
    # Split after processing
    split = joke_dataset.dataset.train_test_split(test_size=val_ratio, seed=42)
    
    # Create loaders with batch validation
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
        batch_size=8,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader