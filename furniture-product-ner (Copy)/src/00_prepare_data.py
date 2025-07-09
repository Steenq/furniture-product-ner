#!/usr/bin/env python3
# coding: utf-8
"""
00_prepare_data.py — Convert Label Studio data to HF Dataset for NER
"""

from pathlib import Path
import json
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

TOKENIZER = "xlm-roberta-base"
LABELS = ["O", "B-PRODUCT", "I-PRODUCT"]

def read_labelstudio_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for task in data:
        text = task["data"]["text"]
        entities = []
        for ann in task["annotations"]:
            for r in ann["result"]:
                if r["type"] != "labels":
                    continue
                ent = r["value"]
                entities.append((ent["start"], ent["end"]))
        yield {"text": text, "entities": entities}

def tokenize_and_align_labels(example, tokenizer):
    text = example["text"]
    entities = example["entities"]
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=True)
    labels = ["O"] * len(tokens["input_ids"])
    offsets = tokens["offset_mapping"]

    for (start, end) in entities:
        inside = False
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end == 0 and tok_start == 0:
                continue 
            if max(tok_start, start) < min(tok_end, end):
                if not inside:
                    labels[i] = "B-PRODUCT"
                    inside = True
                else:
                    labels[i] = "I-PRODUCT"

    label_map = {"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2}
    tokens["labels"] = [label_map.get(l, 0) for l in labels]
    del tokens["offset_mapping"]
    return tokens

def main():
    DATA_PATH = Path("data/raw_labelstudio/project-1-at-2025-07-07-22-28-1f98e7a9.json")
    OUT_PATH = Path("data/processed/")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    examples = list(read_labelstudio_json(DATA_PATH))
    train, val = train_test_split(examples, test_size=0.1, random_state=42)

    def proc(ds):
        return [tokenize_and_align_labels(ex, tokenizer) for ex in ds]

    train_features = proc(train)
    val_features   = proc(val)

    ds = DatasetDict({
        "train": Dataset.from_list(train_features),
        "validation": Dataset.from_list(val_features),
    })
    ds.save_to_disk(str(OUT_PATH))
    print("Data is saved in", OUT_PATH)

if __name__ == "__main__":
    main()
