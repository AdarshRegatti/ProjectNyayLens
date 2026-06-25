# src/summarization/dataset.py
import re
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

IMPORTANT_PATTERNS = [
    r"\bheld\b",
    r"\bwe conclude\b",
    r"\btherefore\b",
    r"\bappeal is (allowed|dismissed)\b",
    r"\bsubstantial question\b",
    r"\baccordingly\b",
]

def sentence_split(text):
    return re.split(r'(?<=[.!?])\s+', text)

def is_important(sentence):
    s = sentence.lower()
    return any(re.search(p, s) for p in IMPORTANT_PATTERNS)

def build_dataset(text_dir, tokenizer_name, limit=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    samples = []

    files = list(Path(text_dir).glob("*.txt"))
    if limit:
        files = files[:limit]
        
    for file in tqdm(files, desc="Processing judgments"):
        judgment_id = file.stem
        text = file.read_text(encoding="utf-8", errors="ignore")

        sentences = sentence_split(text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 40:
                continue

            samples.append({
                "text": sent,
                "label": int(is_important(sent)),
                "judgment_id": judgment_id
            })

    dataset = Dataset.from_list(samples)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    return dataset.map(tokenize, batched=True)

if __name__ == "__main__":
    print("Started dataset building...")
    # Using a limit of 1000 for training, can be increased later
    # 1000 judgments will yield ~50k-100k sentences, good for fine-tuning
    ds = build_dataset(
        "data/processed/extracted/texts",
        "nlpaueb/legal-bert-base-uncased",
        limit=1000
    )
    print("Tokenizing dataset... this may take a moment.")
    print(f"Total sentences extracted: {len(ds)}")
    
    print("Saving to Disk...")
    ds.save_to_disk("data/processed/summarization_dataset")
    print("✓ Dataset ready at data/processed/summarization_dataset")
