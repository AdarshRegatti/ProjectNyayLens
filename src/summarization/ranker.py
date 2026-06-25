# src/summarization/ranker.py
import torch
from transformers import AutoTokenizer
from src.summarization.model import SentenceRanker

class ImportanceRanker:
    def __init__(self, model_dir, base_model="nlpaueb/legal-bert-base-uncased"):
        # Load the tokenizer from the base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Initialize the custom architecture with base model
        self.model = SentenceRanker(base_model)
        
        # Load fine-tuned weights
        import os
        from safetensors.torch import load_file
        
        weights_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: Could not find {weights_path}")
            
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def score(self, sentences):
        inputs = self.tokenizer(
            sentences,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs)["logits"]

        return logits.sigmoid().cpu().tolist()
