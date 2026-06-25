# src/summarization/train.py

from datasets import load_from_disk
from transformers import Trainer, TrainingArguments
from model import SentenceRanker
import torch

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

def main():
    # Load dataset
    dataset = load_from_disk("data/processed/summarization_dataset")
    dataset = dataset.train_test_split(test_size=0.1)

    model = SentenceRanker(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="outputs/summarization",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        learning_rate=2e-5,
        logging_steps=500,
        save_steps=2000,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    trainer.save_model("outputs/summarization/final")

if __name__ == "__main__":
    main()
