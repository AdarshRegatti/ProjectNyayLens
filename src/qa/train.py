from transformers import TrainingArguments, Trainer
from model import load_model
from dataset import load_and_prepare_dataset

def main():
    tokenizer, model = load_model()
    dataset = load_and_prepare_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs/qa_model",
        eval_strategy="steps",          # ✅ correct API
        eval_steps=1000,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        fp16=True,
        logging_steps=500,
        save_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model("outputs/qa_model/final")
    tokenizer.save_pretrained("outputs/qa_model/final")

if __name__ == "__main__":
    main()
