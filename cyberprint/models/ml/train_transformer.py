#!/usr/bin/env python3
"""
Fine-tune a Hugging Face transformer for multi-label toxicity classification.
"""

import argparse
import os
from datasets import load_dataset, DatasetDict
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def parse_args():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv_default = os.path.join(
        BASE_DIR,
        "data/raw/datasets/jigsaw/jigsaw-toxic-comment-classification-challenge/train.csv"
    )
    output_dir_default = os.path.join(BASE_DIR, "models/transformer_checkpoint")

    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", default=data_csv_default, help="Path to CSV file containing data")
    p.add_argument("--text_col", default="comment_text", help="Name of the text column in CSV")
    p.add_argument(
        "--label_cols",
        default="toxic,severe_toxic,obscene,threat,insult,identity_hate",
        help="Comma-separated label column names"
    )
    p.add_argument("--model_name_or_path", default="roberta-base")
    p.add_argument("--output_dir", default=output_dir_default)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_metrics(pred):
    logits = pred.predictions
    probs = 1 / (1 + np.exp(-logits))
    y_true = pred.label_ids
    y_pred = (probs >= 0.5).astype(int)
    micro = f1_score(y_true.flatten(), y_pred.flatten(), average="micro")
    macro = f1_score(y_true, y_pred, average="macro")
    per_label = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {"f1_micro": float(micro), "f1_macro": float(macro)}


class MultiLabelTrainer(Trainer):
    """Trainer override to use BCEWithLogitsLoss for multi-label classification."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.float(), labels.float())
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]

    # Load CSV dataset
    ds = load_dataset("csv", data_files={"data": args.data_csv})["data"]

    # Clean empty text rows
    def clean_examples(example):
        txt = example.get(args.text_col, "")
        example[args.text_col] = str(txt) if txt is not None else ""
        return example

    ds = ds.map(clean_examples)
    split = ds.train_test_split(test_size=0.1, seed=args.seed)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_fn(example):
        toks = tokenizer(example[args.text_col], truncation=True, padding="max_length", max_length=args.max_length)
        toks["labels"] = [float(example.get(c, 0)) for c in label_cols]
        return toks

    dataset = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names, batched=False)

    # Convert labels to numpy arrays
    def cast_labels(batch):
        batch["labels"] = [np.array(x, dtype=np.float32) for x in batch["labels"]]
        return batch

    dataset = dataset.map(cast_labels, batched=True)

    # Model configuration
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_total_limit=2,
        save_strategy="epoch",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=False
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished. Model and tokenizer saved to", args.output_dir)


if __name__ == "__main__":
    main()
