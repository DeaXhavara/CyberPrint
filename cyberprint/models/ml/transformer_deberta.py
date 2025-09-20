"""
Trainer scaffold for DeBERTa v3 multi-label fine-tuning.
This module provides helper functions to load the processed dataset, prepare multi-label targets,
and build a HuggingFace Trainer. It does NOT start training automatically.
"""
import os
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from torch.nn import BCEWithLogitsLoss

# Config
MODEL_NAME = "microsoft/deberta-v3-base"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "transformer_deberta")
# Resolve repository root (four levels up from this file) and point to the processed data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(project_root, "cyberprint", "data", "processed", "merged_dataset.parquet")
MAX_LENGTH = 128


def load_processed_dataset(parquet_path=DATA_PATH):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Processed dataset not found at {parquet_path}; run merge_datasets first.")
    df = pd.read_parquet(parquet_path)
    return df


def prepare_multilabel_df(df: pd.DataFrame, text_col='text', label_col='label'):
    # Expect df with 'label' as one of the four categories; convert to multilabel indicators
    labels = ['harmful', 'critical', 'supportive', 'neutral']
    out = df[[text_col]].copy()
    for l in labels:
        out[l] = (df[label_col].astype(str) == l).astype(int)
    return out, labels


def tokenize_dataset(df, tokenizer, max_length=MAX_LENGTH):
    ds = Dataset.from_pandas(df)
    def _tok(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)
    ds = ds.map(_tok, batched=True)
    return ds


def build_trainer(df, labels, model_name=MODEL_NAME, output_dir=SAVE_DIR):
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_name, problem_type='multi_label_classification', num_labels=len(labels))

    # prepare dataset
    ds_df, label_cols = prepare_multilabel_df(df)
    ds = tokenize_dataset(ds_df, tokenizer)
    # Pack label columns into a single 'labels' column so the Trainer always receives a labels tensor
    def _pack_labels(batch):
        # batch is a dict of lists; build labels as list of lists
        n = len(batch[next(iter(batch))])
        out = {'labels': []}
        for i in range(n):
            out['labels'].append([batch[l][i] for l in label_cols])
        return out

    ds = ds.map(_pack_labels, batched=True)
    # Keep only input_ids, attention_mask and labels
    keep_cols = ['input_ids', 'attention_mask', 'labels']
    existing_keep = [c for c in keep_cols if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in existing_keep])
    ds.set_format(type='torch', columns=existing_keep)

    # compute class weights (per label)
    label_counts = ds_df[label_cols].sum().values
    total = len(ds_df)
    pos_weights = total / (len(label_cols) * (label_counts + 1e-6))
    pos_weights = torch.tensor(pos_weights, dtype=torch.float)

    # Build TrainingArguments in a backward-compatible way (filter kwargs supported by installed transformers)
    import inspect
    training_kwargs = dict(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        save_total_limit=2,
    )
    sig = inspect.signature(TrainingArguments)
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in sig.parameters}
    training_args = TrainingArguments(**filtered_kwargs)

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
        labels_arr = labels_arr
        from sklearn.metrics import f1_score, precision_score, recall_score
        return {
            'micro_f1': f1_score(labels_arr, preds, average='micro', zero_division=0),
            'macro_f1': f1_score(labels_arr, preds, average='macro', zero_division=0),
            'precision': precision_score(labels_arr, preds, average='macro', zero_division=0),
            'recall': recall_score(labels_arr, preds, average='macro', zero_division=0),
        }

    class WeightedTrainer(Trainer):
        def __init__(self, *args, label_cols=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.label_cols = label_cols or []

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Accept extra Trainer kwargs (e.g., num_items_in_batch) for compatibility with HF Trainer versions
            # Ensure label tensors are removed from `inputs` so the model does not compute loss internally
            labels = None
            if 'labels' in inputs:
                labels = inputs.pop('labels')
            elif self.label_cols:
                # build labels tensor from separate columns and remove them from inputs
                label_tensors = [inputs.pop(l) for l in self.label_cols if l in inputs]
                if label_tensors:
                    labels = torch.stack(label_tensors, dim=1)

            # Call model without labels to get logits, then compute BCEWithLogitsLoss manually
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.get('logits')
            if labels is None:
                raise ValueError("No labels found in inputs for computing loss")
            # ensure labels on same device as logits and in float dtype
            labels = labels.to(logits.device).float()
            loss_fct = BCEWithLogitsLoss(pos_weight=pos_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds.select(range(int(len(ds)*0.9))),
        eval_dataset=ds.select(range(int(len(ds)*0.9), len(ds))),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        label_cols=label_cols,
    )
    return trainer


if __name__ == '__main__':
    df = load_processed_dataset()
    trainer = build_trainer(df, ['harmful','critical','supportive','neutral'])
    print('Trainer ready. Call trainer.train() when you want to start fine-tuning.')