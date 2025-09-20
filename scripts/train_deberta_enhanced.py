#!/usr/bin/env python3
"""
Enhanced DeBERTa Training Pipeline with Human-Corrected Gold Standard Integration

This script:
1. Loads the main training dataset
2. Integrates human-corrected examples as gold standard data
3. Trains DeBERTa on both sentiment classification and sub-label prediction
4. Supports multi-task learning for better performance
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
from datetime import datetime

def load_human_corrected_examples():
    """Load and process human-corrected examples from active learning data"""
    # Load challenging examples
    challenging_path = os.path.join(
        os.path.dirname(__file__), '..', 
        'data', 'active_learning', 'challenging_examples.csv'
    )
    
    # Load edge cases
    edge_cases_path = os.path.join(
        os.path.dirname(__file__), '..', 
        'data', 'active_learning', 'edge_cases.csv'
    )
    
    gold_standard_dfs = []
    
    # Process challenging examples
    if os.path.exists(challenging_path):
        df_challenging = pd.read_csv(challenging_path)
        # Filter human-reviewed examples (all rows have human_reviewed=True and valid labels)
        corrected_challenging = df_challenging[
            (df_challenging['human_reviewed'] == True) & 
            (df_challenging['human_sentiment'].notna()) & 
            (df_challenging['human_sub_label'].notna())
        ].copy()
        
        if len(corrected_challenging) > 0:
            gold_challenging = pd.DataFrame({
                'text': corrected_challenging['text'],
                'sentiment': corrected_challenging['human_sentiment'],
                'sub_label': corrected_challenging['human_sub_label'],
                'source': 'human_corrected',
                'confidence': 1.0  # High confidence for human corrections
            })
            gold_standard_dfs.append(gold_challenging)
            print(f"Loaded {len(gold_challenging)} challenging examples")
    
    # Process edge cases
    if os.path.exists(edge_cases_path):
        df_edge = pd.read_csv(edge_cases_path)
        # Filter human-reviewed examples
        corrected_edge = df_edge[
            (df_edge['human_reviewed'] == True) & 
            (df_edge['human_sentiment'].notna()) & 
            (df_edge['human_sub_label'].notna())
        ].copy()
        
        if len(corrected_edge) > 0:
            gold_edge = pd.DataFrame({
                'text': corrected_edge['text'],
                'sentiment': corrected_edge['human_sentiment'],
                'sub_label': corrected_edge['human_sub_label'],
                'source': 'human_corrected',
                'confidence': 1.0  # High confidence for human corrections
            })
            gold_standard_dfs.append(gold_edge)
            print(f"Loaded {len(gold_edge)} edge case examples")
    
    # Combine all gold standard data
    if gold_standard_dfs:
        gold_standard = pd.concat(gold_standard_dfs, ignore_index=True)
        print(f"Total human-corrected gold standard examples: {len(gold_standard)}")
        return gold_standard
    else:
        print("No human-corrected examples found. Training without gold standard.")
        return pd.DataFrame()

def load_main_dataset():
    """Load the main training dataset"""
    # Try multiple possible data sources
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'cyberprint', 'data', 'processed', 'merged_dataset.parquet'),
        os.path.join(os.path.dirname(__file__), '..', 'cyberprint', 'data', 'processed', 'final_database_clean.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'cyberprint_profile_data.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading main dataset from: {path}")
            if path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            
            # Standardize column names
            if 'tweet' in df.columns:
                df = df.rename(columns={'tweet': 'text'})
            
            # Ensure required columns exist
            if 'text' not in df.columns:
                continue
                
            # Add missing columns if needed
            if 'sentiment' not in df.columns and 'label' in df.columns:
                df = df.rename(columns={'label': 'sentiment'})
            
            if 'sub_label' not in df.columns:
                df['sub_label'] = 'general'  # Default sub-label
                
            if 'source' not in df.columns:
                df['source'] = 'original'
                
            if 'confidence' not in df.columns:
                df['confidence'] = 0.8  # Default confidence
            
            print(f"Loaded {len(df)} examples from main dataset")
            return df[['text', 'sentiment', 'sub_label', 'source', 'confidence']]
    
    raise FileNotFoundError("No suitable training dataset found!")

def prepare_enhanced_dataset():
    """Combine main dataset with human-corrected gold standard examples"""
    main_df = load_main_dataset()
    gold_df = load_human_corrected_examples()
    
    if len(gold_df) > 0:
        # Combine datasets, giving priority to gold standard
        combined_df = pd.concat([main_df, gold_df], ignore_index=True)
        
        # Remove duplicates, keeping gold standard versions
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='last')
        
        gold_count = len(combined_df[combined_df['source'] == 'human_corrected'])
        print(f"Final dataset: {len(combined_df)} examples ({gold_count} gold standard)")
    else:
        combined_df = main_df
        print(f"Final dataset: {len(combined_df)} examples (no gold standard)")
    
    return combined_df

def setup_label_encoders(df):
    """Setup label encoders for sentiment and sub-labels"""
    # Map original labels to new 4-class system
    label_mapping = {
        'supportive': 'positive',
        'harmful': 'negative', 
        'critical': 'yellow_flag',
        'neutral': 'neutral',
        # Also handle cases where labels are already in new format
        'positive': 'positive',
        'negative': 'negative',
        'yellow_flag': 'yellow_flag'
    }
    
    # Apply mapping to sentiment column
    df['sentiment'] = df['sentiment'].map(label_mapping).fillna(df['sentiment'])
    
    # Sentiment encoder
    sentiment_labels = ['positive', 'negative', 'neutral', 'yellow_flag']
    sentiment_encoder = LabelEncoder()
    sentiment_encoder.fit(sentiment_labels)
    
    # Sub-label encoder
    unique_sublabels = df['sub_label'].unique().tolist()
    if 'general' not in unique_sublabels:
        unique_sublabels.append('general')
    
    sublabel_encoder = LabelEncoder()
    sublabel_encoder.fit(unique_sublabels)
    
    return sentiment_encoder, sublabel_encoder, sentiment_labels, unique_sublabels

def create_simple_model(model_name, num_sentiment_labels):
    """Create a simple single-task model for sentiment prediction"""
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_sentiment_labels
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Enhanced DeBERTa with Human Corrections")
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base', help='Model name')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--sublabel_weight', type=float, default=0.3, help='Sub-label loss weight')
    args = parser.parse_args()
    
    print("Starting Enhanced DeBERTa Training with Human Corrections")
    print("=" * 60)
    
    # 1. Prepare enhanced dataset
    df = prepare_enhanced_dataset()
    
    # 2. Setup label encoders
    sentiment_encoder, sublabel_encoder, sentiment_labels, sublabel_labels = setup_label_encoders(df)
    
    # 3. Encode labels
    df['sentiment_label'] = sentiment_encoder.transform(df['sentiment'])
    df['sublabel_label'] = sublabel_encoder.transform(df['sub_label'])
    
    # 4. Train/validation split (stratified by sentiment)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, 
        stratify=df['sentiment_label']
    )
    
    print(f"Training set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    
    # 5. Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = create_simple_model(args.model_name, len(sentiment_labels))
    
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length,
        )
        tokenized['labels'] = examples['sentiment_label']
        return tokenized
    
    # 6. Create datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'sentiment_label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'sentiment_label']])
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.out_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="none",
        warmup_steps=100,
        weight_decay=0.01,
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        sentiment_preds = logits.argmax(axis=-1)
        
        # Calculate metrics
        acc = accuracy_score(labels, sentiment_preds)
        f1 = f1_score(labels, sentiment_preds, average='weighted')
        
        return {
            "accuracy": acc,
            "f1": f1,
            "sentiment_accuracy": acc
        }
    
    # 8. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # 9. Train!
    print("Starting training...")
    trainer.train()
    
    # 10. Save everything
    print("Saving model and metadata...")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    
    # Save label mappings
    metadata = {
        'sentiment_labels': sentiment_labels,
        'sublabel_labels': sublabel_labels,
        'training_args': vars(args),
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(df),
        'gold_standard_count': len(df[df['source'] == 'human_corrected']),
    }
    
    with open(os.path.join(args.out_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save label encoders
    import joblib
    joblib.dump(sentiment_encoder, os.path.join(args.out_dir, "sentiment_encoder.pkl"))
    joblib.dump(sublabel_encoder, os.path.join(args.out_dir, "sublabel_encoder.pkl"))
    
    print("Training completed successfully!")
    print(f"Model saved to: {args.out_dir}")
    print(f"Gold standard examples used: {len(df[df['source'] == 'human_corrected'])}")

if __name__ == "__main__":
    main()
