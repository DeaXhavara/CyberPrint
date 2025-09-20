#!/usr/bin/env python3
"""
DeBERTa Model Evaluation Script

This script evaluates the trained DeBERTa model on test data and provides detailed metrics.
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def load_model_and_tokenizer(model_dir):
    """Load the trained model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Load label encoders
    sentiment_encoder = joblib.load(os.path.join(model_dir, "sentiment_encoder.pkl"))
    
    # Load metadata
    with open(os.path.join(model_dir, "training_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    return model, tokenizer, sentiment_encoder, metadata

def load_test_data():
    """Load test data from various sources"""
    # Try to load human-corrected examples as test set
    corrected_path = os.path.join(
        os.path.dirname(__file__), '..', 
        'cyberprint', 'data', 'active_learning', 'misclassified_examples.csv'
    )
    
    test_data = []
    
    if os.path.exists(corrected_path):
        df = pd.read_csv(corrected_path)
        # Use human-corrected examples as test set
        corrected = df[
            (df['human_reviewed'] == 'yes') & 
            (df['corrected_sentiment'].notna())
        ].copy()
        
        if len(corrected) > 0:
            test_df = pd.DataFrame({
                'text': corrected['text'],
                'true_sentiment': corrected['corrected_sentiment'],
                'source': 'human_corrected'
            })
            test_data.append(('Human Corrected', test_df))
    
    # Also test on original predictions to see improvement
    if os.path.exists(corrected_path):
        df = pd.read_csv(corrected_path)
        original_df = pd.DataFrame({
            'text': df['text'].head(50),  # Sample of original predictions
            'true_sentiment': df['rule_suggested_sentiment'].head(50),
            'source': 'rule_suggested'
        })
        test_data.append(('Rule Suggested', original_df))
    
    return test_data

def predict_batch(model, tokenizer, texts, max_length=256):
    """Make predictions on a batch of texts"""
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_classes = torch.argmax(predictions, dim=-1)
        confidences = torch.max(predictions, dim=-1)[0]
    
    return predicted_classes.numpy(), confidences.numpy(), predictions.numpy()

def evaluate_model(model, tokenizer, sentiment_encoder, test_data, output_dir):
    """Evaluate model on test data"""
    results = {}
    
    for test_name, test_df in test_data:
        print(f"\nEvaluating on {test_name} ({len(test_df)} examples)")
        
        if len(test_df) == 0:
            continue
        
        # Map true labels to encoded format
        label_mapping = {
            'supportive': 'positive',
            'harmful': 'negative', 
            'critical': 'yellow_flag',
            'neutral': 'neutral',
            'positive': 'positive',
            'negative': 'negative',
            'yellow_flag': 'yellow_flag'
        }
        
        test_df['mapped_sentiment'] = test_df['true_sentiment'].map(label_mapping).fillna(test_df['true_sentiment'])
        
        # Filter out unmappable labels
        valid_labels = set(sentiment_encoder.classes_)
        test_df = test_df[test_df['mapped_sentiment'].isin(valid_labels)]
        
        if len(test_df) == 0:
            print(f"No valid examples for {test_name}")
            continue
        
        # Encode true labels
        true_labels = sentiment_encoder.transform(test_df['mapped_sentiment'])
        
        # Make predictions
        pred_classes, confidences, probabilities = predict_batch(
            model, tokenizer, test_df['text'].tolist()
        )
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_classes)
        f1_macro = f1_score(true_labels, pred_classes, average='macro')
        f1_weighted = f1_score(true_labels, pred_classes, average='weighted')
        precision = precision_score(true_labels, pred_classes, average='weighted')
        recall = recall_score(true_labels, pred_classes, average='weighted')
        
        # Store results
        results[test_name] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'num_examples': len(test_df),
            'avg_confidence': np.mean(confidences)
        }
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 (macro): {f1_macro:.3f}")
        print(f"F1 (weighted): {f1_weighted:.3f}")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        
        # Detailed classification report
        class_names = sentiment_encoder.classes_
        report = classification_report(
            true_labels, pred_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        print(f"\nDetailed Classification Report for {test_name}:")
        print(classification_report(true_labels, pred_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_classes)
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {test_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{test_name.lower().replace(" ", "_")}.png'))
        plt.close()
        
        # Save detailed results
        detailed_results = pd.DataFrame({
            'text': test_df['text'],
            'true_sentiment': test_df['mapped_sentiment'],
            'predicted_sentiment': [class_names[i] for i in pred_classes],
            'confidence': confidences,
            'correct': true_labels == pred_classes
        })
        
        detailed_results.to_csv(
            os.path.join(output_dir, f'detailed_results_{test_name.lower().replace(" ", "_")}.csv'),
            index=False
        )
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DeBERTa model")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained model')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    args = parser.parse_args()
    
    print("DeBERTa Model Evaluation")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("📥 Loading model and tokenizer...")
    model, tokenizer, sentiment_encoder, metadata = load_model_and_tokenizer(args.model_dir)
    
    print(f"Model trained on: {metadata.get('training_date', 'Unknown')}")
    print(f"Dataset size: {metadata.get('dataset_size', 'Unknown')}")
    print(f"Gold standard examples: {metadata.get('gold_standard_count', 'Unknown')}")
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data()
    
    if not test_data:
        print("❌ No test data found!")
        return
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, sentiment_encoder, test_data, args.output_dir)
    
    # Save overall results
    evaluation_summary = {
        'evaluation_date': datetime.now().isoformat(),
        'model_dir': args.model_dir,
        'results': results,
        'model_metadata': metadata
    }
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    evaluation_summary = convert_numpy_types(evaluation_summary)
    
    with open(os.path.join(args.output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    for test_name, metrics in results.items():
        print(f"\n{test_name}:")
        print(f"  • Accuracy: {metrics['accuracy']:.3f}")
        print(f"  • F1 (weighted): {metrics['f1_weighted']:.3f}")
        print(f"  • Examples: {metrics['num_examples']}")
        print(f"  • Avg Confidence: {metrics['avg_confidence']:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
