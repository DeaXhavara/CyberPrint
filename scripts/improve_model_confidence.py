#!/usr/bin/env python3
"""
Script to legitimately improve DeBERTa model confidence through proper ML techniques
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def collect_misclassified_examples():
    """Identify examples where the model has low confidence for retraining"""
    from cyberprint_ml_predictor import CyberPrintMLPredictor
    
    predictor = CyberPrintMLPredictor()
    
    # Test examples that might be challenging
    test_cases = [
        "This is amazing! Love it so much :)",
        "I hate this stupid thing",
        "It's okay I guess",
        "DM me for more info",
        "Thanks for sharing this",
        "You're an idiot",
        "Not sure about this one",
        "Best product ever!!!",
        "Worst experience of my life",
        "Could be better"
    ]
    
    low_confidence_examples = []
    
    for text in test_cases:
        result = predictor.predict([text])[0]
        confidence = result['predicted_score']
        
        if confidence < 0.85:  # Flag low confidence predictions
            low_confidence_examples.append({
                'text': text,
                'predicted_label': result['predicted_label'],
                'confidence': confidence,
                'needs_review': True
            })
    
    return low_confidence_examples

def create_confidence_calibration():
    """Implement temperature scaling for better confidence calibration"""
    
    class TemperatureScaling:
        def __init__(self):
            self.temperature = 1.0
        
        def fit(self, logits, labels):
            """Find optimal temperature using validation set"""
            from scipy.optimize import minimize_scalar
            
            def temperature_loss(temp):
                scaled_logits = logits / temp
                probabilities = torch.softmax(scaled_logits, dim=1)
                # Negative log likelihood
                return -torch.sum(torch.log(probabilities[range(len(labels)), labels]))
            
            result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
            self.temperature = result.x
            return self.temperature
        
        def predict(self, logits):
            """Apply temperature scaling to logits"""
            return torch.softmax(logits / self.temperature, dim=1)
    
    return TemperatureScaling()

def create_ensemble_predictor():
    """Create ensemble of multiple models for higher confidence"""
    
    class EnsemblePredictor:
        def __init__(self):
            self.models = []
            self.weights = []
        
        def add_model(self, model, weight=1.0):
            self.models.append(model)
            self.weights.append(weight)
        
        def predict(self, texts):
            """Combine predictions from multiple models"""
            all_predictions = []
            
            for model, weight in zip(self.models, self.weights):
                preds = model.predict(texts)
                all_predictions.append([(pred, weight) for pred in preds])
            
            # Weighted average of predictions
            ensemble_results = []
            for i in range(len(texts)):
                combined_probs = {}
                total_weight = 0
                
                for model_preds in all_predictions:
                    pred, weight = model_preds[i]
                    total_weight += weight
                    
                    for label, prob in pred['probs'].items():
                        if label not in combined_probs:
                            combined_probs[label] = 0
                        combined_probs[label] += prob * weight
                
                # Normalize
                for label in combined_probs:
                    combined_probs[label] /= total_weight
                
                # Find best prediction
                best_label = max(combined_probs, key=combined_probs.get)
                best_score = combined_probs[best_label]
                
                ensemble_results.append({
                    'probs': combined_probs,
                    'predicted_label': best_label,
                    'predicted_score': best_score,
                    'ensemble_confidence': best_score  # Higher due to ensemble
                })
            
            return ensemble_results
    
    return EnsemblePredictor()

def improve_preprocessing():
    """Enhanced text preprocessing for better model performance"""
    
    import re
    from textblob import TextBlob
    
    def enhanced_preprocess(text):
        """Improved preprocessing pipeline"""
        
        # 1. Handle special cases
        text = str(text).strip()
        
        # 2. Normalize repeated characters (e.g., "sooooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 3. Handle emojis and emoticons better
        # Convert common emoticons to words
        emoticon_map = {
            ':)': ' positive_emotion ',
            ':D': ' very_positive_emotion ',
            ':(': ' negative_emotion ',
            ':P': ' playful_emotion ',
            ':/': ' uncertain_emotion ',
            '<3': ' love_emotion '
        }
        
        for emoticon, replacement in emoticon_map.items():
            text = text.replace(emoticon, replacement)
        
        # 4. Handle negations better
        text = re.sub(r'\b(not|no|never|nothing|nowhere|nobody|none)\b', r'NOT_\1', text)
        
        # 5. Handle intensifiers
        intensifiers = ['very', 'really', 'extremely', 'super', 'totally', 'absolutely']
        for intensifier in intensifiers:
            text = re.sub(f'\\b{intensifier}\\b', f'INTENSIFIER_{intensifier}', text, flags=re.IGNORECASE)
        
        return text
    
    return enhanced_preprocess

def create_training_plan():
    """Create a comprehensive plan to improve model confidence"""
    
    plan = {
        "immediate_actions": [
            "1. Collect more domain-specific training data (Reddit/social media)",
            "2. Implement active learning to identify challenging examples",
            "3. Use temperature scaling for confidence calibration",
            "4. Improve text preprocessing pipeline"
        ],
        
        "medium_term": [
            "1. Train ensemble of multiple models",
            "2. Implement cross-validation for robust evaluation",
            "3. Add uncertainty quantification",
            "4. Fine-tune on domain-specific data"
        ],
        
        "advanced_techniques": [
            "1. Bayesian neural networks for uncertainty",
            "2. Monte Carlo dropout for confidence estimation",
            "3. Adversarial training for robustness",
            "4. Knowledge distillation from larger models"
        ],
        
        "data_collection": [
            "1. Scrape more Reddit comments with clear sentiment",
            "2. Use human annotation for edge cases",
            "3. Balance dataset across all sentiment classes",
            "4. Add domain-specific examples (social media, reviews)"
        ]
    }
    
    return plan

def main():
    """Main function to demonstrate confidence improvement techniques"""
    
    print("🚀 CyberPrint Model Confidence Improvement Plan")
    print("=" * 60)
    
    # 1. Analyze current model performance
    print("\n1. Analyzing current model performance...")
    low_conf_examples = collect_misclassified_examples()
    
    if low_conf_examples:
        print(f"Found {len(low_conf_examples)} low-confidence examples:")
        for example in low_conf_examples:
            print(f"  - '{example['text']}' -> {example['predicted_label']} ({example['confidence']:.2f})")
    
    # 2. Create improvement plan
    print("\n2. Creating improvement plan...")
    plan = create_training_plan()
    
    print("\n📋 IMMEDIATE ACTIONS:")
    for action in plan["immediate_actions"]:
        print(f"  {action}")
    
    print("\n📋 MEDIUM-TERM GOALS:")
    for action in plan["medium_term"]:
        print(f"  {action}")
    
    # 3. Demonstrate techniques
    print("\n3. Available techniques:")
    print("  ✅ Temperature Scaling - for confidence calibration")
    print("  ✅ Ensemble Methods - combine multiple models")
    print("  ✅ Enhanced Preprocessing - better text handling")
    print("  ✅ Active Learning - focus on challenging examples")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run this script to identify low-confidence examples")
    print("2. Collect more training data for those cases")
    print("3. Retrain model with enhanced data")
    print("4. Apply temperature scaling for calibration")
    print("5. Consider ensemble methods for optimization")
    
    return plan

if __name__ == "__main__":
    main()
