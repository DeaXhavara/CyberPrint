#!/usr/bin/env python3
"""
Updated Prediction Script for CyberPrint
=======================================

This script provides prediction functionality using the updated models
trained on the new balanced dataset structure.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CyberPrintPredictor:
    """Predictor for CyberPrint sentiment analysis."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__))
        
        # Model paths
        self.sentiment_model_path = os.path.join(self.model_dir, "cyberprint_ml_model.joblib")
        self.sentiment_vectorizer_path = os.path.join(self.model_dir, "cyberprint_vectorizer.joblib")
        self.mental_health_model_path = os.path.join(self.model_dir, "mental_health_model.joblib")
        self.mental_health_vectorizer_path = os.path.join(self.model_dir, "mental_health_vectorizer.joblib")
        self.yellow_flag_model_path = os.path.join(self.model_dir, "yellow_flag_model.joblib")
        self.yellow_flag_vectorizer_path = os.path.join(self.model_dir, "yellow_flag_vectorizer.joblib")
        
        # Load models
        self.models = {}
        self.vectorizers = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models and vectorizers."""
        logger.info("Loading CyberPrint models...")
        
        # Load sentiment model
        if os.path.exists(self.sentiment_model_path) and os.path.exists(self.sentiment_vectorizer_path):
            self.models['sentiment'] = joblib.load(self.sentiment_model_path)
            self.vectorizers['sentiment'] = joblib.load(self.sentiment_vectorizer_path)
            logger.info("Sentiment model loaded successfully")
        else:
            logger.warning("Sentiment model not found. Please train the model first.")
        
        # Load mental health model
        if os.path.exists(self.mental_health_model_path) and os.path.exists(self.mental_health_vectorizer_path):
            self.models['mental_health'] = joblib.load(self.mental_health_model_path)
            self.vectorizers['mental_health'] = joblib.load(self.mental_health_vectorizer_path)
            logger.info("Mental health model loaded successfully")
        else:
            logger.warning("Mental health model not found. Please train the model first.")
        
        # Load yellow flag model
        if os.path.exists(self.yellow_flag_model_path) and os.path.exists(self.yellow_flag_vectorizer_path):
            self.models['yellow_flag'] = joblib.load(self.yellow_flag_model_path)
            self.vectorizers['yellow_flag'] = joblib.load(self.yellow_flag_vectorizer_path)
            logger.info("Yellow flag model loaded successfully")
        else:
            logger.warning("Yellow flag model not found. Please train the model first.")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for prediction."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove URLs, mentions, hashtags
        import re
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict_sentiment(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Predict sentiment for given texts."""
        if 'sentiment' not in self.models:
            raise ValueError("Sentiment model not loaded. Please train the model first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizers['sentiment'].transform(processed_texts)
        
        # Predict
        predictions = self.models['sentiment'].predict(X)
        probabilities = self.models['sentiment'].predict_proba(X)
        
        # Get class names
        class_names = self.models['sentiment'].classes_
        
        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[i],
                'sentiment': pred,
                'confidence': float(max(probs)),
                'probabilities': dict(zip(class_names, probs))
            }
            results.append(result)
        
        return {
            'predictions': results,
            'model_type': 'sentiment'
        }
    
    def predict_mental_health(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Predict mental health warnings for given texts."""
        if 'mental_health' not in self.models:
            raise ValueError("Mental health model not loaded. Please train the model first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizers['mental_health'].transform(processed_texts)
        
        # Predict
        predictions = self.models['mental_health'].predict(X)
        probabilities = self.models['mental_health'].predict_proba(X)
        
        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[i],
                'mental_health_warning': bool(pred),
                'confidence': float(max(probs)),
                'probabilities': {
                    'no_warning': float(probs[0]),
                    'warning': float(probs[1])
                }
            }
            results.append(result)
        
        return {
            'predictions': results,
            'model_type': 'mental_health'
        }
    
    def predict_yellow_flag(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Predict yellow flags (sarcasm/irony) for given texts."""
        if 'yellow_flag' not in self.models:
            raise ValueError("Yellow flag model not loaded. Please train the model first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizers['yellow_flag'].transform(processed_texts)
        
        # Predict
        predictions = self.models['yellow_flag'].predict(X)
        probabilities = self.models['yellow_flag'].predict_proba(X)
        
        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[i],
                'yellow_flag': bool(pred),
                'confidence': float(max(probs)),
                'probabilities': {
                    'normal': float(probs[0]),
                    'sarcasm_irony': float(probs[1])
                }
            }
            results.append(result)
        
        return {
            'predictions': results,
            'model_type': 'yellow_flag'
        }
    
    def predict_all(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """Predict all aspects (sentiment, mental health, yellow flags) for given texts."""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'sentiment': None,
                'mental_health_warning': None,
                'yellow_flag': None,
                'confidence_scores': {}
            }
            
            # Predict sentiment
            try:
                sentiment_result = self.predict_sentiment([text])
                result['sentiment'] = sentiment_result['predictions'][0]['sentiment']
                result['confidence_scores']['sentiment'] = sentiment_result['predictions'][0]['confidence']
            except Exception as e:
                logger.warning(f"Sentiment prediction failed for text {i}: {e}")
            
            # Predict mental health
            try:
                mental_health_result = self.predict_mental_health([text])
                result['mental_health_warning'] = mental_health_result['predictions'][0]['mental_health_warning']
                result['confidence_scores']['mental_health'] = mental_health_result['predictions'][0]['confidence']
            except Exception as e:
                logger.warning(f"Mental health prediction failed for text {i}: {e}")
            
            # Predict yellow flag
            try:
                yellow_flag_result = self.predict_yellow_flag([text])
                result['yellow_flag'] = yellow_flag_result['predictions'][0]['yellow_flag']
                result['confidence_scores']['yellow_flag'] = yellow_flag_result['predictions'][0]['confidence']
            except Exception as e:
                logger.warning(f"Yellow flag prediction failed for text {i}: {e}")
            
            results.append(result)
        
        return {
            'predictions': results,
            'model_type': 'all'
        }

def main():
    """Main function to demonstrate prediction functionality."""
    predictor = CyberPrintPredictor()
    
    # Example texts
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "Sure, that's exactly what I needed...",
        "I'm feeling really sad and hopeless today.",
        "The weather is nice today."
    ]
    
    print("CyberPrint Prediction Demo")
    print("=" * 50)
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        try:
            result = predictor.predict_all([text])
            prediction = result['predictions'][0]
            
            print(f"  Sentiment: {prediction['sentiment']}")
            print(f"  Mental Health Warning: {prediction['mental_health_warning']}")
            print(f"  Yellow Flag (Sarcasm/Irony): {prediction['yellow_flag']}")
            print(f"  Confidence Scores: {prediction['confidence_scores']}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
