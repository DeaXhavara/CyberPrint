#!/usr/bin/env python3
"""
Updated ML Model Training Script for CyberPrint
==============================================

This script trains ML models using the new balanced dataset structure.
It supports both traditional ML models (Logistic Regression) and transformer models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CyberPrintModelTrainer:
    """Trainer for CyberPrint sentiment analysis models."""
    
    def __init__(self, data_path: str = None, model_dir: str = None):
        self.data_path = data_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "processed", "final_database_clean.csv"
        )
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__))
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load dataset
        self.df = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the balanced dataset."""
        logger.info(f"Loading dataset from: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with {len(self.df)} examples")
        
        # Check required columns
        required_cols = ['tweet', 'sentiment', 'mental_health_alert', 'yellow_flag']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Display dataset info
        logger.info("Dataset info:")
        logger.info(f"  Sentiment distribution: {self.df['sentiment'].value_counts().to_dict()}")
        logger.info(f"  Mental health warnings: {self.df['mental_health_alert'].sum()}")
        logger.info(f"  Yellow flags: {self.df['yellow_flag'].sum()}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for training."""
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
    
    def train_traditional_ml(self, test_size: float = 0.2, random_state: int = 42):
        """Train traditional ML models (Logistic Regression with TF-IDF)."""
        logger.info("Training traditional ML model...")
        
        # Preprocess text
        self.df['processed_text'] = self.df['tweet'].apply(self.preprocess_text)
        
        # Remove empty texts
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        
        # Split data
        X = self.df['processed_text']
        y = self.df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Vectorize text
        logger.info("Vectorizing text with TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        logger.info("Training Logistic Regression model...")
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        
        logger.info("Model evaluation:")
        logger.info(f"Accuracy: {model.score(X_test_vec, y_test):.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        model_path = os.path.join(self.model_dir, "cyberprint_ml_model.joblib")
        vectorizer_path = os.path.join(self.model_dir, "cyberprint_vectorizer.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Vectorizer saved to: {vectorizer_path}")
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': model.score(X_test_vec, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def train_mental_health_classifier(self, test_size: float = 0.2, random_state: int = 42):
        """Train a separate classifier for mental health warnings."""
        logger.info("Training mental health warning classifier...")
        
        # Preprocess text
        self.df['processed_text'] = self.df['tweet'].apply(self.preprocess_text)
        
        # Remove empty texts
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        
        # Split data
        X = self.df['processed_text']
        y = self.df['mental_health_alert'].astype(int)  # Convert boolean to int
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Vectorize text
        logger.info("Vectorizing text for mental health classifier...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        logger.info("Training mental health classifier...")
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        
        logger.info("Mental health classifier evaluation:")
        logger.info(f"Accuracy: {model.score(X_test_vec, y_test):.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        model_path = os.path.join(self.model_dir, "mental_health_model.joblib")
        vectorizer_path = os.path.join(self.model_dir, "mental_health_vectorizer.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"Mental health model saved to: {model_path}")
        logger.info(f"Mental health vectorizer saved to: {vectorizer_path}")
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': model.score(X_test_vec, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def train_yellow_flag_classifier(self, test_size: float = 0.2, random_state: int = 42):
        """Train a separate classifier for yellow flags (sarcasm/irony)."""
        logger.info("Training yellow flag classifier...")
        
        # Preprocess text
        self.df['processed_text'] = self.df['tweet'].apply(self.preprocess_text)
        
        # Remove empty texts
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        
        # Split data
        X = self.df['processed_text']
        y = self.df['yellow_flag'].astype(int)  # Convert boolean to int
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Vectorize text
        logger.info("Vectorizing text for yellow flag classifier...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        logger.info("Training yellow flag classifier...")
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        
        logger.info("Yellow flag classifier evaluation:")
        logger.info(f"Accuracy: {model.score(X_test_vec, y_test):.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        model_path = os.path.join(self.model_dir, "yellow_flag_model.joblib")
        vectorizer_path = os.path.join(self.model_dir, "yellow_flag_vectorizer.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"Yellow flag model saved to: {model_path}")
        logger.info(f"Yellow flag vectorizer saved to: {vectorizer_path}")
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': model.score(X_test_vec, y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def train_all_models(self):
        """Train all models (sentiment, mental health, yellow flags)."""
        logger.info("Training all CyberPrint models...")
        
        results = {}
        
        # Train sentiment classifier
        results['sentiment'] = self.train_traditional_ml()
        
        # Train mental health classifier
        results['mental_health'] = self.train_mental_health_classifier()
        
        # Train yellow flag classifier
        results['yellow_flag'] = self.train_yellow_flag_classifier()
        
        logger.info("All models trained successfully!")
        return results

def main():
    """Main function to train all models."""
    trainer = CyberPrintModelTrainer()
    results = trainer.train_all_models()
    
    print("\n" + "="*50)
    print("CyberPrint Model Training Complete!")
    print("="*50)
    
    for model_type, result in results.items():
        print(f"\n{model_type.title()} Model:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
    
    print(f"\nModels saved to: {trainer.model_dir}")

if __name__ == "__main__":
    main()
