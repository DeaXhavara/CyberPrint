#!/usr/bin/env python3
"""
Active Learning Retraining Pipeline
===================================

Integrates corrected examples from active learning into model retraining
to continuously improve CyberPrint performance.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

class ActiveLearningRetrainingPipeline:
    """Pipeline for retraining models with active learning corrections."""
    
    def __init__(self, model_dir: str = None, data_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "ml")
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.active_learning_dir = os.path.join(data_dir, "active_learning")
        
        # Model paths
        self.model_path = os.path.join(model_dir, "cyberprint_ml_model.joblib")
        self.vectorizer_path = os.path.join(model_dir, "cyberprint_vectorizer.joblib")
        
        # Backup paths for previous versions
        self.backup_dir = os.path.join(model_dir, "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Training data paths
        self.main_dataset_path = os.path.join(data_dir, "processed", "cyberprint_dataset.csv")
        self.misclassified_path = os.path.join(self.active_learning_dir, "misclassified_examples.csv")
        
        self.model = None
        self.vectorizer = None
    
    def load_existing_model(self) -> bool:
        """Load existing model and vectorizer."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            
            if os.path.exists(self.vectorizer_path):
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info(f"Loaded existing vectorizer from {self.vectorizer_path}")
            
            return self.model is not None and self.vectorizer is not None
            
        except Exception as e:
            logger.error(f"Failed to load existing model: {e}")
            return False
    
    def backup_existing_model(self):
        """Create backup of existing model before retraining."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if os.path.exists(self.model_path):
                backup_model_path = os.path.join(self.backup_dir, f"model_backup_{timestamp}.joblib")
                joblib.dump(joblib.load(self.model_path), backup_model_path)
                logger.info(f"Backed up model to {backup_model_path}")
            
            if os.path.exists(self.vectorizer_path):
                backup_vectorizer_path = os.path.join(self.backup_dir, f"vectorizer_backup_{timestamp}.joblib")
                joblib.dump(joblib.load(self.vectorizer_path), backup_vectorizer_path)
                logger.info(f"Backed up vectorizer to {backup_vectorizer_path}")
                
        except Exception as e:
            logger.error(f"Failed to backup existing model: {e}")
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load original training data and corrected examples."""
        
        # Load main dataset
        main_df = pd.DataFrame()
        if os.path.exists(self.main_dataset_path):
            try:
                main_df = pd.read_csv(self.main_dataset_path)
                logger.info(f"Loaded main dataset: {len(main_df)} examples")
            except Exception as e:
                logger.error(f"Failed to load main dataset: {e}")
        
        # Load corrected examples from active learning
        corrected_df = pd.DataFrame()
        if os.path.exists(self.misclassified_path):
            try:
                misclass_df = pd.read_csv(self.misclassified_path)
                
                # Filter for human-reviewed examples with corrections
                corrected_df = misclass_df[
                    (misclass_df['human_reviewed'] == 'yes') & 
                    (misclass_df['corrected_sentiment'].notna()) & 
                    (misclass_df['corrected_sentiment'] != '')
                ].copy()
                
                if len(corrected_df) > 0:
                    # Rename columns to match training format
                    corrected_df = corrected_df[['text', 'corrected_sentiment', 'corrected_sub_label']].copy()
                    corrected_df.columns = ['text', 'sentiment', 'sub_label']
                    logger.info(f"Loaded corrected examples: {len(corrected_df)} examples")
                else:
                    logger.info("No corrected examples found for retraining")
                    
            except Exception as e:
                logger.error(f"Failed to load corrected examples: {e}")
        
        return main_df, corrected_df
    
    def prepare_training_data(self, main_df: pd.DataFrame, corrected_df: pd.DataFrame, 
                            augment_corrections: bool = True) -> Tuple[List[str], List[str]]:
        """Prepare combined training data with optional augmentation of corrections."""
        
        # Combine datasets
        if len(corrected_df) > 0:
            # Augment corrected examples to give them more weight
            if augment_corrections:
                # Duplicate corrected examples 3x to increase their influence
                augmented_corrected = pd.concat([corrected_df] * 3, ignore_index=True)
                logger.info(f"Augmented corrected examples: {len(corrected_df)} -> {len(augmented_corrected)}")
                combined_df = pd.concat([main_df, augmented_corrected], ignore_index=True)
            else:
                combined_df = pd.concat([main_df, corrected_df], ignore_index=True)
        else:
            combined_df = main_df.copy()
        
        # Prepare texts and labels
        texts = combined_df['text'].astype(str).tolist()
        labels = combined_df['sentiment'].astype(str).tolist()
        
        # Remove any invalid entries
        valid_indices = [i for i, (text, label) in enumerate(zip(texts, labels)) 
                        if text.strip() and label.strip() and label in ['positive', 'negative', 'neutral', 'yellow_flag']]
        
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        logger.info(f"Prepared training data: {len(texts)} examples")
        logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text exactly like the original training."""
        import re
        
        if not isinstance(text, str):
            text = str(text)
        
        # Remove newlines and extra whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)
        text = re.sub(r'/r/\w+', '', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&amp;', '&', text)
        
        # Handle emojis - normalize excessive repetition
        text = re.sub(r'([😀-🿿])\1{3,}', r'\1\1\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{4,}', '!!!', text)
        text = re.sub(r'[?]{4,}', '???', text)
        text = re.sub(r'[.]{4,}', '...', text)
        
        # Strip and normalize spaces
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def retrain_model(self, texts: List[str], labels: List[str], 
                     test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Retrain the model with combined data."""
        
        logger.info("Starting model retraining...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        logger.info(f"Training set: {len(X_train)} examples")
        logger.info(f"Test set: {len(X_test)} examples")
        
        # Create or update vectorizer
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.95
            )
            X_train_vec = self.vectorizer.fit_transform(X_train)
        else:
            # Use existing vectorizer but potentially update vocabulary
            X_train_vec = self.vectorizer.transform(X_train)
        
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train_vec))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test_vec))
        
        y_pred = self.model.predict(X_test_vec)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_rep,
            'training_examples': len(X_train),
            'test_examples': len(X_test)
        }
        
        logger.info(f"Retraining completed - Train Accuracy: {train_accuracy:.3f}, Test Accuracy: {test_accuracy:.3f}")
        
        return results
    
    def save_retrained_model(self):
        """Save the retrained model and vectorizer."""
        try:
            # Ensure model directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Saved retrained model to {self.model_path}")
            
            # Save vectorizer
            joblib.dump(self.vectorizer, self.vectorizer_path)
            logger.info(f"Saved vectorizer to {self.vectorizer_path}")
            
        except Exception as e:
            logger.error(f"Failed to save retrained model: {e}")
            raise
    
    def run_retraining_pipeline(self, augment_corrections: bool = True, 
                              backup_existing: bool = True) -> Dict[str, Any]:
        """Run the complete retraining pipeline."""
        
        logger.info("Starting active learning retraining pipeline...")
        
        try:
            # Load existing model
            self.load_existing_model()
            
            # Backup existing model
            if backup_existing:
                self.backup_existing_model()
            
            # Load training data
            main_df, corrected_df = self.load_training_data()
            
            if len(main_df) == 0:
                raise ValueError("No main training data found")
            
            if len(corrected_df) == 0:
                logger.warning("No corrected examples found - retraining with original data only")
            
            # Prepare combined training data
            texts, labels = self.prepare_training_data(main_df, corrected_df, augment_corrections)
            
            if len(texts) == 0:
                raise ValueError("No valid training data prepared")
            
            # Retrain model
            results = self.retrain_model(texts, labels)
            
            # Save retrained model
            self.save_retrained_model()
            
            # Add pipeline metadata to results
            results.update({
                'pipeline_timestamp': datetime.now().isoformat(),
                'original_examples': len(main_df),
                'corrected_examples': len(corrected_df),
                'total_training_examples': len(texts),
                'augment_corrections': augment_corrections
            })
            
            logger.info("Active learning retraining pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            raise
    
    def evaluate_improvement(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
        """Evaluate improvement on a held-out test set."""
        
        if not self.model or not self.vectorizer:
            raise ValueError("Model not loaded")
        
        # Preprocess test texts
        processed_texts = [self.preprocess_text(text) for text in test_texts]
        
        # Vectorize
        X_test = self.vectorizer.transform(processed_texts)
        
        # Predict
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, y_pred)
        classification_rep = classification_report(test_labels, y_pred, output_dict=True)
        
        # Calculate average confidence
        max_probas = np.max(y_proba, axis=1)
        avg_confidence = np.mean(max_probas)
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'classification_report': classification_rep,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
