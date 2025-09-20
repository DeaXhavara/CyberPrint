#!/usr/bin/env python3
"""
Active Learning Misclassification Detection System
=================================================

Detects systematic misclassifications using rule-based validation
and captures examples for manual review and model improvement.
"""

import os
import csv
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class MisclassificationDetector:
    """Detects and logs misclassified examples for active learning."""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "active_learning")
        
        self.output_dir = output_dir
        self.misclassified_file = os.path.join(output_dir, "misclassified_examples.csv")
        self.validation_rules = self._initialize_validation_rules()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CSV file if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize the misclassified examples CSV file."""
        if not os.path.exists(self.misclassified_file):
            headers = [
                'timestamp', 'text', 'predicted_sentiment', 'predicted_sub_label', 
                'predicted_confidence', 'rule_suggested_sentiment', 'rule_suggested_sub_label',
                'misclassification_type', 'rule_triggered', 'corrected_sentiment', 
                'corrected_sub_label', 'human_reviewed', 'notes'
            ]
            
            with open(self.misclassified_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            logger.info(f"Initialized misclassified examples file: {self.misclassified_file}")
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rule-based validation for detecting systematic misclassifications."""
        
        return {
            # Gratitude should be positive
            'gratitude_positive': {
                'keywords': ['thank', 'thanks', 'grateful', 'appreciate', 'blessing'],
                'patterns': [r'\bthank\s+you\b', r'\bthanks\s+for\b', r'\bappreciate\s+it\b'],
                'expected_sentiment': 'positive',
                'expected_sub_label': 'gratitude',
                'confidence_threshold': 0.3,
                'description': 'Expressions of gratitude should be classified as positive'
            },
            
            # Humor/jokes should be yellow_flag
            'humor_yellow_flag': {
                'keywords': ['lol', 'haha', 'funny', 'hilarious', 'joke', 'comedy'],
                'patterns': [r'\blol\b', r'\bha+h+a+\b', r'\bso\s+funny\b'],
                'expected_sentiment': 'yellow_flag',
                'expected_sub_label': 'humor',
                'confidence_threshold': 0.4,
                'description': 'Humor and jokes should be classified as yellow_flag'
            },
            
            # Questions should be neutral
            'questions_neutral': {
                'keywords': ['what', 'how', 'why', 'when', 'where', 'who'],
                'patterns': [r'\?', r'\bwhat\s+(is|are|do)\b', r'\bhow\s+do\b'],
                'expected_sentiment': 'neutral',
                'expected_sub_label': 'question_based',
                'confidence_threshold': 0.3,
                'description': 'Questions should typically be classified as neutral'
            },
            
            # Compliments should be positive
            'compliments_positive': {
                'keywords': ['amazing', 'awesome', 'excellent', 'brilliant', 'fantastic', 'great job'],
                'patterns': [r'\byou\s+are\s+(amazing|awesome|great)\b', r'\bgreat\s+job\b'],
                'expected_sentiment': 'positive',
                'expected_sub_label': 'compliments',
                'confidence_threshold': 0.4,
                'description': 'Compliments should be classified as positive'
            },
            
            # Threats should be negative
            'threats_negative': {
                'keywords': ['kill', 'destroy', 'hurt', 'violence', 'threat'],
                'patterns': [r'\bgoing\s+to\s+(kill|hurt|destroy)\b', r'\bmake\s+you\s+pay\b'],
                'expected_sentiment': 'negative',
                'expected_sub_label': 'threatening',
                'confidence_threshold': 0.5,
                'description': 'Threats should be classified as negative'
            },
            
            # Insults should be negative
            'insults_negative': {
                'keywords': ['stupid', 'idiot', 'moron', 'pathetic', 'worthless'],
                'patterns': [r'\byou\s+are\s+(stupid|an\s+idiot|pathetic)\b'],
                'expected_sentiment': 'negative',
                'expected_sub_label': 'insulting',
                'confidence_threshold': 0.4,
                'description': 'Insults should be classified as negative'
            },
            
            # Factual statements should be neutral
            'facts_neutral': {
                'keywords': ['according to', 'research shows', 'data indicates', 'studies show'],
                'patterns': [r'\baccording\s+to\b', r'\bresearch\s+shows\b', r'\bdata\s+(shows|indicates)\b'],
                'expected_sentiment': 'neutral',
                'expected_sub_label': 'fact_based',
                'confidence_threshold': 0.3,
                'description': 'Factual statements should be classified as neutral'
            },
            
            # Sarcasm should be yellow_flag
            'sarcasm_yellow_flag': {
                'keywords': ['oh sure', 'yeah right', 'obviously', 'totally'],
                'patterns': [r'\boh\s+sure\b', r'\byeah\s+right\b', r'\bobviously\b.*\bnot\b'],
                'expected_sentiment': 'yellow_flag',
                'expected_sub_label': 'sarcasm',
                'confidence_threshold': 0.4,
                'description': 'Sarcastic comments should be classified as yellow_flag'
            }
        }
    
    def validate_prediction(self, text: str, predicted_sentiment: str, predicted_sub_label: str, 
                          confidence: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate a prediction against rule-based expectations.
        
        Args:
            text: The comment text
            predicted_sentiment: Model's predicted sentiment
            predicted_sub_label: Model's predicted sub-label
            confidence: Model's confidence score
            
        Returns:
            Tuple of (is_misclassified, misclassification_info)
        """
        text_lower = text.lower()
        
        for rule_name, rule in self.validation_rules.items():
            # Check if rule applies to this text
            rule_matches = False
            
            # Check keywords
            for keyword in rule['keywords']:
                if keyword.lower() in text_lower:
                    rule_matches = True
                    break
            
            # Check patterns if no keyword match
            if not rule_matches:
                import re
                for pattern in rule['patterns']:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        rule_matches = True
                        break
            
            if rule_matches:
                expected_sentiment = rule['expected_sentiment']
                expected_sub_label = rule['expected_sub_label']
                
                # Check if prediction differs from expectation
                sentiment_mismatch = predicted_sentiment != expected_sentiment
                sub_label_mismatch = predicted_sub_label != expected_sub_label
                low_confidence = confidence < rule['confidence_threshold']
                
                if sentiment_mismatch or sub_label_mismatch or low_confidence:
                    misclassification_info = {
                        'rule_name': rule_name,
                        'rule_description': rule['description'],
                        'expected_sentiment': expected_sentiment,
                        'expected_sub_label': expected_sub_label,
                        'sentiment_mismatch': sentiment_mismatch,
                        'sub_label_mismatch': sub_label_mismatch,
                        'low_confidence': low_confidence,
                        'confidence_threshold': rule['confidence_threshold']
                    }
                    return True, misclassification_info
        
        return False, None
    
    def log_misclassification(self, text: str, predicted_sentiment: str, predicted_sub_label: str,
                            predicted_confidence: float, misclassification_info: Dict[str, Any],
                            corrected_sentiment: str = None, corrected_sub_label: str = None,
                            notes: str = None):
        """Log a misclassified example to the CSV file."""
        
        timestamp = datetime.now().isoformat()
        
        # Determine misclassification type
        misclassification_types = []
        if misclassification_info.get('sentiment_mismatch', False):
            misclassification_types.append('sentiment_mismatch')
        if misclassification_info.get('sub_label_mismatch', False):
            misclassification_types.append('sub_label_mismatch')
        if misclassification_info.get('low_confidence', False):
            misclassification_types.append('low_confidence')
        
        misclassification_type = ';'.join(misclassification_types)
        
        row = [
            timestamp,
            text,
            predicted_sentiment,
            predicted_sub_label,
            predicted_confidence,
            misclassification_info['expected_sentiment'],
            misclassification_info['expected_sub_label'],
            misclassification_type,
            misclassification_info['rule_name'],
            corrected_sentiment or '',
            corrected_sub_label or '',
            'no',  # human_reviewed
            notes or ''
        ]
        
        try:
            with open(self.misclassified_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            logger.info(f"Logged misclassification: {misclassification_info['rule_name']}")
            
        except Exception as e:
            logger.error(f"Failed to log misclassification: {e}")
    
    def process_predictions(self, predictions: List[Dict[str, Any]], texts: List[str]) -> Dict[str, Any]:
        """
        Process a batch of predictions and detect misclassifications.
        
        Args:
            predictions: List of prediction dictionaries
            texts: List of corresponding text comments
            
        Returns:
            Dictionary with misclassification statistics
        """
        total_predictions = len(predictions)
        misclassified_count = 0
        misclassification_types = {}
        
        for i, (pred, text) in enumerate(zip(predictions, texts)):
            predicted_sentiment = pred.get('predicted_label', 'neutral')
            predicted_sub_label = pred.get('sub_label', 'general')
            confidence = pred.get('predicted_score', 0.0)
            
            is_misclassified, misclass_info = self.validate_prediction(
                text, predicted_sentiment, predicted_sub_label, confidence
            )
            
            if is_misclassified:
                misclassified_count += 1
                
                # Track misclassification types
                rule_name = misclass_info['rule_name']
                if rule_name not in misclassification_types:
                    misclassification_types[rule_name] = 0
                misclassification_types[rule_name] += 1
                
                # Log the misclassification
                self.log_misclassification(
                    text, predicted_sentiment, predicted_sub_label, 
                    confidence, misclass_info
                )
        
        stats = {
            'total_predictions': total_predictions,
            'misclassified_count': misclassified_count,
            'misclassification_rate': misclassified_count / total_predictions if total_predictions > 0 else 0,
            'misclassification_types': misclassification_types
        }
        
        logger.info(f"Processed {total_predictions} predictions, found {misclassified_count} misclassifications")
        return stats
    
    def get_misclassified_examples(self, limit: int = None, rule_name: str = None) -> pd.DataFrame:
        """Load misclassified examples from CSV."""
        try:
            df = pd.read_csv(self.misclassified_file)
            
            if rule_name:
                df = df[df['rule_triggered'] == rule_name]
            
            if limit:
                df = df.head(limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load misclassified examples: {e}")
            return pd.DataFrame()
    
    def mark_as_reviewed(self, indices: List[int], corrected_sentiments: List[str] = None,
                        corrected_sub_labels: List[str] = None, notes: List[str] = None):
        """Mark examples as human-reviewed with corrections."""
        try:
            df = pd.read_csv(self.misclassified_file)
            
            for i, idx in enumerate(indices):
                if idx < len(df):
                    df.loc[idx, 'human_reviewed'] = 'yes'
                    
                    if corrected_sentiments and i < len(corrected_sentiments):
                        df.loc[idx, 'corrected_sentiment'] = corrected_sentiments[i]
                    
                    if corrected_sub_labels and i < len(corrected_sub_labels):
                        df.loc[idx, 'corrected_sub_label'] = corrected_sub_labels[i]
                    
                    if notes and i < len(notes):
                        df.loc[idx, 'notes'] = notes[i]
            
            df.to_csv(self.misclassified_file, index=False)
            logger.info(f"Marked {len(indices)} examples as reviewed")
            
        except Exception as e:
            logger.error(f"Failed to mark examples as reviewed: {e}")
    
    def get_training_data(self) -> pd.DataFrame:
        """Get corrected examples for retraining."""
        try:
            df = pd.read_csv(self.misclassified_file)
            
            # Filter for human-reviewed examples with corrections
            corrected_df = df[
                (df['human_reviewed'] == 'yes') & 
                (df['corrected_sentiment'].notna()) & 
                (df['corrected_sentiment'] != '')
            ].copy()
            
            # Rename columns to match training data format
            training_df = corrected_df[['text', 'corrected_sentiment', 'corrected_sub_label']].copy()
            training_df.columns = ['text', 'sentiment', 'sub_label']
            
            logger.info(f"Retrieved {len(training_df)} corrected examples for retraining")
            return training_df
            
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about misclassifications."""
        try:
            df = pd.read_csv(self.misclassified_file)
            
            stats = {
                'total_misclassifications': len(df),
                'human_reviewed': len(df[df['human_reviewed'] == 'yes']),
                'pending_review': len(df[df['human_reviewed'] == 'no']),
                'corrected_examples': len(df[df['corrected_sentiment'].notna()]),
                'misclassification_by_rule': df['rule_triggered'].value_counts().to_dict(),
                'misclassification_by_type': df['misclassification_type'].value_counts().to_dict()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
