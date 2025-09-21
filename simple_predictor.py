#!/usr/bin/env python3
"""
Simple Rule-Based Sentiment Predictor for CyberPrint
===================================================

Fallback predictor that uses keyword-based sentiment analysis
when ML models fail to load properly.
"""

import re
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

class SimpleRuleBasedPredictor:
    """Simple rule-based sentiment predictor using keyword matching."""
    
    def __init__(self):
        self.positive_keywords = [
            'love', 'like', 'great', 'awesome', 'amazing', 'excellent', 'fantastic',
            'wonderful', 'good', 'nice', 'cool', 'thank', 'thanks', 'grateful',
            'appreciate', 'happy', 'joy', 'excited', 'perfect', 'brilliant',
            'outstanding', 'superb', 'incredible', 'beautiful', 'lovely'
        ]
        
        self.negative_keywords = [
            'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst', 'sucks',
            'disgusting', 'stupid', 'idiot', 'pathetic', 'useless', 'trash',
            'garbage', 'disappointing', 'annoying', 'frustrating', 'angry'
        ]
        
        self.neutral_keywords = [
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can',
            'could', 'would', 'should', 'might', 'maybe', 'perhaps'
        ]
        
        # Initialize sub-label classifier
        try:
            from cyberprint.models.sublabel_classifier import EnhancedSubLabelClassifier
            self.sub_label_classifier = EnhancedSubLabelClassifier()
        except ImportError:
            self.sub_label_classifier = None
            logger.warning("Sub-label classifier not available")
    
    def predict(self, texts: List[str], include_sub_labels: bool = True) -> List[Dict]:
        """Predict sentiment for a list of texts."""
        results = []
        
        for text in texts:
            sentiment, confidence = self._classify_text(text)
            
            result = {
                "predicted_label": sentiment,
                "predicted_score": confidence,
                "probs": self._get_probabilities(sentiment, confidence),
                "sub_label": "general",
                "sub_label_confidence": 0.0,
                "applied_rules": [],
                "enhanced": False,
                "enhancement_metadata": {}
            }
            
            # Add sub-label classification
            if include_sub_labels and self.sub_label_classifier:
                try:
                    sub_label, sub_confidence, applied_rules = self.sub_label_classifier.classify_sub_label(
                        text, sentiment, log_rules=True
                    )
                    result.update({
                        "sub_label": sub_label,
                        "sub_label_confidence": sub_confidence,
                        "applied_rules": applied_rules
                    })
                except Exception as e:
                    logger.warning(f"Sub-label classification failed: {e}")
            
            results.append(result)
        
        return results
    
    def _classify_text(self, text: str) -> tuple:
        """Classify a single text into sentiment and confidence."""
        text_lower = text.lower()
        
        # Count keyword matches
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # Check for questions (neutral indicators)
        has_question = '?' in text or any(keyword in text_lower for keyword in self.neutral_keywords)
        
        # Determine sentiment
        if positive_count > negative_count and positive_count > 0:
            sentiment = "positive"
            confidence = min(0.6 + (positive_count * 0.1), 0.9)
        elif negative_count > positive_count and negative_count > 0:
            sentiment = "negative"
            confidence = min(0.6 + (negative_count * 0.1), 0.9)
        elif has_question:
            sentiment = "neutral"
            confidence = 0.7
        else:
            sentiment = "neutral"
            confidence = 0.6
        
        # Check for sarcasm indicators (yellow flag)
        sarcasm_patterns = [
            r'\boh\s+(sure|right|great)\b',
            r'\byeah\s+right\b',
            r'\breal\s+(smart|clever)\b'
        ]
        
        if any(re.search(pattern, text_lower) for pattern in sarcasm_patterns):
            sentiment = "yellow_flag"
            confidence = 0.7
        
        return sentiment, confidence
    
    def _get_probabilities(self, sentiment: str, confidence: float) -> Dict[str, float]:
        """Generate probability distribution based on predicted sentiment."""
        probs = {"positive": 0.25, "negative": 0.25, "neutral": 0.25, "yellow_flag": 0.25}
        
        # Adjust probabilities based on prediction
        remaining = 1.0 - confidence
        other_prob = remaining / 3
        
        probs[sentiment] = confidence
        for label in probs:
            if label != sentiment:
                probs[label] = other_prob
        
        return probs

def predict_text(texts: Union[str, List[str]], 
                 include_sub_labels: bool = True,
                 confidence_threshold: float = 0.6) -> List[Dict]:
    """
    Simple prediction function that always works.
    
    Args:
        texts: Text(s) to predict
        include_sub_labels: Whether to include sub-label classification
        confidence_threshold: Threshold for confidence scoring (unused in simple predictor)
        
    Returns:
        List of prediction dictionaries
    """
    if isinstance(texts, str):
        texts = [texts]
    
    predictor = SimpleRuleBasedPredictor()
    return predictor.predict(texts, include_sub_labels)
