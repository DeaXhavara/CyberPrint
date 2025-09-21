#!/usr/bin/env python3
"""
Enhanced CyberPrint ML Predictor with DeBERTa Model
==================================================

ML predictor using trained DeBERTa model for sentiment analysis,
enhanced with rule-based sub-label classification.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CyberPrintMLPredictor:
    """ML predictor using trained DeBERTa model with fallback to Logistic Regression."""
    
    def __init__(self, model_dir: str = None, enable_gpt_oss: bool = False, enable_active_learning: bool = True, use_ensemble: bool = True):
        # Try ensemble first for highest confidence
        if use_ensemble:
            try:
                from cyberprint.models.ml.ensemble_predictor import create_cyberprint_ensemble
                self.predictor = create_cyberprint_ensemble()
                self.predictor_type = "ensemble"
                logger.info("Using ensemble predictor for highest confidence")
            except Exception as e:
                logger.warning(f"Failed to load ensemble: {e}")
                self._init_single_model(model_dir)
        else:
            self._init_single_model(model_dir)
    
    def _init_single_model(self, model_dir: str = None):
        """Initialize single model (DeBERTa or logistic regression)"""
        # Try DeBERTa model first - use the 4-epoch active learning model
        deberta_model_dir = os.path.join(os.path.dirname(__file__), "models", "deberta_active_learning_4epochs")
        
        if os.path.exists(deberta_model_dir):
            # Use DeBERTa predictor
            try:
                from cyberprint.models.ml.deberta_predictor import DeBERTaPredictor
                self.predictor = DeBERTaPredictor(deberta_model_dir)
                self.predictor_type = "deberta"
                logger.info("Using DeBERTa model for predictions")
            except Exception as e:
                logger.warning(f"Failed to load DeBERTa model: {e}")
                self._init_logistic_regression(model_dir)
        else:
            self._init_logistic_regression(model_dir)
        
        # Initialize additional components
        self.labels = ['positive', 'negative', 'neutral', 'yellow_flag']
        self.sub_label_classifier = None
        self.misclassification_detector = None
        self._initialize_enhancers(enable_active_learning=True)
    
    def _init_logistic_regression(self, model_dir: str = None):
        """Initialize logistic regression fallback model."""
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "cyberprint", "models", "ml")
        
        self.model_path = os.path.join(model_dir, "cyberprint_ml_model.joblib")
        self.vectorizer_path = os.path.join(model_dir, "cyberprint_vectorizer.joblib")
        
        # Try alternative paths if main ones don't exist
        if not os.path.exists(self.model_path):
            alt_model_path = os.path.join(model_dir, "polished_model.joblib")
            if os.path.exists(alt_model_path):
                self.model_path = alt_model_path
        
        if not os.path.exists(self.vectorizer_path):
            alt_vectorizer_path = os.path.join(model_dir, "polished_vectorizer.joblib")
            if os.path.exists(alt_vectorizer_path):
                self.vectorizer_path = alt_vectorizer_path
        
        self.model = None
        self.vectorizer = None
        self.predictor_type = "logistic_regression"
        self.load_model()
        logger.info("Using Logistic Regression model for predictions")
    
    def _init_additional_components(self, enable_gpt_oss: bool, enable_active_learning: bool):
        self._initialize_enhancers(enable_gpt_oss, enable_active_learning)
    
    def load_model(self):
        """Load the trained model and vectorizer."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded ML model from {self.model_path}")
            else:
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if os.path.exists(self.vectorizer_path):
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info(f"Loaded vectorizer from {self.vectorizer_path}")
            else:
                logger.error(f"Vectorizer file not found: {self.vectorizer_path}")
                raise FileNotFoundError(f"Vectorizer file not found: {self.vectorizer_path}")
                
        except Exception as e:
            logger.error(f"Error loading model or vectorizer: {e}")
            raise
    
    def _initialize_enhancers(self, enable_active_learning: bool = True):
        """Initialize sub-label classifier and active learning components."""
        try:
            # Initialize sub-label classifier
            from cyberprint.models.sublabel_classifier import EnhancedSubLabelClassifier
            self.sub_label_classifier = EnhancedSubLabelClassifier()
            logger.info("Initialized enhanced sub-label classifier")
            
            # Initialize active learning misclassification detector
            if enable_active_learning:
                try:
                    from cyberprint.active_learning.misclassification_detector import MisclassificationDetector
                    self.misclassification_detector = MisclassificationDetector()
                    logger.info("Initialized active learning misclassification detector")
                except ImportError as e:
                    logger.warning(f"Active learning not available: {e}")
            
        except ImportError as e:
            logger.warning(f"Enhanced features not available: {e}")
            logger.warning("Falling back to basic prediction only")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text exactly like during training.
        Remove newlines, strip spaces, handle emojis/markdown.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove newlines and extra whitespace
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic*
        text = re.sub(r'__(.*?)__', r'\1', text)      # __bold__
        text = re.sub(r'_(.*?)_', r'\1', text)        # _italic_
        text = re.sub(r'`(.*?)`', r'\1', text)        # `code`
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove /u/username
        text = re.sub(r'/r/\w+', '', text)  # Remove /r/subreddit
        text = re.sub(r'&gt;', '>', text)   # HTML entities
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&amp;', '&', text)
        
        # Handle emojis - keep them as they might be meaningful for sentiment
        # Just normalize excessive emoji repetition
        text = re.sub(r'([😀-🿿])\1{3,}', r'\1\1\1', text)  # Limit emoji repetition to 3
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{4,}', '!!!', text)
        text = re.sub(r'[?]{4,}', '???', text)
        text = re.sub(r'[.]{4,}', '...', text)
        
        # Strip and normalize spaces
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def predict_text(self, texts: Union[str, List[str]], 
                     include_sub_labels: bool = True,
                     confidence_threshold: float = 0.6) -> List[Dict[str, any]]:
        """
        Enhanced prediction with sub-label classification.
        
        Args:
            texts: Text(s) to predict
            include_sub_labels: Whether to include sub-label classification
            confidence_threshold: Threshold for confidence scoring
            
        Returns:
            List of dicts with:
            - "text": original text
            - "predicted_label": main sentiment (positive/negative/neutral/yellow_flag)
            - "predicted_score": confidence score (0-1)
            - "sub_label": sub-label classification (if enabled)
            - "sub_label_confidence": sub-label confidence score
            - "applied_rules": list of applied sub-label rules
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Use ensemble for highest confidence, then DeBERTa, then logistic regression
            if hasattr(self, 'predictor') and self.predictor_type == "ensemble":
                return self._predict_with_ensemble(texts, include_sub_labels, confidence_threshold)
            elif hasattr(self, 'predictor') and self.predictor_type == "deberta":
                return self._predict_with_deberta(texts, include_sub_labels, confidence_threshold)
            else:
                return self._predict_with_logistic_regression(texts, include_sub_labels, confidence_threshold)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return fallback results
            return [{"probs": {label: 0.0 for label in self.labels}, 
                    "predicted_label": "neutral", 
                    "predicted_score": 0.0,
                    "sub_label": "general",
                    "sub_label_confidence": 0.0,
                    "applied_rules": [],
                    "enhanced": False,
                    "enhancement_metadata": {}} for _ in texts]
    
    def _predict_with_ensemble(self, texts: List[str], include_sub_labels: bool, confidence_threshold: float) -> List[Dict[str, any]]:
        """Predict using ensemble of models for highest confidence."""
        try:
            # Use ensemble predictor directly
            results = self.predictor.predict_batch(texts)
            
            # Apply sub-label classification if enabled
            if include_sub_labels and self.sub_label_classifier:
                for i, (text, result) in enumerate(zip(texts, results)):
                    try:
                        sub_label, sub_confidence = self.sub_label_classifier.classify(text)
                        applied_rules = self.sub_label_classifier.get_applied_rules()
                        result.update({
                            "sub_label": sub_label,
                            "sub_label_confidence": sub_confidence,
                            "applied_rules": applied_rules
                        })
                    except Exception as e:
                        logger.warning(f"Sub-label classification failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            # Fallback to DeBERTa
            return self._predict_with_deberta(texts, include_sub_labels, confidence_threshold)
    
    def _predict_with_deberta(self, texts: List[str], include_sub_labels: bool, confidence_threshold: float) -> List[Dict[str, any]]:
        """Predict using DeBERTa model."""
        try:
            # Use DeBERTa predictor
            deberta_results = self.predictor.predict_batch(texts)
            
            results = []
            for i, text in enumerate(texts):
                deberta_result = deberta_results[i]
                
                # Initialize result dictionary
                result = {
                    "probs": deberta_result.get("probs", {}),
                    "predicted_label": deberta_result.get("predicted_label", "neutral"),
                    "predicted_score": deberta_result.get("predicted_score", 0.0),
                    "sub_label": deberta_result.get("sub_label", "general"),
                    "sub_label_confidence": deberta_result.get("sub_label_confidence", 0.0),
                    "applied_rules": deberta_result.get("applied_rules", []),
                    "enhanced": False,
                    "enhancement_metadata": {}
                }
                
                # Apply post-processing for mixed sentiment with gratitude
                result = self._apply_gratitude_override(text, result)
                
                # Override sub-labels if we have our own classifier and it's requested
                # But preserve DeBERTa gratitude overrides
                if include_sub_labels and self.sub_label_classifier:
                    try:
                        # Check if DeBERTa already applied a gratitude override
                        deberta_gratitude_override = deberta_result.get("gratitude_override", False)
                        
                        if deberta_gratitude_override and deberta_result.get("sub_label") == "gratitude":
                            # Preserve DeBERTa's gratitude classification
                            logger.info(f"Preserving DeBERTa gratitude override for: {text[:50]}...")
                        else:
                            # Use our sub-label classifier
                            sub_label, sub_confidence, applied_rules = self.sub_label_classifier.classify_sub_label(
                                text, result["predicted_label"], log_rules=True
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
            
        except Exception as e:
            logger.error(f"DeBERTa prediction failed: {e}")
            # Fallback to logistic regression
            return self._predict_with_logistic_regression(texts, include_sub_labels, confidence_threshold)
    
    def _predict_with_logistic_regression(self, texts: List[str], include_sub_labels: bool, confidence_threshold: float) -> List[Dict[str, any]]:
        """Predict using logistic regression model."""
        if not self.model or not self.vectorizer:
            logger.error("Model or vectorizer not loaded")
            # Return fallback results
            return [{"probs": {label: 0.0 for label in self.labels}, 
                    "predicted_label": "neutral", 
                    "predicted_score": 0.0,
                    "sub_label": "general",
                    "sub_label_confidence": 0.0,
                    "applied_rules": [],
                    "enhanced": False,
                    "enhancement_metadata": {}} for _ in texts]
        
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Vectorize texts
            X = self.vectorizer.transform(processed_texts)
            
            # Get predictions and probabilities
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            results = []
            for i, text in enumerate(texts):
                # Get probabilities for all classes
                probs_dict = {}
                for j, label in enumerate(self.labels):
                    if j < probabilities.shape[1]:
                        probs_dict[label] = float(probabilities[i][j])
                    else:
                        probs_dict[label] = 0.0
                
                # Find predicted label and score with confidence boost
                predicted_label = max(probs_dict, key=probs_dict.get)
                base_score = probs_dict[predicted_label]
                
                # Use authentic model confidence
                predicted_score = base_score
                
                # Initialize result dictionary
                result = {
                    "probs": probs_dict,
                    "predicted_label": predicted_label,
                    "predicted_score": predicted_score,
                    "sub_label": "general",
                    "sub_label_confidence": 0.0,
                    "applied_rules": [],
                    "enhanced": False,
                    "enhancement_metadata": {}
                }
                
                # Apply gratitude override for mixed sentiment
                result = self._apply_gratitude_override(text, result)
                
                # Add sub-label classification if requested
                if include_sub_labels and self.sub_label_classifier:
                    try:
                        sub_label, sub_confidence, applied_rules = self.sub_label_classifier.classify_sub_label(
                            text, result["predicted_label"], log_rules=True  # Use potentially overridden sentiment
                        )
                        # Don't override if gratitude override was already applied
                        if not result.get("enhanced", False) or result.get("sub_label") != "gratitude":
                            result.update({
                                "sub_label": sub_label,
                                "sub_label_confidence": sub_confidence,
                                "applied_rules": applied_rules
                            })
                    except Exception as e:
                        logger.warning(f"Sub-label classification failed: {e}")
                
                result.update({"enhanced": False, "enhancement_metadata": {"reason": "gpt_oss_removed"}})
                results.append(result)
            
            # Run active learning detection on the batch
            if self.misclassification_detector:
                try:
                    misclass_stats = self.misclassification_detector.process_predictions(results, texts)
                    logger.debug(f"Active learning stats: {misclass_stats}")
                except Exception as e:
                    logger.warning(f"Active learning detection failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return fallback results with all fields
            fallback_result = {
                "probs": {label: 0.0 for label in self.labels}, 
                "predicted_label": "neutral", 
                "predicted_score": 0.0,
                "sub_label": "general",
                "sub_label_confidence": 0.0,
                "applied_rules": [],
                "enhanced": False,
                "enhancement_metadata": {}
            }
            return [fallback_result for _ in texts]
    
    def _apply_gratitude_override(self, text: str, result: Dict) -> Dict:
        """Apply gratitude override for mixed sentiment comments."""
        text_lower = text.lower()
        
        # Gratitude indicators
        gratitude_patterns = [
            r'\bthanks?\s+for\b',
            r'\bthank\s+you\b',
            r'\bgrateful\b',
            r'\bappreciate\b',
            r'\bthanks?\s+(everyone|all|guys|folks)\b',
            r'\bthanks?\s+for\s+(being|staying|coming)\b'
        ]
        
        # Positive intent indicators
        positive_intent_patterns = [
            r'\bhope\s+(you|this|that|people|someone)\b',
            r'\bwish\s+(you|everyone)\b',
            r'\bmade\s+(you|people|someone)\s+(smile|laugh|happy|giggle)\b',
            r'\bsmile[d]?\b.*\b(:|;|:D|:P|:\))\b',
            r'\bgiggle[d]?\b',
            r'\bsmile[d]?\b'
        ]
        
        # Check for gratitude + positive intent combination
        has_gratitude = any(re.search(pattern, text_lower) for pattern in gratitude_patterns)
        has_positive_intent = any(re.search(pattern, text_lower) for pattern in positive_intent_patterns)
        
        # Override negative prediction if both gratitude and positive intent are present
        if has_gratitude and has_positive_intent and result["predicted_label"] in ["negative", "neutral"]:
            logger.info(f"Applying gratitude override for mixed sentiment: {text[:50]}...")
            
            # Update to positive sentiment
            result.update({
                "predicted_label": "positive",
                "predicted_score": 0.75,  # Moderate confidence
                "sub_label": "gratitude",
                "sub_label_confidence": 0.8,
                "applied_rules": ["gratitude_override: mixed_sentiment_with_thanks_and_hope"],
                "enhanced": True,
                "enhancement_metadata": {
                    "override_type": "gratitude_mixed_sentiment",
                    "original_sentiment": result["predicted_label"],
                    "gratitude_detected": has_gratitude,
                    "positive_intent_detected": has_positive_intent
                }
            })
            
            # Update probabilities
            result["probs"] = {
                "positive": 0.75,
                "negative": 0.10,
                "neutral": 0.10,
                "yellow_flag": 0.05
            }
        
        return result

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CyberPrintMLPredictor()
    return _predictor

def predict_text(texts: Union[str, List[str]], 
                 include_sub_labels: bool = True,
                 confidence_threshold: float = 0.6) -> List[Dict[str, any]]:
    """
    Enhanced convenience function to predict text using the global predictor.
    Compatible with the existing predict.py interface but with enhanced features.
    
    Args:
        texts: Text(s) to predict
        include_sub_labels: Whether to include sub-label classification
        confidence_threshold: Threshold for confidence scoring
        
    Returns:
        List of enhanced prediction dictionaries
    """
    predictor = get_predictor()
    return predictor.predict_text(texts, include_sub_labels, confidence_threshold)
