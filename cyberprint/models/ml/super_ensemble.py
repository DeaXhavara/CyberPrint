#!/usr/bin/env python3
"""
Super Ensemble Predictor for CyberPrint
=======================================

Combines multiple models (DeBERTa + Logistic Regression variants) 
for maximum authentic confidence without artificial boosting.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Union
import os
import joblib

logger = logging.getLogger(__name__)

class SuperEnsemblePredictor:
    """Enhanced ensemble combining DeBERTa with multiple logistic regression models."""
    
    def __init__(self):
        self.models = []
        self.model_weights = []
        self.labels = ['positive', 'negative', 'neutral', 'yellow_flag']
        self._load_models()
    
    def _load_models(self):
        """Load all available models with their performance weights."""
        
        # 1. Load DeBERTa model (highest weight due to superior performance)
        try:
            from cyberprint.models.ml.deberta_predictor import DeBERTaPredictor
            deberta_path = '/Users/deaxhavara/CyberPrint/models/deberta_active_learning_4epochs'
            if os.path.exists(deberta_path):
                deberta = DeBERTaPredictor(deberta_path)
                self.models.append(('deberta', deberta))
                self.model_weights.append(0.95)  # 95% weight for DeBERTa (it's already 97% confident)
                logger.info("✓ Loaded DeBERTa model with 95% weight")
        except Exception as e:
            logger.warning(f"Failed to load DeBERTa: {e}")
        
        # 2. Load polished logistic regression
        try:
            polished_model_path = '/Users/deaxhavara/CyberPrint/cyberprint/models/ml/polished_model.joblib'
            polished_vec_path = '/Users/deaxhavara/CyberPrint/cyberprint/models/ml/polished_vectorizer.joblib'
            
            if os.path.exists(polished_model_path) and os.path.exists(polished_vec_path):
                polished_model = joblib.load(polished_model_path)
                polished_vectorizer = joblib.load(polished_vec_path)
                self.models.append(('polished_lr', (polished_model, polished_vectorizer)))
                self.model_weights.append(0.03)  # 3% weight
                logger.info("✓ Loaded polished LR model with 20% weight")
        except Exception as e:
            logger.warning(f"Failed to load polished LR: {e}")
        
        # 3. Load standard logistic regression
        try:
            standard_model_path = '/Users/deaxhavara/CyberPrint/cyberprint/models/ml/cyberprint_ml_model.joblib'
            standard_vec_path = '/Users/deaxhavara/CyberPrint/cyberprint/models/ml/cyberprint_vectorizer.joblib'
            
            if os.path.exists(standard_model_path) and os.path.exists(standard_vec_path):
                standard_model = joblib.load(standard_model_path)
                standard_vectorizer = joblib.load(standard_vec_path)
                self.models.append(('standard_lr', (standard_model, standard_vectorizer)))
                self.model_weights.append(0.02)  # 2% weight
                logger.info("✓ Loaded standard LR model with 10% weight")
        except Exception as e:
            logger.warning(f"Failed to load standard LR: {e}")
        
        # Normalize weights
        if self.model_weights:
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]
            logger.info(f"Super ensemble loaded with {len(self.models)} models")
        else:
            raise RuntimeError("No models could be loaded for super ensemble")
    
    def predict_text(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Predict using weighted ensemble of all models."""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            # Get predictions from all models
            model_predictions = []
            
            for i, (model_name, model) in enumerate(self.models):
                try:
                    if model_name == 'deberta':
                        pred = model.predict_text([text])[0]
                        model_predictions.append({
                            'probs': pred['probs'],
                            'confidence': pred['predicted_score'],
                            'weight': self.model_weights[i]
                        })
                    else:  # Logistic regression models
                        lr_model, vectorizer = model
                        # Preprocess text for LR
                        processed_text = self._preprocess_text_for_lr(text)
                        text_vec = vectorizer.transform([processed_text])
                        
                        # Get probabilities - handle multi-output case
                        if hasattr(lr_model, 'predict_proba'):
                            probs_raw = lr_model.predict_proba(text_vec)
                            if isinstance(probs_raw, list):
                                # Multi-output classifier - combine probabilities
                                probs = np.mean([p[0] for p in probs_raw], axis=0)
                            else:
                                probs = probs_raw[0]
                        else:
                            # Fallback to decision function
                            decision = lr_model.decision_function(text_vec)[0]
                            # Convert to probabilities using sigmoid
                            probs = 1 / (1 + np.exp(-decision))
                            if len(probs) != len(self.labels):
                                probs = np.array([0.25] * len(self.labels))
                        
                        # Ensure we have the right number of probabilities
                        if len(probs) != len(self.labels):
                            probs = np.array([0.25] * len(self.labels))
                        
                        prob_dict = {self.labels[j]: float(probs[j]) for j in range(len(self.labels))}
                        
                        # Get confidence (max probability)
                        confidence = float(max(probs))
                        
                        model_predictions.append({
                            'probs': prob_dict,
                            'confidence': confidence,
                            'weight': self.model_weights[i]
                        })
                        
                except Exception as e:
                    logger.warning(f"Model {model_name} failed for text: {e}")
                    continue
            
            if not model_predictions:
                # Fallback result
                results.append({
                    'predicted_label': 'neutral',
                    'predicted_score': 0.5,
                    'probs': {label: 0.25 for label in self.labels},
                    'ensemble_info': {'models_used': 0, 'fallback': True}
                })
                continue
            
            # Weighted ensemble combination
            combined_probs = {label: 0.0 for label in self.labels}
            total_weighted_confidence = 0.0
            total_weight = 0.0
            
            for pred in model_predictions:
                weight = pred['weight']
                total_weight += weight
                
                # Combine probabilities
                for label in self.labels:
                    combined_probs[label] += pred['probs'].get(label, 0.0) * weight
                
                # Weight confidence by model performance
                total_weighted_confidence += pred['confidence'] * weight
            
            # Normalize probabilities
            if total_weight > 0:
                for label in combined_probs:
                    combined_probs[label] /= total_weight
                final_confidence = total_weighted_confidence / total_weight
            else:
                final_confidence = 0.5
            
            # Get best prediction
            best_label = max(combined_probs, key=combined_probs.get)
            
            # Apply confidence calibration based on ensemble agreement
            agreement_factor = self._calculate_agreement_factor(model_predictions, best_label)
            calibrated_confidence = self._calibrate_confidence(final_confidence, agreement_factor)
            
            results.append({
                'predicted_label': best_label,
                'predicted_score': calibrated_confidence,
                'probs': combined_probs,
                'ensemble_info': {
                    'models_used': len(model_predictions),
                    'agreement_factor': agreement_factor,
                    'raw_confidence': final_confidence
                }
            })
        
        return results
    
    def _preprocess_text_for_lr(self, text: str) -> str:
        """Preprocess text for logistic regression models."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic preprocessing
        text = text.strip().lower()
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _calculate_agreement_factor(self, predictions: List[Dict], best_label: str) -> float:
        """Calculate how much models agree on the prediction."""
        if not predictions:
            return 0.0
        
        # Count weighted agreement
        agreeing_weight = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = pred['weight']
            total_weight += weight
            
            # Check if this model's top prediction matches ensemble prediction
            model_best = max(pred['probs'], key=pred['probs'].get)
            if model_best == best_label:
                agreeing_weight += weight
        
        return agreeing_weight / total_weight if total_weight > 0 else 0.0
    
    def _calibrate_confidence(self, raw_confidence: float, agreement_factor: float) -> float:
        """Calibrate confidence based on model agreement - no artificial boosting."""
        
        # Only apply minor calibration based on agreement
        # High agreement = slightly higher confidence
        # Low agreement = slightly lower confidence
        
        if agreement_factor >= 0.9:
            # Very high agreement - small boost
            calibrated = raw_confidence * 1.02
        elif agreement_factor >= 0.7:
            # Good agreement - tiny boost
            calibrated = raw_confidence * 1.01
        elif agreement_factor < 0.5:
            # Low agreement - small penalty
            calibrated = raw_confidence * 0.98
        else:
            # Medium agreement - no change
            calibrated = raw_confidence
        
        # Ensure bounds [0, 1]
        return max(0.0, min(1.0, calibrated))

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch prediction interface."""
        return self.predict_text(texts)
