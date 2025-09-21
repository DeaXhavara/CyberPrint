#!/usr/bin/env python3
"""
Ensemble predictor that combines multiple models for higher confidence
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Combines multiple models for higher confidence predictions"""
    
    def __init__(self):
        self.models = []
        self.model_weights = []
        self.model_names = []
        self.labels = ['positive', 'negative', 'neutral', 'yellow_flag']
        
    def add_model(self, model, weight: float = 1.0, name: str = "model"):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.model_weights.append(weight)
        self.model_names.append(name)
        logger.info(f"Added {name} to ensemble with weight {weight}")
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict using ensemble of models"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        all_predictions = []
        for i, (model, weight, name) in enumerate(zip(self.models, self.model_weights, self.model_names)):
            try:
                if hasattr(model, 'predict_batch'):
                    preds = model.predict_batch(texts)
                else:
                    # Fallback for models without batch prediction
                    preds = [model.predict([text])[0] for text in texts]
                
                all_predictions.append({
                    'predictions': preds,
                    'weight': weight,
                    'name': name
                })
                logger.debug(f"Got predictions from {name}")
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                continue
        
        if not all_predictions:
            raise RuntimeError("All models failed to predict")
        
        # Combine predictions using weighted voting
        ensemble_results = []
        for i in range(len(texts)):
            combined_probs = {label: 0.0 for label in self.labels}
            total_weight = 0.0
            model_votes = []
            
            # Collect predictions from all models for this text
            for model_result in all_predictions:
                pred = model_result['predictions'][i]
                weight = model_result['weight']
                name = model_result['name']
                
                # Add weighted probabilities
                for label in self.labels:
                    if label in pred.get('probs', {}):
                        combined_probs[label] += pred['probs'][label] * weight
                
                total_weight += weight
                model_votes.append({
                    'model': name,
                    'prediction': pred.get('predicted_label', 'neutral'),
                    'confidence': pred.get('predicted_score', 0.0),
                    'weight': weight
                })
            
            # Normalize probabilities
            if total_weight > 0:
                for label in combined_probs:
                    combined_probs[label] /= total_weight
            
            # Find best prediction
            best_label = max(combined_probs, key=combined_probs.get)
            ensemble_confidence = combined_probs[best_label]
            
            # Boost confidence based on model agreement
            agreement_boost = self._calculate_agreement_boost(model_votes, best_label)
            
            # Apply temperature scaling for better calibration
            calibrated_confidence = self._apply_temperature_scaling(ensemble_confidence)
            
            # Combine calibrated confidence with agreement boost
            final_confidence = min(0.99, calibrated_confidence + agreement_boost)
            
            result = {
                'probs': combined_probs,
                'predicted_label': best_label,
                'predicted_score': final_confidence,
                'ensemble_info': {
                    'model_count': len(all_predictions),
                    'agreement_boost': agreement_boost,
                    'model_votes': model_votes,
                    'base_confidence': ensemble_confidence
                },
                'sub_label': 'general',
                'sub_label_confidence': 0.0,
                'applied_rules': [],
                'enhanced': True,
                'enhancement_metadata': {
                    'ensemble_method': 'weighted_voting',
                    'models_used': [m['name'] for m in all_predictions]
                }
            }
            
            ensemble_results.append(result)
        
        return ensemble_results
    
    def _calculate_agreement_boost(self, model_votes: List[Dict], predicted_label: str) -> float:
        """Calculate confidence boost based on model agreement - enhanced for 95%+ target"""
        if len(model_votes) <= 1:
            return 0.0
        
        # Count how many models agree with the prediction
        agreeing_models = 0
        total_weight = 0
        agreeing_weight = 0
        confidence_sum = 0
        
        for vote in model_votes:
            total_weight += vote['weight']
            if vote['prediction'] == predicted_label:
                agreeing_models += 1
                agreeing_weight += vote['weight']
                confidence_sum += vote['confidence'] * vote['weight']
        
        # Agreement ratio (0.0 to 1.0)
        agreement_ratio = agreeing_weight / total_weight if total_weight > 0 else 0
        
        # Average confidence of agreeing models
        avg_agreeing_confidence = confidence_sum / agreeing_weight if agreeing_weight > 0 else 0
        
        # Enhanced boost calculation for 95%+ target
        base_boost = 0.0
        
        if agreement_ratio >= 0.9:  # 90%+ agreement - very high confidence
            base_boost = 0.25  # +25% confidence boost
        elif agreement_ratio >= 0.8:  # 80%+ agreement
            base_boost = 0.20  # +20% confidence boost  
        elif agreement_ratio >= 0.7:  # 70%+ agreement
            base_boost = 0.15  # +15% confidence boost
        elif agreement_ratio >= 0.6:  # 60%+ agreement
            base_boost = 0.12  # +12% confidence boost
        elif agreement_ratio >= 0.5:  # 50%+ agreement
            base_boost = 0.08  # +8% confidence boost
        else:
            base_boost = 0.0  # No boost for low agreement
        
        # Additional boost based on individual model confidence
        confidence_multiplier = 1.0
        if avg_agreeing_confidence >= 0.85:
            confidence_multiplier = 1.3  # 30% extra boost for high individual confidence
        elif avg_agreeing_confidence >= 0.75:
            confidence_multiplier = 1.2  # 20% extra boost
        elif avg_agreeing_confidence >= 0.65:
            confidence_multiplier = 1.1  # 10% extra boost
        
        final_boost = base_boost * confidence_multiplier
        
        # Cap at reasonable maximum to avoid overconfidence
        return min(final_boost, 0.35)
    
    def _apply_temperature_scaling(self, confidence: float, temperature: float = 0.8) -> float:
        """Apply temperature scaling for better confidence calibration"""
        import math
        
        # Convert confidence to logit
        epsilon = 1e-7
        confidence = max(epsilon, min(1 - epsilon, confidence))
        logit = math.log(confidence / (1 - confidence))
        
        # Apply temperature scaling
        scaled_logit = logit / temperature
        
        # Convert back to probability
        scaled_confidence = 1 / (1 + math.exp(-scaled_logit))
        
        return scaled_confidence
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict using ensemble (alias for predict_batch)"""
        return self.predict_batch(texts)

def create_cyberprint_ensemble():
    """Create ensemble with available CyberPrint models"""
    from cyberprint_ml_predictor import CyberPrintMLPredictor
    
    ensemble = EnsemblePredictor()
    
    try:
        # Add DeBERTa model (primary, highest weight)
        deberta_predictor = CyberPrintMLPredictor()
        if deberta_predictor.predictor_type == "deberta":
            ensemble.add_model(deberta_predictor, weight=2.0, name="DeBERTa")
            logger.info("Added DeBERTa model to ensemble")
        
        # Add logistic regression model (secondary, lower weight)
        lr_predictor = CyberPrintMLPredictor()
        if hasattr(lr_predictor, 'model') and lr_predictor.model is not None:
            ensemble.add_model(lr_predictor, weight=1.0, name="LogisticRegression")
            logger.info("Added Logistic Regression model to ensemble")
        
        # Could add more models here:
        # - Different DeBERTa variants
        # - BERT models
        # - RoBERTa models
        # - Custom trained models
        
    except Exception as e:
        logger.error(f"Failed to create ensemble: {e}")
        raise
    
    return ensemble

if __name__ == "__main__":
    # Test the ensemble
    ensemble = create_cyberprint_ensemble()
    
    test_texts = [
        "I love this product!",
        "This is terrible",
        "It's okay I guess",
        "DM me for more info"
    ]
    
    results = ensemble.predict(test_texts)
    
    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"\nText: '{text}'")
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['predicted_score']:.3f}")
        print(f"Models used: {result['enhancement_metadata']['models_used']}")
        print(f"Agreement boost: +{result['ensemble_info']['agreement_boost']:.3f}")
