#!/usr/bin/env python3
"""
Test the ensemble predictor to measure confidence improvements
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_ensemble_confidence():
    """Test ensemble vs single model confidence"""
    
    print("🚀 Testing Ensemble vs Single Model Confidence")
    print("=" * 60)
    
    # Test examples with varying difficulty
    test_cases = [
        "I absolutely love this product! It's amazing!",
        "This is the worst thing I've ever bought",
        "It's okay, nothing special",
        "DM me for more details",
        "Thanks so much for sharing this!",
        "You're such an idiot",
        "Not sure about this one...",
        "Best purchase ever!!!",
        "Terrible customer service",
        "Could be better I guess"
    ]
    
    try:
        # Test with ensemble (default)
        print("\n📊 ENSEMBLE PREDICTOR RESULTS:")
        print("-" * 40)
        
        from cyberprint_ml_predictor import CyberPrintMLPredictor
        ensemble_predictor = CyberPrintMLPredictor(use_ensemble=True)
        
        ensemble_results = []
        for text in test_cases:
            result = ensemble_predictor.predict_text([text])[0]
            ensemble_results.append(result)
            
            print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"  Prediction: {result['predicted_label']}")
            print(f"  Confidence: {result['predicted_score']:.3f}")
            if 'ensemble_info' in result:
                print(f"  Models used: {result['enhancement_metadata'].get('models_used', [])}")
                print(f"  Agreement boost: +{result['ensemble_info'].get('agreement_boost', 0):.3f}")
            print()
        
        # Calculate average confidence
        avg_ensemble_confidence = sum(r['predicted_score'] for r in ensemble_results) / len(ensemble_results)
        print(f"📈 AVERAGE ENSEMBLE CONFIDENCE: {avg_ensemble_confidence:.3f} ({avg_ensemble_confidence*100:.1f}%)")
        
        # Test with single DeBERTa model
        print("\n🤖 SINGLE DeBERTa MODEL RESULTS:")
        print("-" * 40)
        
        single_predictor = CyberPrintMLPredictor(use_ensemble=False)
        
        single_results = []
        for text in test_cases:
            result = single_predictor.predict_text([text])[0]
            single_results.append(result)
            
            print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"  Prediction: {result['predicted_label']}")
            print(f"  Confidence: {result['predicted_score']:.3f}")
            print()
        
        # Calculate average confidence
        avg_single_confidence = sum(r['predicted_score'] for r in single_results) / len(single_results)
        print(f"📈 AVERAGE SINGLE MODEL CONFIDENCE: {avg_single_confidence:.3f} ({avg_single_confidence*100:.1f}%)")
        
        # Compare results
        print("\n🎯 CONFIDENCE COMPARISON:")
        print("-" * 40)
        improvement = avg_ensemble_confidence - avg_single_confidence
        improvement_pct = (improvement / avg_single_confidence) * 100
        
        print(f"Single Model Average: {avg_single_confidence:.3f} ({avg_single_confidence*100:.1f}%)")
        print(f"Ensemble Average:     {avg_ensemble_confidence:.3f} ({avg_ensemble_confidence*100:.1f}%)")
        print(f"Improvement:          +{improvement:.3f} (+{improvement_pct:.1f}%)")
        
        # Check if we reached 93%+ target
        target_reached = avg_ensemble_confidence >= 0.93
        print(f"\n🎯 TARGET STATUS: {'✅ REACHED' if target_reached else '❌ NOT REACHED'} (93%+ goal)")
        
        if target_reached:
            print("🎉 SUCCESS! Ensemble method achieved 93%+ authentic confidence!")
        else:
            needed = 0.93 - avg_ensemble_confidence
            print(f"📊 Need +{needed:.3f} more confidence to reach 93% target")
            print("\n🔧 NEXT STEPS TO REACH 93%:")
            print("1. Add more diverse models to ensemble")
            print("2. Collect more high-quality training data")
            print("3. Fine-tune existing models with better hyperparameters")
            print("4. Implement temperature scaling for confidence calibration")
        
        return {
            'ensemble_avg': avg_ensemble_confidence,
            'single_avg': avg_single_confidence,
            'improvement': improvement,
            'target_reached': target_reached
        }
        
    except Exception as e:
        print(f"❌ Error testing ensemble: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_ensemble_confidence()
    
    if results:
        print(f"\n📋 SUMMARY:")
        print(f"Ensemble achieved {results['ensemble_avg']*100:.1f}% average confidence")
        print(f"Target (93%+): {'✅ REACHED' if results['target_reached'] else '❌ NOT REACHED'}")
