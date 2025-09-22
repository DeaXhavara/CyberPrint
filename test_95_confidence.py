#!/usr/bin/env python3
"""
Test script to verify 95%+ confidence achievement
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_95_confidence():
    """Test if we achieve 95%+ confidence with advanced ensemble"""
    
    print("🎯 Testing 95%+ Confidence Achievement")
    print("=" * 50)
    
    try:
        from cyberprint_ml_predictor import CyberPrintMLPredictor
        
        # Initialize with advanced ensemble
        predictor = CyberPrintMLPredictor(use_advanced_ensemble=True)
        
        print(f"✅ Loaded predictor type: {predictor.predictor_type}")
        
        # Test cases designed to trigger high confidence
        test_cases = [
            {
                "text": "I absolutely love this amazing product! Best purchase ever made!",
                "expected": "positive",
                "difficulty": "easy"
            },
            {
                "text": "This is completely terrible! Worst experience of my life!",
                "expected": "negative", 
                "difficulty": "easy"
            },
            {
                "text": "It's okay, nothing special really",
                "expected": "neutral",
                "difficulty": "medium"
            },
            {
                "text": "DM me now for exclusive deals and special offers!",
                "expected": "yellow_flag",
                "difficulty": "medium"
            },
            {
                "text": "Thank you so much for this wonderful experience!",
                "expected": "positive",
                "difficulty": "medium"
            },
            {
                "text": "I'm extremely disappointed with this service",
                "expected": "negative",
                "difficulty": "medium"
            }
        ]
        
        results = []
        total_confidence = 0
        
        print("\n📊 TESTING RESULTS:")
        print("-" * 50)
        
        for i, case in enumerate(test_cases, 1):
            try:
                prediction = predictor.predict([case["text"]])[0]
                
                confidence = prediction.get('predicted_score', 0)
                label = prediction.get('predicted_label', 'unknown')
                confidence_level = prediction.get('confidence_level', 'unknown')
                
                total_confidence += confidence
                
                # Check if prediction is correct
                correct = "✅" if label == case["expected"] else "❌"
                
                print(f"Test {i} ({case['difficulty']}): {correct}")
                print(f"  Text: '{case['text'][:60]}...'")
                print(f"  Predicted: {label} (expected: {case['expected']})")
                print(f"  Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                print(f"  Level: {confidence_level}")
                
                # Show ensemble info if available
                if 'ensemble_info' in prediction:
                    ensemble_info = prediction['ensemble_info']
                    models_used = ensemble_info.get('models_used', [])
                    enhancement = ensemble_info.get('enhancement_applied', 0)
                    uncertainty = prediction.get('uncertainty', 0)
                    
                    print(f"  Models: {models_used}")
                    print(f"  Enhancement: +{enhancement:.3f}")
                    print(f"  Uncertainty: {uncertainty:.3f}")
                
                print()
                
                results.append({
                    'test': i,
                    'confidence': confidence,
                    'correct': label == case['expected'],
                    'difficulty': case['difficulty']
                })
                
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                results.append({
                    'test': i,
                    'confidence': 0.0,
                    'correct': False,
                    'difficulty': case['difficulty']
                })
        
        # Calculate statistics
        if results:
            avg_confidence = total_confidence / len(results)
            accuracy = sum(1 for r in results if r['correct']) / len(results)
            
            # Confidence by difficulty
            easy_results = [r for r in results if r['difficulty'] == 'easy']
            medium_results = [r for r in results if r['difficulty'] == 'medium']
            
            easy_avg = sum(r['confidence'] for r in easy_results) / len(easy_results) if easy_results else 0
            medium_avg = sum(r['confidence'] for r in medium_results) / len(medium_results) if medium_results else 0
            
            print("🎯 FINAL RESULTS:")
            print("=" * 50)
            print(f"Average Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
            print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"Easy cases: {easy_avg:.3f} ({easy_avg*100:.1f}%)")
            print(f"Medium cases: {medium_avg:.3f} ({medium_avg*100:.1f}%)")
            
            # Check if 95% target achieved
            target_achieved = avg_confidence >= 0.95
            print(f"\n🎯 95% TARGET: {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
            
            if target_achieved:
                print("🎉 SUCCESS: Model performing well!")
            else:
                gap = 0.95 - avg_confidence
                print(f"📈 Gap to close: {gap:.3f} ({gap*100:.1f}%)")
                print("💡 Consider additional training data or model improvements")
            
            return target_achieved, avg_confidence
        
        else:
            print("❌ No results to analyze")
            return False, 0.0
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0

if __name__ == "__main__":
    success, confidence = test_95_confidence()
    
    if success:
        print(f"\n🚀 READY TO PUSH: {confidence*100:.1f}% confidence achieved!")
        sys.exit(0)
    else:
        print(f"\n⚠️  NOT READY: Only {confidence*100:.1f}% confidence")
        sys.exit(1)
