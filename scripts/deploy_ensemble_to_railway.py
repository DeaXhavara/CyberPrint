#!/usr/bin/env python3
"""
Script to deploy ensemble predictor to Railway and test confidence improvements
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_railway_deployment_summary():
    """Create summary of what needs to be deployed to Railway"""
    
    deployment_info = {
        "new_files": [
            "cyberprint/models/ml/ensemble_predictor.py",
            "data/high_confidence_training_data.csv",
            "config/high_confidence_training.json",
            "scripts/improve_model_confidence.py",
            "scripts/collect_training_data.py",
            "scripts/train_high_confidence_model.py",
            "test_ensemble_confidence.py"
        ],
        "modified_files": [
            "cyberprint_ml_predictor.py",
            "cyberprint/models/ml/deberta_predictor.py"
        ],
        "expected_improvements": {
            "current_authentic": "~80%",
            "ensemble_boost": "+10-15%",
            "expected_result": "90-95%",
            "target": "93%+"
        },
        "testing_plan": [
            "1. Commit and push all changes to GitHub",
            "2. Railway will auto-deploy from GitHub",
            "3. Test ensemble predictor via API endpoints",
            "4. Measure actual confidence improvements",
            "5. Compare single model vs ensemble results"
        ]
    }
    
    return deployment_info

def create_test_api_calls():
    """Create API test calls to measure confidence"""
    
    test_cases = [
        {
            "text": "I absolutely love this product! Best purchase ever!",
            "expected_label": "positive",
            "difficulty": "easy"
        },
        {
            "text": "This is terrible! Worst experience ever!",
            "expected_label": "negative", 
            "difficulty": "easy"
        },
        {
            "text": "It's okay I guess, nothing special",
            "expected_label": "neutral",
            "difficulty": "medium"
        },
        {
            "text": "DM me for exclusive deals and discounts!",
            "expected_label": "yellow_flag",
            "difficulty": "medium"
        },
        {
            "text": "Thanks for sharing, really appreciate it :)",
            "expected_label": "positive",
            "difficulty": "hard"
        },
        {
            "text": "Not sure if this is worth the money...",
            "expected_label": "neutral",
            "difficulty": "hard"
        }
    ]
    
    api_test_script = '''
import requests
import json

# Railway API endpoint
API_URL = "https://cyberprint-production-a8f616c.up.railway.app"

def test_ensemble_confidence():
    """Test ensemble vs single model confidence"""
    
    test_cases = ''' + str(test_cases) + '''
    
    print("🚀 Testing Ensemble Confidence on Railway")
    print("=" * 60)
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\\nTest {i}: {case['difficulty'].upper()} - '{case['text'][:50]}...'")
        
        try:
            # Test with ensemble (default)
            response = requests.post(f"{API_URL}/predict", 
                json={"text": case["text"]},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                confidence = result.get('predicted_score', 0)
                label = result.get('predicted_label', 'unknown')
                
                print(f"  Prediction: {label}")
                print(f"  Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                
                # Check if ensemble info is available
                if 'ensemble_info' in result:
                    models_used = result.get('enhancement_metadata', {}).get('models_used', [])
                    agreement_boost = result.get('ensemble_info', {}).get('agreement_boost', 0)
                    print(f"  Models: {models_used}")
                    print(f"  Agreement boost: +{agreement_boost:.3f}")
                
                results.append({
                    'text': case['text'],
                    'expected': case['expected_label'],
                    'predicted': label,
                    'confidence': confidence,
                    'difficulty': case['difficulty'],
                    'correct': label == case['expected_label']
                })
            else:
                print(f"  ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
    
    # Calculate statistics
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        
        print(f"\\n📊 RESULTS SUMMARY:")
        print(f"Average Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Target Reached: {'✅ YES' if avg_confidence >= 0.93 else '❌ NO'} (93%+ goal)")
        
        # Breakdown by difficulty
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in results if r['difficulty'] == difficulty]
            if diff_results:
                diff_avg = sum(r['confidence'] for r in diff_results) / len(diff_results)
                print(f"{difficulty.capitalize()} cases: {diff_avg:.3f} ({diff_avg*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    test_ensemble_confidence()
'''
    
    return api_test_script

def main():
    """Main function to prepare Railway deployment"""
    
    print("🚀 Preparing Railway Deployment for Ensemble Testing")
    print("=" * 60)
    
    # Get deployment info
    deployment = create_railway_deployment_summary()
    
    print("📁 New Files to Deploy:")
    for file in deployment["new_files"]:
        print(f"  ✅ {file}")
    
    print("\n📝 Modified Files:")
    for file in deployment["modified_files"]:
        print(f"  🔄 {file}")
    
    print(f"\n📈 Expected Improvements:")
    for key, value in deployment["expected_improvements"].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎯 Deployment Plan:")
    for step in deployment["testing_plan"]:
        print(f"  {step}")
    
    # Create API test script
    test_script = create_test_api_calls()
    
    # Save test script
    test_script_path = project_root / "test_railway_ensemble.py"
    with open(test_script_path, 'w') as f:
        f.write(test_script)
    
    print(f"\n✅ Created API test script: {test_script_path}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Commit all changes to GitHub")
    print(f"2. Railway will auto-deploy the ensemble predictor")
    print(f"3. Run: python3 test_railway_ensemble.py")
    print(f"4. Measure actual confidence improvements")
    print(f"5. If 93%+ achieved, deploy to production!")
    
    return deployment

if __name__ == "__main__":
    main()
