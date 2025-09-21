
import requests
import json

# Railway API endpoint
API_URL = "https://cyberprint-production-a8f616c.up.railway.app"

def test_ensemble_confidence():
    """Test ensemble vs single model confidence"""
    
    test_cases = [{'text': 'I absolutely love this product! Best purchase ever!', 'expected_label': 'positive', 'difficulty': 'easy'}, {'text': 'This is terrible! Worst experience ever!', 'expected_label': 'negative', 'difficulty': 'easy'}, {'text': "It's okay I guess, nothing special", 'expected_label': 'neutral', 'difficulty': 'medium'}, {'text': 'DM me for exclusive deals and discounts!', 'expected_label': 'yellow_flag', 'difficulty': 'medium'}, {'text': 'Thanks for sharing, really appreciate it :)', 'expected_label': 'positive', 'difficulty': 'hard'}, {'text': 'Not sure if this is worth the money...', 'expected_label': 'neutral', 'difficulty': 'hard'}]
    
    print("🚀 Testing Ensemble Confidence on Railway")
    print("=" * 60)
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['difficulty'].upper()} - '{case['text'][:50]}...'")
        
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
        
        print(f"\n📊 RESULTS SUMMARY:")
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
