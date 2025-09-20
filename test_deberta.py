#!/usr/bin/env python3
import os
import sys
sys.path.append('/Users/deaxhavara/CyberPrint')
from cyberprint_ml_predictor import CyberPrintMLPredictor

# Test multiple comments that should work correctly
test_comments = [
    "Thank you so much!",
    "DM please :)",
    "Thank youu :DD", 
    "OMG I was inactive for 4 months in reddit and this comment made my day ■■ Thank youu",
    "Rap needs grind. A great example is Jessi."
]

try:
    predictor = CyberPrintMLPredictor()
    print(f'Predictor type: {predictor.predictor_type}')
    
    for comment in test_comments:
        result = predictor.predict_text([comment], include_sub_labels=True)
        pred = result[0]
        print(f'"{comment}" → {pred["predicted_label"]} ({pred["predicted_score"]:.1%})')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
