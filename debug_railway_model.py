#!/usr/bin/env python3
"""Debug script to check which model Railway is actually using"""
import os
import sys
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append('/Users/deaxhavara/CyberPrint')
from cyberprint_ml_predictor import CyberPrintMLPredictor

# Check what directories exist
possible_dirs = [
    os.path.join(os.path.dirname(__file__), "models", "deberta_active_learning_4epochs"),
    os.path.join(os.path.dirname(__file__), "cyberprint", "models", "deberta_full_e4"),
    os.path.join(os.path.dirname(__file__), "cyberprint", "models", "deberta_full"),
    os.path.join(os.path.dirname(__file__), "cyberprint", "models", "deberta_enhanced")
]

print("=== DIRECTORY CHECK ===")
for dir_path in possible_dirs:
    exists = os.path.exists(dir_path)
    print(f"{dir_path}: {'EXISTS' if exists else 'NOT FOUND'}")
    if exists:
        files = os.listdir(dir_path)[:5]  # Show first 5 files
        print(f"  Files: {files}")

print("\n=== MODEL INITIALIZATION ===")
try:
    predictor = CyberPrintMLPredictor()
    print(f"Predictor type: {predictor.predictor_type}")
    
    if hasattr(predictor, 'predictor'):
        print(f"Predictor object: {type(predictor.predictor)}")
    
    print("\n=== TEST PREDICTION ===")
    result = predictor.predict_text(['Thank you so much!'], include_sub_labels=True)
    print(f"Result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
