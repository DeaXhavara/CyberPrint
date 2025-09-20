#!/usr/bin/env python3
"""
Train DeBERTa model with human-reviewed active learning data for 4 epochs

This script specifically uses the challenging_examples.csv and edge_cases.csv
that have been human-reviewed to train the DeBERTa model.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """Main training function"""
    # Set up the training arguments
    model_output_dir = project_root / "models" / "deberta_active_learning_4epochs"
    
    # Create output directory if it doesn't exist
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import and run the enhanced training script
    from scripts.train_deberta_enhanced import main as train_main
    
    # Set up arguments for 4 epochs
    import sys
    original_argv = sys.argv
    
    try:
        # Override sys.argv to pass arguments to the training script
        sys.argv = [
            'train_deberta_enhanced.py',
            '--epochs', '4',
            '--out_dir', str(model_output_dir),
            '--model_name', 'microsoft/deberta-v3-base',
            '--batch_size', '8',
            '--learning_rate', '2e-5'
        ]
        
        print("Starting DeBERTa training with active learning data")
        print(f"Output directory: {model_output_dir}")
        print("Using human-reviewed challenging_examples.csv and edge_cases.csv")
        print("Training for 4 epochs")
        print("=" * 60)
        
        # Run the training
        train_main()
        
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()
