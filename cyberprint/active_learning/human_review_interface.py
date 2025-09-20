#!/usr/bin/env python3
"""
Human Review Interface for Active Learning
==========================================

Simple command-line interface for reviewing and correcting misclassified examples.
"""

import os
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from cyberprint.active_learning.misclassification_detector import MisclassificationDetector
from cyberprint.active_learning.retraining_pipeline import ActiveLearningRetrainingPipeline

logger = logging.getLogger(__name__)

class HumanReviewInterface:
    """Command-line interface for reviewing misclassified examples."""
    
    def __init__(self):
        self.detector = MisclassificationDetector()
        self.retraining_pipeline = ActiveLearningRetrainingPipeline()
        self.valid_sentiments = ['positive', 'negative', 'neutral', 'yellow_flag']
        self.valid_sub_labels = {
            'positive': ['gratitude', 'compliments', 'reinforcing_positive_actions', 'joy_happiness'],
            'negative': ['offensive', 'insulting', 'threatening', 'harsh_criticism'],
            'neutral': ['fact_based', 'question_based', 'lack_of_bias', 'informational'],
            'yellow_flag': ['sarcasm', 'irony', 'internet_slang', 'humor']
        }
    
    def display_statistics(self):
        """Display current misclassification statistics."""
        stats = self.detector.get_statistics()
        
        print("\n" + "="*60)
        print("ACTIVE LEARNING STATISTICS")
        print("="*60)
        print(f"Total Misclassifications: {stats.get('total_misclassifications', 0)}")
        print(f"Human Reviewed: {stats.get('human_reviewed', 0)}")
        print(f"Pending Review: {stats.get('pending_review', 0)}")
        print(f"Corrected Examples: {stats.get('corrected_examples', 0)}")
        
        print("\nMisclassifications by Rule:")
        for rule, count in stats.get('misclassification_by_rule', {}).items():
            print(f"  {rule}: {count}")
        
        print("\nMisclassifications by Type:")
        for mistype, count in stats.get('misclassification_by_type', {}).items():
            print(f"  {mistype}: {count}")
        print("="*60)
    
    def review_examples(self, limit: int = 10, rule_name: str = None):
        """Review misclassified examples interactively."""
        
        # Load unreviewed examples
        df = self.detector.get_misclassified_examples()
        if len(df) == 0:
            print("No misclassified examples found.")
            return
        
        # Filter unreviewed examples
        unreviewed = df[df['human_reviewed'] == 'no'].copy()
        if rule_name:
            unreviewed = unreviewed[unreviewed['rule_triggered'] == rule_name]
        
        if len(unreviewed) == 0:
            print("No unreviewed examples found.")
            return
        
        # Limit examples
        examples_to_review = unreviewed.head(limit)
        
        print(f"\nReviewing {len(examples_to_review)} examples...")
        print("Commands: 'c' = correct, 's' = skip, 'q' = quit, 'a' = accept suggestion")
        
        corrections = []
        
        for idx, (_, row) in enumerate(examples_to_review.iterrows()):
            print("\n" + "-"*80)
            print(f"Example {idx + 1}/{len(examples_to_review)}")
            print(f"Text: {row['text']}")
            print(f"Predicted: {row['predicted_sentiment']} - {row['predicted_sub_label']} ({row['predicted_confidence']:.3f})")
            print(f"Rule Suggestion: {row['rule_suggested_sentiment']} - {row['rule_suggested_sub_label']}")
            print(f"Rule: {row['rule_triggered']}")
            print(f"Issue: {row['misclassification_type']}")
            
            while True:
                action = input("\nAction [c/s/a/q]: ").lower().strip()
                
                if action == 'q':
                    print("Quitting review...")
                    break
                elif action == 's':
                    print("Skipping example...")
                    break
                elif action == 'a':
                    # Accept rule suggestion
                    corrected_sentiment = row['rule_suggested_sentiment']
                    corrected_sub_label = row['rule_suggested_sub_label']
                    notes = "Accepted rule suggestion"
                    
                    corrections.append({
                        'index': row.name,
                        'corrected_sentiment': corrected_sentiment,
                        'corrected_sub_label': corrected_sub_label,
                        'notes': notes
                    })
                    
                    print(f"Accepted: {corrected_sentiment} - {corrected_sub_label}")
                    break
                elif action == 'c':
                    # Manual correction
                    corrected_sentiment, corrected_sub_label, notes = self._get_manual_correction()
                    
                    if corrected_sentiment:
                        corrections.append({
                            'index': row.name,
                            'corrected_sentiment': corrected_sentiment,
                            'corrected_sub_label': corrected_sub_label,
                            'notes': notes
                        })
                        print(f"Corrected: {corrected_sentiment} - {corrected_sub_label}")
                    break
                else:
                    print("Invalid action. Use 'c', 's', 'a', or 'q'.")
            
            if action == 'q':
                break
        
        # Save corrections
        if corrections:
            self._save_corrections(corrections)
            print(f"\nSaved {len(corrections)} corrections.")
        else:
            print("\nNo corrections made.")
    
    def _get_manual_correction(self) -> tuple:
        """Get manual correction from user input."""
        
        print("\nAvailable sentiments: positive, negative, neutral, yellow_flag")
        
        while True:
            sentiment = input("Correct sentiment: ").lower().strip()
            if sentiment in self.valid_sentiments:
                break
            print("Invalid sentiment. Please choose from: positive, negative, neutral, yellow_flag")
        
        print(f"\nAvailable sub-labels for {sentiment}:")
        for sub_label in self.valid_sub_labels[sentiment]:
            print(f"  {sub_label}")
        print("  general")
        
        while True:
            sub_label = input("Correct sub-label: ").lower().strip()
            if sub_label in self.valid_sub_labels[sentiment] or sub_label == 'general':
                break
            print(f"Invalid sub-label for {sentiment}.")
        
        notes = input("Notes (optional): ").strip()
        
        return sentiment, sub_label, notes
    
    def _save_corrections(self, corrections: List[Dict[str, Any]]):
        """Save corrections to the misclassified examples file."""
        
        indices = [c['index'] for c in corrections]
        corrected_sentiments = [c['corrected_sentiment'] for c in corrections]
        corrected_sub_labels = [c['corrected_sub_label'] for c in corrections]
        notes = [c['notes'] for c in corrections]
        
        self.detector.mark_as_reviewed(indices, corrected_sentiments, corrected_sub_labels, notes)
    
    def retrain_model(self):
        """Retrain the model with corrected examples."""
        
        print("\nStarting model retraining with corrected examples...")
        
        try:
            results = self.retraining_pipeline.run_retraining_pipeline()
            
            print("\nRetraining Results:")
            print(f"  Training Accuracy: {results['train_accuracy']:.3f}")
            print(f"  Test Accuracy: {results['test_accuracy']:.3f}")
            print(f"  Original Examples: {results['original_examples']}")
            print(f"  Corrected Examples: {results['corrected_examples']}")
            print(f"  Total Training Examples: {results['total_training_examples']}")
            
            print("\nModel retrained successfully!")
            
        except Exception as e:
            print(f"Retraining failed: {e}")
            logger.error(f"Retraining failed: {e}")
    
    def run_interactive_session(self):
        """Run an interactive review session."""
        
        while True:
            print("\n" + "="*60)
            print("CYBERPRINT ACTIVE LEARNING REVIEW")
            print("="*60)
            print("1. Show statistics")
            print("2. Review examples")
            print("3. Review specific rule")
            print("4. Retrain model")
            print("5. Quit")
            
            choice = input("\nChoose an option [1-5]: ").strip()
            
            if choice == '1':
                self.display_statistics()
            elif choice == '2':
                limit = input("Number of examples to review (default 10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                self.review_examples(limit=limit)
            elif choice == '3':
                self.display_statistics()
                rule_name = input("Enter rule name to review: ").strip()
                limit = input("Number of examples to review (default 10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                self.review_examples(limit=limit, rule_name=rule_name)
            elif choice == '4':
                confirm = input("Are you sure you want to retrain the model? [y/N]: ").lower().strip()
                if confirm == 'y':
                    self.retrain_model()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-5.")

def main():
    """Main entry point for the review interface."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    interface = HumanReviewInterface()
    interface.run_interactive_session()

if __name__ == "__main__":
    main()
