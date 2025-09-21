#!/usr/bin/env python3
"""
Script to collect high-quality training data for improving model confidence
"""

import csv
import json
from pathlib import Path

def create_high_quality_dataset():
    """Create a curated dataset of high-confidence examples"""
    
    # High-confidence positive examples
    positive_examples = [
        "I absolutely love this product! Best purchase ever!",
        "This is amazing! Exceeded all my expectations!",
        "Fantastic quality and great customer service!",
        "Perfect! Exactly what I was looking for!",
        "Outstanding! Would definitely recommend to everyone!",
        "Incredible value for money! So happy with this!",
        "Brilliant! This solved all my problems!",
        "Excellent! Five stars all the way!",
        "Wonderful experience! Thank you so much!",
        "Superb quality! Will buy again for sure!",
        "This made my day! Absolutely perfect!",
        "Love love love this! Can't recommend enough!",
        "Best decision I made! So grateful!",
        "This is exactly what I needed! Perfect!",
        "Amazing results! Better than expected!",
        "Fantastic! This is a game changer!",
        "Perfect solution! Works flawlessly!",
        "Outstanding service! Very impressed!",
        "This is incredible! So well made!",
        "Excellent quality! Worth every penny!"
    ]
    
    # High-confidence negative examples
    negative_examples = [
        "This is absolutely terrible! Worst purchase ever!",
        "Complete waste of money! Total garbage!",
        "Horrible quality! Broke immediately!",
        "Terrible customer service! Very disappointed!",
        "This is awful! Nothing works properly!",
        "Disgusting! Would never recommend this!",
        "Pathetic quality! Completely useless!",
        "Horrible experience! Total disaster!",
        "This is trash! Don't waste your money!",
        "Terrible! Regret buying this completely!",
        "Awful! This ruined everything!",
        "Horrible! Worst company ever!",
        "This is garbage! Completely broken!",
        "Terrible quality! Falls apart easily!",
        "Disgusting service! Never again!",
        "This sucks! Total disappointment!",
        "Horrible! Doesn't work at all!",
        "Terrible! Complete rip-off!",
        "This is awful! Waste of time!",
        "Horrible experience! Very angry!"
    ]
    
    # High-confidence neutral examples
    neutral_examples = [
        "The product arrived on time as expected.",
        "Standard quality, nothing special about it.",
        "It works as described in the specifications.",
        "Average performance, meets basic requirements.",
        "The item is functional but unremarkable.",
        "Standard shipping and packaging process.",
        "The product does what it's supposed to do.",
        "Basic functionality, no major issues.",
        "It's an okay product for the price range.",
        "Standard quality, similar to other brands.",
        "The item works but could be improved.",
        "Average experience, nothing noteworthy.",
        "It's functional but not particularly impressive.",
        "Standard product, meets expectations.",
        "The item is adequate for basic needs.",
        "Regular quality, nothing stands out.",
        "It works fine for everyday use.",
        "Standard service, no complaints.",
        "The product is functional as advertised.",
        "Average quality, typical for this category."
    ]
    
    # High-confidence yellow flag examples (suspicious/promotional)
    yellow_flag_examples = [
        "DM me for exclusive deals and discounts!",
        "Check out my profile for amazing offers!",
        "Message me privately for special pricing!",
        "Contact me for wholesale opportunities!",
        "DM for bulk orders and better rates!",
        "Private message me for custom solutions!",
        "Reach out to me for partnership deals!",
        "Message me for exclusive access!",
        "DM me for limited time offers!",
        "Contact me privately for special deals!",
        "Message me for insider information!",
        "DM for VIP customer benefits!",
        "Private message for exclusive discounts!",
        "Contact me for referral bonuses!",
        "Message me for special promotions!",
        "DM me for early access deals!",
        "Reach out privately for better prices!",
        "Message me for exclusive opportunities!",
        "Contact me for premium services!",
        "DM for personalized offers!"
    ]
    
    # Create training dataset
    training_data = []
    
    # Add positive examples
    for text in positive_examples:
        training_data.append({
            'text': text,
            'label': 'positive',
            'confidence_level': 'high',
            'source': 'curated'
        })
    
    # Add negative examples
    for text in negative_examples:
        training_data.append({
            'text': text,
            'label': 'negative',
            'confidence_level': 'high',
            'source': 'curated'
        })
    
    # Add neutral examples
    for text in neutral_examples:
        training_data.append({
            'text': text,
            'label': 'neutral',
            'confidence_level': 'high',
            'source': 'curated'
        })
    
    # Add yellow flag examples
    for text in yellow_flag_examples:
        training_data.append({
            'text': text,
            'label': 'yellow_flag',
            'confidence_level': 'high',
            'source': 'curated'
        })
    
    return training_data

def save_training_data(data, filename="high_confidence_training_data.csv"):
    """Save training data to CSV file"""
    
    output_path = Path(__file__).parent.parent / "data" / filename
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label', 'confidence_level', 'source']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"✅ Saved {len(data)} training examples to {output_path}")
    return output_path

def main():
    """Main function to create and save training data"""
    
    print("🔄 Creating high-quality training dataset...")
    training_data = create_high_quality_dataset()
    
    print(f"📊 Created {len(training_data)} examples:")
    labels = {}
    for item in training_data:
        label = item['label']
        labels[label] = labels.get(label, 0) + 1
    
    for label, count in labels.items():
        print(f"  - {label}: {count} examples")
    
    # Save the data
    output_path = save_training_data(training_data)
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Use this data to retrain your DeBERTa model")
    print(f"2. Combine with existing training data")
    print(f"3. Train for more epochs with better hyperparameters")
    print(f"4. Test the improved model confidence")
    
    return output_path

if __name__ == "__main__":
    main()
