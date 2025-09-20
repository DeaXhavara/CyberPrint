#!/usr/bin/env python3
"""
Create proper sentiment encoder for DeBERTa active learning model
"""
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# Create sentiment encoder with correct labels
sentiment_encoder = LabelEncoder()
sentiment_labels = ['negative', 'neutral', 'positive', 'yellow_flag']
sentiment_encoder.fit(sentiment_labels)

# Create sublabel encoder
sublabel_encoder = LabelEncoder()
sublabel_labels = ['general', 'gratitude', 'sarcasm', 'mental_health', 'criticism']
sublabel_encoder.fit(sublabel_labels)

# Save encoders
model_dir = 'models/deberta_active_learning_4epochs'
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, 'sentiment_encoder.pkl'), 'wb') as f:
    pickle.dump(sentiment_encoder, f)

with open(os.path.join(model_dir, 'sublabel_encoder.pkl'), 'wb') as f:
    pickle.dump(sublabel_encoder, f)

print("✅ Created sentiment_encoder.pkl and sublabel_encoder.pkl")
print(f"Sentiment classes: {sentiment_encoder.classes_}")
print(f"Sublabel classes: {sublabel_encoder.classes_}")

# Test loading
try:
    with open(os.path.join(model_dir, 'sentiment_encoder.pkl'), 'rb') as f:
        test_encoder = pickle.load(f)
    print("✅ Sentiment encoder loads correctly")
except Exception as e:
    print(f"❌ Error loading sentiment encoder: {e}")

try:
    with open(os.path.join(model_dir, 'sublabel_encoder.pkl'), 'rb') as f:
        test_encoder = pickle.load(f)
    print("✅ Sublabel encoder loads correctly")
except Exception as e:
    print(f"❌ Error loading sublabel encoder: {e}")
