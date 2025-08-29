# cyberprint/models/test_model.py
import os
import joblib

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load polished vectorizer + model
vectorizer = joblib.load(os.path.join(BASE_DIR, "polished_vectorizer.joblib"))
model = joblib.load(os.path.join(BASE_DIR, "polished_model.joblib"))

# Labels
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Test comments
test_comments = [
    "I hate you so much, you're the worst!",
    "You are such a kind and amazing person!",
    "This is stupid and pathetic.",
    "I will find you and hurt you."
]

# Vectorize
X_test = vectorizer.transform(test_comments)

# Predict probabilities
y_proba = model.predict_proba(X_test)

for comment, prob_list in zip(test_comments, y_proba):
    print(f"\nComment: {comment}")
    for label, probs in zip(labels, prob_list):
        confidence = probs[1]  # probability of the positive class
        if confidence >= 0.5:
            print(f" - {label} ({confidence*100:.1f}%)")
