# cyberprint/label_mapping.py

FLAT_LABELS = {
    "toxic": [
        "anger", "annoyance", "disgust", "disapproval",
        "fear", "grief", "remorse", "sadness"
    ],
    "constructive": [
        "approval", "caring", "curiosity", "gratitude",
        "love", "pride"
    ],
    "neutral": [
        "confusion", "embarrassment", "nervousness",
        "realization", "surprise", "neutral", "disappointment"
    ],
    "positive": [
        "admiration", "amusement", "desire",
        "excitement", "joy", "optimism", "relief"
    ]
}

# Flatten for easier PDF integration
LABELS_FLAT = list(FLAT_LABELS.keys())

# Mapping from Jigsaw dataset labels to our categories
JIGSAW_TO_FLAT_MAPPING = {
    "toxic": "toxic",
    "severe_toxic": "toxic",
    "obscene": "toxic",
    "threat": "toxic",
    "insult": "toxic",
    "identity_hate": "toxic"
}

# Mapping for detailed emotion breakdown within each category
CATEGORY_EMOTIONS = {
    "toxic": ["anger", "annoyance", "disgust", "disapproval", "fear", "grief", "remorse", "sadness"],
    "constructive": ["approval", "caring", "curiosity", "gratitude", "love", "pride"],
    "neutral": ["confusion", "embarrassment", "nervousness", "realization", "surprise", "neutral", "disappointment"],
    "positive": ["admiration", "amusement", "desire", "excitement", "joy", "optimism", "relief"]
}
