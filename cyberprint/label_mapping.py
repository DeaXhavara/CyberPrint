# cyberprint/label_mapping.py

"""
This module provides a three-level hierarchical mapping system for emotion labels 
used in the CyberPrint project. The hierarchy consists of:
- Level 1: Impact categories (Harmful, Neutral, Supportive)
- Level 2: Behavioral subcategories within each impact category
- Level 3: Emotion overlays associated with each behavioral subcategory

This structure facilitates integration with smaller curated datasets and the GoEmotions dataset,
standardizing label categories for annotation, model training, and analysis.
"""

from typing import Dict, List

# Level 1: Main sentiment categories (updated to match new dataset structure)
IMPACT_CATEGORIES: List[str] = ["positive", "negative", "neutral", "yellow_flag"]

# Level 2 & 3: Hierarchical mapping from Main Sentiment -> Subcategory -> Emotions
LABEL_HIERARCHY: Dict[str, Dict[str, List[str]]] = {
    "positive": {
        "gratitude": ["thank", "thanks", "grateful", "appreciate", "blessed"],
        "compliments": ["amazing", "awesome", "brilliant", "excellent", "fantastic", "great", "wonderful"],
        "positive_actions": ["help", "support", "assist", "encourage", "motivate", "inspire"],
        "joy_happiness": ["happy", "joy", "excited", "thrilled", "delighted", "pleased"]
    },
    "negative": {
        "toxic": ["hate", "despise", "loathe", "disgusting", "revolting", "vile"],
        "personal_attacks": ["stupid", "idiot", "moron", "dumb", "fool", "loser", "pathetic"],
        "harsh_language": ["damn", "hell", "crap", "sucks", "terrible", "awful", "horrible"],
        "anger_frustration": ["angry", "mad", "furious", "rage", "frustrated", "annoyed"]
    },
    "neutral": {
        "fact_based": ["according to", "research shows", "studies indicate", "data suggests"],
        "question_based": ["what", "how", "why", "when", "where", "who", "which"],
        "unbiased": ["neutral", "objective", "impartial", "fair", "balanced"],
        "informational": ["update", "news", "report", "announcement", "notice"]
    },
    "yellow_flag": {
        "irony": ["oh great", "perfect", "exactly what i needed", "just what i wanted"],
        "sarcasm": ["sure", "right", "of course", "obviously", "clearly"],
        "humor": ["lol", "haha", "funny", "hilarious", "comedy", "joke"],
        "skeptical": ["really", "seriously", "come on", "give me a break"]
    }
}

# Flattened mappings for convenience

# Mapping from emotion to (main sentiment, subcategory)
EMOTION_TO_LABELS: Dict[str, Dict[str, str]] = {}
for sentiment, subcategories in LABEL_HIERARCHY.items():
    for subcategory, emotions in subcategories.items():
        for emotion in emotions:
            EMOTION_TO_LABELS[emotion] = {
                "sentiment": sentiment,
                "subcategory": subcategory
            }

# Flattened list of all emotions
ALL_EMOTIONS: List[str] = list(EMOTION_TO_LABELS.keys())

# Flattened list of subcategories with their main sentiment for reference
SUBCATEGORY_TO_SENTIMENT: Dict[str, str] = {
    subcategory: sentiment
    for sentiment, subcategories in LABEL_HIERARCHY.items()
    for subcategory in subcategories.keys()
}

# Legacy compatibility mappings (for backward compatibility with old code)
LABELS_FLAT = IMPACT_CATEGORIES  # Main sentiment categories
CATEGORY_EMOTIONS = {sentiment: [emotion for subcat in subcats.values() for emotion in subcat] 
                    for sentiment, subcats in LABEL_HIERARCHY.items()}

# Mental health warning categories (separate from sentiment)
MENTAL_HEALTH_CATEGORIES = ["emotional_distress", "anxiety", "crisis", "seeking_help"]


def validate_label_hierarchy() -> None:
    """
    Validates the consistency and integrity of the label hierarchy mappings.
    Checks include:
    - All main sentiment categories are valid and complete.
    - No overlapping emotions across different subcategories or sentiments.
    - Reverse mappings are consistent with the hierarchy.
    Raises AssertionError on any inconsistency.
    """
    # Check main sentiment categories completeness
    assert set(LABEL_HIERARCHY.keys()) == set(IMPACT_CATEGORIES), \
        f"Main sentiment categories mismatch: expected {set(IMPACT_CATEGORIES)}, found {set(LABEL_HIERARCHY.keys())}"

    # Collect all emotions and check for duplicates
    emotions_seen = set()
    for sentiment, subcategories in LABEL_HIERARCHY.items():
        for subcategory, emotions in subcategories.items():
            for emotion in emotions:
                assert emotion not in emotions_seen, f"Duplicate emotion '{emotion}' found in hierarchy"
                emotions_seen.add(emotion)

                # Check reverse mapping exists and is correct
                assert emotion in EMOTION_TO_LABELS, f"Emotion '{emotion}' missing in EMOTION_TO_LABELS"
                labels = EMOTION_TO_LABELS[emotion]
                assert labels["sentiment"] == sentiment, f"Sentiment mismatch for emotion '{emotion}': expected '{sentiment}', found '{labels['sentiment']}'"
                assert labels["subcategory"] == subcategory, f"Subcategory mismatch for emotion '{emotion}': expected '{subcategory}', found '{labels['subcategory']}'"

    # Check all emotions in EMOTION_TO_LABELS exist in hierarchy
    hierarchy_emotions = set(ALL_EMOTIONS)
    assert emotions_seen == hierarchy_emotions, \
        "Mismatch between emotions in hierarchy and EMOTION_TO_LABELS"

    # Check subcategories map correctly to main sentiment
    for subcategory, sentiment in SUBCATEGORY_TO_SENTIMENT.items():
        assert subcategory in [s for subcats in LABEL_HIERARCHY.values() for s in subcats], \
            f"Subcategory '{subcategory}' not found in LABEL_HIERARCHY"
        assert sentiment in IMPACT_CATEGORIES, \
            f"Main sentiment '{sentiment}' for subcategory '{subcategory}' is invalid"

validate_label_hierarchy()
