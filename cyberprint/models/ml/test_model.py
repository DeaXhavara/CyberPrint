import os
import logging
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "transformer_checkpoint")

_MODEL_AVAILABLE = False
_TOKENIZER = None
_MODEL = None
_THRESHOLDS = [0.5, 0.5, 0.5, 0.5]
_LABELS = ["harmful", "critical", "supportive", "neutral"]

def _load_model_artifacts():
    """Attempt to load tokenizer, model, and labels.json from MODEL_PATH.
    Returns True on success, False otherwise. This is lazy and safe to call at runtime.
    """
    global _MODEL_AVAILABLE, _TOKENIZER, _MODEL, _THRESHOLDS, _LABELS
    if _MODEL_AVAILABLE:
        return True
    try:
        if not os.path.isdir(MODEL_PATH):
            return False
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        labels_path = os.path.join(MODEL_PATH, "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                label_data = json.load(f)
            _THRESHOLDS = label_data.get("thresholds", _THRESHOLDS)
            # allow labels override if present
            _LABELS = label_data.get("labels", _LABELS)
        _MODEL_AVAILABLE = True
        return True
    except Exception:
        # Avoid noisy failures during test collection; caller can handle False
        _MODEL_AVAILABLE = False
        return False

test_comments = [
    "I hate you so much, you're the worst!",
    "You are such a kind and amazing person!",
    "This is stupid and pathetic.",
    "I will find you and hurt you."
]

def predict_comments(comments):
    """
    Tokenize comments, run model, apply sigmoid to logits,
    and return list of dicts with probabilities, thresholds,
    active flags, and dominant category.
    """
    # Ensure artifacts are loaded
    if not _load_model_artifacts():
        raise RuntimeError("Model artifacts not available under MODEL_PATH; cannot run predictions.")

    _MODEL.eval()
    encoded = _TOKENIZER(comments, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = _MODEL(**encoded)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().tolist()

    results = []
    for prob in probs:
        active = [p >= t for p, t in zip(prob, _THRESHOLDS)]
        dominant_idx = int(np.argmax(prob))
        result = {
            "probabilities": {label: p for label, p in zip(_LABELS, prob)},
            "thresholds": {label: t for label, t in zip(_LABELS, _THRESHOLDS)},
            "active": {label: a for label, a in zip(_LABELS, active)},
            "dominant_category": _LABELS[dominant_idx]
        }
        results.append(result)
    return results

def main() -> None:
    """
    Run model predictions on test comments and print results.
    """
    setup_logging()
    logging.info("Predicting comments with transformer model...")
    try:
        results = predict_comments(test_comments)
    except RuntimeError as e:
        logging.error("%s", e)
        logging.info("Skipping prediction run because model artifacts are missing.")
        return

    for comment, res in zip(test_comments, results):
        logging.info(f"\nComment: {comment}")
        for label in _LABELS:
            prob = res["probabilities"][label] * 100
            thresh = res["thresholds"][label] * 100
            active = res["active"][label]
            logging.info(f" - {label}: {prob:.1f}% (threshold {thresh:.1f}%) Active: {active}")
        logging.info(f" Dominant category: {res['dominant_category']}")

if __name__ == "__main__":
    main()
