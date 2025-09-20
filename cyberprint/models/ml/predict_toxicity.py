import os
import pathlib
import logging
from typing import List, Optional, Union, Dict, Any
import json

import joblib
import numpy as np
from cyberprint.label_mapping import LABELS_FLAT as CATEGORY_LABELS, CATEGORY_EMOTIONS
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
if os.environ.get('CYBERPRINT_DEBUG') == '1':
    logging.basicConfig(level=logging.DEBUG)

# Force load .env immediately
_env_path = pathlib.Path(__file__).parents[3] / '.env'
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path))
    logger.info("Loaded .env from %s", _env_path)
else:
    logger.info(".env not found at %s", _env_path)

# Load .env automatically (fallback)
try:
    _env_path = pathlib.Path(__file__).parents[2] / '.env'
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path))
        logger.debug('Loaded environment variables from %s via python-dotenv', _env_path)
except Exception:
    # Fallback manual parser
    try:
        _env_path = pathlib.Path(__file__).parents[2] / '.env'
        if _env_path.exists():
            with open(_env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            logger.debug('Loaded environment variables from %s via fallback parser', _env_path)
    except Exception as ex:
        logger.debug('Could not auto-load .env: %s', ex)

# Globals
_model = None
_vectorizer = None
_pos_index_list = None  # positive-class indices
_thresholds = None  # thresholds loaded from labels.json

# Candidate artifact locations (built dynamically so repo-relative files are found)
MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_DIR, '..', '..'))

CANDIDATE_MODEL_PATHS = [
    os.path.join(MODULE_DIR, 'polished_model.joblib'),
    os.path.join(MODULE_DIR, 'polished_model.pkl'),
    os.path.join(MODULE_DIR, 'multi_label_model.joblib'),
    os.path.join(MODULE_DIR, 'final_model.pkl'),
    os.path.join(REPO_ROOT, 'models', 'ml', 'polished_model.joblib'),
    os.path.join(REPO_ROOT, 'models', 'ml', 'multi_label_model.joblib'),
    os.path.join(REPO_ROOT, 'cyberprint', 'models', 'ml', 'polished_model.joblib'),
    os.path.join(REPO_ROOT, 'cyberprint', 'models', 'ml', 'polished_model.pkl'),
    'models/ml/polished_model.joblib',
    '../models/ml/polished_model.joblib',
]

CANDIDATE_VECTORIZER_PATHS = [
    os.path.join(MODULE_DIR, 'polished_vectorizer.joblib'),
    os.path.join(MODULE_DIR, 'vectorizer.pkl'),
    os.path.join(MODULE_DIR, 'tfidf_vectorizer.joblib'),
    os.path.join(REPO_ROOT, 'models', 'ml', 'polished_vectorizer.joblib'),
    os.path.join(REPO_ROOT, 'cyberprint', 'models', 'ml', 'polished_vectorizer.joblib'),
    'models/ml/polished_vectorizer.joblib',
    '../models/ml/polished_vectorizer.joblib',
]


def _find_existing_path(candidate_paths: List[str]) -> Optional[str]:
    """Return the first existing path from the list of candidate paths, or None if none exist."""
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def _load_model_and_vectorizer() -> tuple:
    """Load the model and vectorizer, preferring environment variables, with sanity checks."""
    global _model, _vectorizer, _pos_index_list, _thresholds
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    env_model = os.environ.get('CYBERPRINT_MODEL_PATH')
    env_vec = os.environ.get('CYBERPRINT_VECTORIZER_PATH')

    model_path = env_model if env_model and os.path.exists(env_model) else _find_existing_path(CANDIDATE_MODEL_PATHS)
    vec_path = env_vec if env_vec and os.path.exists(env_vec) else _find_existing_path(CANDIDATE_VECTORIZER_PATHS)

    if not model_path or not vec_path:
        raise FileNotFoundError(
            f"Missing artifacts: model={model_path}, vectorizer={vec_path}\n"
            f"Env model={env_model or 'unset'}\nEnv vectorizer={env_vec or 'unset'}\n"
            f"Candidate models={CANDIDATE_MODEL_PATHS}\nCandidate vectorizers={CANDIDATE_VECTORIZER_PATHS}"
        )

    _model = joblib.load(model_path)
    _vectorizer = joblib.load(vec_path)

    # Sanity check: feature alignment
    if hasattr(_vectorizer, 'vocabulary_') and hasattr(_model, 'n_features_in_'):
        if len(_vectorizer.vocabulary_) != _model.n_features_in_:
            raise ValueError(
                f"Vectorizer features ({len(_vectorizer.vocabulary_)}) != model input size ({_model.n_features_in_}). "
                "Ensure model and vectorizer are from the same training run."
            )

    # Positive-class index inference
    _pos_index_list = None
    try:
        estimators = getattr(_model, 'estimators_', None) or getattr(_model, 'estimators', None)
        if isinstance(estimators, (list, tuple)):
            _pos_index_list = []
            for est in estimators:
                classes = getattr(est, 'classes_', None)
                if classes is not None:
                    idx = next((i for i, c in enumerate(classes) if c in (1, '1', True)), len(classes) - 1)
                    _pos_index_list.append(idx)
                else:
                    _pos_index_list.append(-1)
        else:
            if hasattr(_model, 'classes_'):
                _pos_index_list = None
    except Exception as ex:
        logger.debug('Could not infer positive-class indices: %s', ex)

    # Load thresholds from labels.json in same directory as model_path
    _thresholds = None
    try:
        model_dir = os.path.dirname(model_path)
        labels_json_path = os.path.join(model_dir, 'labels.json')
        if os.path.exists(labels_json_path):
            with open(labels_json_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
            if isinstance(labels_data, dict) and 'thresholds' in labels_data:
                if isinstance(labels_data['thresholds'], dict):
                    _thresholds = labels_data['thresholds']
                    logger.debug("Loaded thresholds (dict) from %s", labels_json_path)
                elif isinstance(labels_data['thresholds'], list):
                    try:
                        from cyberprint.label_mapping import LABELS_FLAT
                        _thresholds = {label: float(val) for label, val in zip(LABELS_FLAT, labels_data['thresholds'])}
                        logger.debug("Loaded thresholds (list mapped to labels) from %s", labels_json_path)
                    except Exception as ex:
                        logger.debug("Failed to map list thresholds to labels: %s", ex)
    except Exception as ex:
        logger.debug("Failed to load thresholds from labels.json: %s", ex)

    logger.debug('Loaded model=%s, vectorizer=%s', model_path, vec_path)
    return _model, _vectorizer


def _extract_positive_probs(probs: Any, n_samples: int, n_labels: int) -> np.ndarray:
    """Normalize predict_proba outputs into (n_samples, n_labels) array with probabilities in [0,1]."""
    global _pos_index_list
    pos = np.zeros((n_samples, n_labels), dtype=float)

    if isinstance(probs, (list, tuple)):
        for j, p in enumerate(probs):
            if j >= n_labels:
                break
            p = np.asarray(p)
            idx = _pos_index_list[j] if _pos_index_list and j < len(_pos_index_list) and _pos_index_list[j] >= 0 else None
            if p.ndim == 2:
                pos[:, j] = p[:, idx] if idx is not None and idx < p.shape[1] else p[:, -1]
            elif p.ndim == 1:
                pos[:, j] = p
            else:
                try:
                    pos[:, j] = p.reshape(n_samples, -1)[:, 0]
                except Exception:
                    pos[:, j] = 0.0
        return np.clip(pos, 0.0, 1.0)

    p = np.asarray(probs)
    if p.ndim == 2:
        if p.shape == (n_samples, n_labels):
            return np.clip(p.astype(float), 0.0, 1.0)
        if p.shape == (n_labels, n_samples):
            return np.clip(p.T.astype(float), 0.0, 1.0)
        if p.shape == (n_samples, 2 * n_labels):
            try:
                return np.clip(p.reshape(n_samples, n_labels, 2)[:, :, 1].astype(float), 0.0, 1.0)
            except Exception:
                pass
        if p.shape[0] == n_samples and p.shape[1] >= n_labels:
            return np.clip(p[:, :n_labels].astype(float), 0.0, 1.0)
    return pos


def _safe_transform(vectorizer: Any, texts: List[str]) -> Optional[Any]:
    """Safely transform texts using the vectorizer, returning None on failure."""
    try:
        return vectorizer.transform(texts)
    except Exception as e:
        logger.error('Vectorizer.transform failed: %s', e)
        return None


def predict_toxicity(texts: Union[str, List[Optional[str]]]) -> List[Dict[str, Any]]:
    """
    Predict toxicity for a list of texts.

    Args:
        texts: A single text string or a list of text strings to classify.

    Returns:
        A list of dictionaries containing category probabilities, emotion scores,
        dominant category, dominant category score, top emotion, and top emotion score.
    """
    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [texts]
    texts = ["" if t is None else str(t) for t in texts]
    n = len(texts)
    if n == 0:
        return []

    try:
        model, vectorizer = _load_model_and_vectorizer()
    except Exception as e:
        logger.error('Artifact load error: %s', e)
        return [_zero_result() for _ in range(n)]

    X = _safe_transform(vectorizer, texts)
    if X is None:
        return [_zero_result() for _ in range(n)]

    try:
        probs = model.predict_proba(X)
    except Exception:
        try:
            probs = np.asarray(model.predict(X))
        except Exception:
            try:
                raw = model.decision_function(X)
                probs = 1.0 / (1.0 + np.exp(-np.asarray(raw)))
            except Exception as e:
                logger.error('Model prediction failed: %s', e)
                probs = None

    n_labels = len(CATEGORY_LABELS)
    pos = _extract_positive_probs(probs, n, n_labels) if probs is not None else np.zeros((n, n_labels))
    pos = np.clip(pos, 0.0, 1.0)

    results = []
    for i in range(n):
        category_probs = {}
        category_flags = {}
        for j, label in enumerate(CATEGORY_LABELS):
            prob = float(pos[i, j])
            threshold = 0.5
            if _thresholds and label in _thresholds:
                try:
                    threshold = float(_thresholds[label])
                except Exception:
                    threshold = 0.5
            is_active = prob >= threshold
            logger.debug("Text %d label=%s prob=%.3f threshold=%.2f active=%s", i, label, prob, threshold, is_active)
            category_probs[label] = round(prob * 100.0, 1)
            category_flags[label] = is_active

        result = {**category_probs}
        thresholds_dict = {label: float(_thresholds.get(label, 0.5)) if _thresholds else 0.5 for label in CATEGORY_LABELS}
        result["thresholds"] = thresholds_dict
        result["active"] = category_flags

        # emotions default to 0
        for category, emotions in CATEGORY_EMOTIONS.items():
            for e in emotions:
                result[e] = 0.0

        # Determine dominant_category from thresholded positives if any, else fallback to max prob
        active_labels = [label for label, active in category_flags.items() if active]
        if active_labels:
            # Among active labels, pick the one with highest prob
            dominant_category = max(active_labels, key=lambda lbl: category_probs.get(lbl, 0.0))
        else:
            dominant_category = max(category_probs, key=category_probs.get, default="")

        result['dominant_category'] = dominant_category
        result['dominant_category_score'] = category_probs.get(dominant_category, 0.0)
        emotions_in_dominant = CATEGORY_EMOTIONS.get(dominant_category, [])
        if emotions_in_dominant:
            result['top_emotion'] = emotions_in_dominant[0]
            result['top_emotion_score'] = 0.0

        results.append(result)
    return results


def _zero_result() -> Dict[str, Any]:
    """
    Generate a zeroed-out result dictionary for toxicity prediction.

    Returns:
        A dictionary with all category and emotion scores set to 0,
        and dominant category and scores set to empty or zero.
    """
    r = {label: 0.0 for label in CATEGORY_LABELS}
    for emotions in CATEGORY_EMOTIONS.values():
        for e in emotions:
            r[e] = 0.0
    r['dominant_category'] = ''
    r['dominant_category_score'] = 0.0
    r['top_emotion'] = ''
    r['top_emotion_score'] = 0.0
    return r
