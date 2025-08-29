import os
import pathlib
from dotenv import load_dotenv

# Force load .env immediately
_env_path = pathlib.Path(__file__).parents[3] / '.env'
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path))
    print("Loaded .env from", _env_path)
else:
    print(".env not found at", _env_path)

# Now import the rest of your stuff
import joblib
import numpy as np
import logging
from cyberprint.label_mapping import LABELS_FLAT as CATEGORY_LABELS, CATEGORY_EMOTIONS

logger = logging.getLogger(__name__)
if os.environ.get('CYBERPRINT_DEBUG') == '1':
    logging.basicConfig(level=logging.DEBUG)

# Load .env automatically
try:
    from dotenv import load_dotenv
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

# Candidate artifact locations
CANDIDATE_MODEL_PATHS = [
    "C:/Users/User/cyberprintproject/models/ml/polished_model.joblib",
    "C:/Users/User/cyberprintproject/models/ml/polished_model.pkl",
    "C:/Users/User/cyberprintproject/models/ml/multi_label_model.joblib",
    "C:/Users/User/cyberprintproject/models/ml/final_model.pkl",
    "models/ml/polished_model.joblib",
    "models/ml/multi_label_model.joblib",
    "../models/ml/polished_model.joblib",
]

CANDIDATE_VECTORIZER_PATHS = [
    "C:/Users/User/cyberprintproject/models/ml/polished_vectorizer.joblib",
    "C:/Users/User/cyberprintproject/models/ml/vectorizer.pkl",
    "C:/Users/User/cyberprintproject/models/ml/tfidf_vectorizer.joblib",
    "models/ml/polished_vectorizer.joblib",
    "models/ml/vectorizer.pkl",
    "../models/ml/polished_vectorizer.joblib",
]


def _find_existing_path(candidate_paths):
    """Return first valid path from candidates."""
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def _load_model_and_vectorizer():
    """Load model + vectorizer, preferring env vars, with sanity checks."""
    global _model, _vectorizer, _pos_index_list
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

    logger.debug('Loaded model=%s, vectorizer=%s', model_path, vec_path)
    return _model, _vectorizer


def _extract_positive_probs(probs, n_samples, n_labels):
    """Normalize predict_proba outputs into (n_samples, n_labels)."""
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
        return pos

    p = np.asarray(probs)
    if p.ndim == 2:
        if p.shape == (n_samples, n_labels):
            return p.astype(float)
        if p.shape == (n_labels, n_samples):
            return p.T.astype(float)
        if p.shape == (n_samples, 2 * n_labels):
            try:
                return p.reshape(n_samples, n_labels, 2)[:, :, 1].astype(float)
            except Exception:
                pass
        if p.shape[0] == n_samples and p.shape[1] >= n_labels:
            return p[:, :n_labels].astype(float)
    return pos


def predict_toxicity(texts):
    """Predict toxicity for a list of texts -> list of dicts with categories + emotions."""
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

    try:
        X = vectorizer.transform(texts)
    except Exception as e:
        logger.error('Vectorizer.transform failed: %s', e)
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
        category_probs = {label: round(float(pos[i, j]) * 100.0, 1) for j, label in enumerate(CATEGORY_LABELS)}
        result = {**category_probs}

        # emotions default to 0
        for category, emotions in CATEGORY_EMOTIONS.items():
            for e in emotions:
                result[e] = 0.0

        dominant_category = max(category_probs, key=category_probs.get, default="")
        result['dominant_category'] = dominant_category
        result['dominant_category_score'] = category_probs.get(dominant_category, 0.0)
        emotions_in_dominant = CATEGORY_EMOTIONS.get(dominant_category, [])
        if emotions_in_dominant:
            result['top_emotion'] = emotions_in_dominant[0]
            result['top_emotion_score'] = 0.0

        results.append(result)
    return results


def _zero_result():
    r = {label: 0.0 for label in CATEGORY_LABELS}
    for emotions in CATEGORY_EMOTIONS.values():
        for e in emotions:
            r[e] = 0.0
    r['dominant_category'] = ''
    r['dominant_category_score'] = 0.0
    return r
