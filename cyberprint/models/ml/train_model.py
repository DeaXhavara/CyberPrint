import os
import re
import emoji
import joblib
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from datasets import load_dataset
from cyberprint.label_mapping import JIGSAW_TO_FLAT_MAPPING
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

JIGSAW_PATH = os.path.join(
    PROJECT_ROOT, "cyberprint", "data", "raw", "datasets",
    "jigsaw", "jigsaw-toxic-comment-classification-challenge", "train.csv"
)

ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "cyberprint", "models", "ml")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "polished_vectorizer.joblib")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "polished_model.joblib")

# ---------------------------------------------------------
# Clean function
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="ignore")
    text = str(text).lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'s\b", " is", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------
# GoEmotions category order (used by TFDS 'goemotions')
# ---------------------------------------------------------
GOEMOTIONS_CATEGORIES = [
    'admiration','amusement','anger','annoyance','approval',
    'caring','confusion','curiosity','desire','disappointment',
    'disapproval','disgust','embarrassment','excitement','fear',
    'gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]

# ---------------------------------------------------------
# Load GoEmotions (robust)
# ---------------------------------------------------------
def load_goemotions():
    ds = tfds.load("goemotions", split="train")

    rows = []
    for ex in tfds.as_numpy(ds):
        # --- Text ---
        if b"comment_text" in ex or "comment_text" in ex:
            txt = ex.get(b"comment_text", ex.get("comment_text"))
        elif b"text" in ex or "text" in ex:
            txt = ex.get(b"text", ex.get("text"))
        else:
            raise KeyError(f"No text field found in example: {list(ex.keys())}")

        if isinstance(txt, (bytes, bytearray)):
            txt = txt.decode("utf-8", errors="ignore")

        # --- Labels ---
        labels_idx = []
        # Case A: TFDS provides a 'labels' field (array of indices)
        if b"labels" in ex or "labels" in ex:
            labs = ex.get(b"labels", ex.get("labels"))
            # labs may be an ndarray of ints
            try:
                for v in labs:
                    labels_idx.append(int(v))
            except Exception:
                # scalar label
                labels_idx = [int(labs)]
        else:
            # Case B: TFDS exposes per-emotion boolean fields
            for i, emo in enumerate(GOEMOTIONS_CATEGORIES):
                # try bytes and str keys
                val = None
                if emo in ex:
                    val = ex[emo]
                elif emo.encode("utf-8") in ex:
                    val = ex[emo.encode("utf-8")]
                if val is None:
                    continue
                try:
                    if int(val) != 0:
                        labels_idx.append(i)
                except Exception:
                    # ignore non-numeric
                    continue

        rows.append({"text": txt, "labels": labels_idx})

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Load Emotion
# ---------------------------------------------------------
def load_emotion():
    ds = load_dataset("emotion", split="train")
    rows = []
    for ex in ds:
        txt = clean_text(ex["text"])
        lbl = int(ex["label"])
        rows.append({"text": txt, "labels": [lbl]})
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
if not os.path.exists(JIGSAW_PATH):
    raise FileNotFoundError(f"Jigsaw dataset not found at {JIGSAW_PATH}")

df_jig = pd.read_csv(JIGSAW_PATH)
df_go = load_goemotions()
df_emo = load_emotion()

df_jig["comment_text"] = df_jig["comment_text"].apply(clean_text)
df_go["text"] = df_go["text"].apply(clean_text)
df_emo["text"] = df_emo["text"].apply(clean_text)

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------
from cyberprint.label_mapping import JIGSAW_TO_FLAT_MAPPING

FINAL_LABELS = ["harmful", "critical", "supportive", "neutral"]

CATEGORIES = FINAL_LABELS
cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

# Emotion dataset labels (index to string)
Emotion_labels = [
    "anger","disgust","fear","joy","neutral","sadness","surprise"
]

# Jigsaw Y (map via JIGSAW_TO_FLAT_MAPPING and filter to FINAL_LABELS)
y_jig = np.zeros((len(df_jig), len(CATEGORIES)), dtype=np.int8)
for orig, flat in JIGSAW_TO_FLAT_MAPPING.items():
    if orig in df_jig.columns and flat in CATEGORIES:
        idx = cat_to_idx[flat]
        y_jig[:, idx] = np.maximum(y_jig[:, idx], df_jig[orig].astype(int).values)
X_jig = df_jig["comment_text"].values

# GoEmotions Y (map only neutral → neutral, else ignore for now)
y_go = np.zeros((len(df_go), len(CATEGORIES)), dtype=np.int8)
for i, row in df_go.iterrows():
    for lbl in row["labels"]:
        if 0 <= int(lbl) < len(GOEMOTIONS_CATEGORIES):
            cat = GOEMOTIONS_CATEGORIES[int(lbl)]
            if cat == "neutral":
                y_go[i, cat_to_idx["neutral"]] = 1
X_go = df_go["text"].values

# Emotion Y (map neutral, joy→supportive, anger/disgust/fear→harmful, sadness→critical, surprise→neutral)
EMO_MAP = {
    "neutral": "neutral",
    "joy": "supportive",
    "anger": "harmful",
    "disgust": "harmful",
    "fear": "harmful",
    "sadness": "critical",
    "surprise": "neutral"
}
y_emo = np.zeros((len(df_emo), len(CATEGORIES)), dtype=np.int8)
for i, row in df_emo.iterrows():
    for lbl in row["labels"]:
        idx_label = int(lbl)
        if 0 <= idx_label < len(Emotion_labels):
            emo_cat = Emotion_labels[idx_label]
            if emo_cat in EMO_MAP:
                mapped = EMO_MAP[emo_cat]
                if mapped in CATEGORIES:
                    y_emo[i, cat_to_idx[mapped]] = 1
X_emo = df_emo["text"].values

# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------
X_all = np.concatenate([X_jig, X_go, X_emo], axis=0)
Y_all = np.vstack([y_jig, y_go, y_emo])

# ---------------------------------------------------------
# Split + vectorize
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, Y_all, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
base_lr = LogisticRegression(max_iter=300, solver="liblinear", class_weight="balanced")
model = MultiOutputClassifier(base_lr, n_jobs=-1)
model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

for i, cat in enumerate(CATEGORIES):
    y_true_i = y_test[:, i]
    if y_true_i.sum() == 0 and y_pred[:, i].sum() == 0:
        continue
    print(f"\n=== Report for {cat} ===")
    print(classification_report(y_true_i, y_pred[:, i], zero_division=0))

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(model, MODEL_PATH)

print(f"\n✅ Model and vectorizer saved successfully!")
print(f"   Vectorizer → {VECTORIZER_PATH}")
print(f"   Model      → {MODEL_PATH}")
