# generate_summary.py
import os
import re
import math
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- Domain mappings ----------
try:
    from cyberprint.label_mapping import LABELS_FLAT, FLAT_LABELS, CATEGORY_EMOTIONS
    LABELS = LABELS_FLAT  # expected: ['toxic','constructive','neutral','positive']
except Exception:
    # Safe fallback if label_mapping isn't available
    LABELS = ['toxic','constructive','neutral','positive']
    FLAT_LABELS = {k: [] for k in LABELS}
    CATEGORY_EMOTIONS = {k: [] for k in LABELS}

CSV_TO_LABELS_MAPPING = {l: l for l in LABELS}
ALL_EMOTIONS = [e for lst in FLAT_LABELS.values() for e in lst]

# ---------- PDF deps ----------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- TensorFlow 2.x (multi-label) ----------
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception:
    tf = None
    layers = None
    models = None
    EarlyStopping = None
    ReduceLROnPlateau = None
    TF_AVAILABLE = False

SEED = 42
random.seed(SEED); np.random.seed(SEED)
if TF_AVAILABLE:
    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass

# =========================================
# Text preprocessing / vectorization
# =========================================
_url_re = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
_md_img_re = re.compile(r'!\[[^\]]*\]\([^)]+\)')
_md_link_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
_emote_re = re.compile(r'\(emote\|[^)]+\)')
_user_re = re.compile(r'@[A-Za-z0-9_]+')
_hash_re = re.compile(r'#[A-Za-z0-9_]+')

def _standardize_text(s):
    # Keras TextVectorization expects a tf op when TF is available.
    # This implementation assumes TF is present when used; if TF is not available
    # the train_and_predict() path will not call this function (it raises earlier).
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, r'\r|\n', ' ')
    s = tf.strings.regex_replace(s, r'https?://\S+|www\.\S+', ' <url> ')
    s = tf.strings.regex_replace(s, r'!\[[^\]]*\]\([^)]+\)', ' ')
    s = tf.strings.regex_replace(s, r'\[([^\]]+)\]\(([^)]+)\)', r'\1')
    s = tf.strings.regex_replace(s, r'\(emote\|[^)]+\)', ' ')
    s = tf.strings.regex_replace(s, r'@[A-Za-z0-9_]+', ' <user> ')
    s = tf.strings.regex_replace(s, r'#[A-Za-z0-9_]+', ' <hashtag> ')
    # keep letters, numbers, basic punctuation
    s = tf.strings.regex_replace(s, r'[^a-z0-9\s\.,!\?:;()\']', ' ')
    s = tf.strings.regex_replace(s, r'\s+', ' ')
    return tf.strings.strip(s)

def build_vectorizer(texts, max_tokens=20000, seq_len=160):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow/TextVectorization is required to build the vectorizer. Install TensorFlow or provide precomputed features in the CSV.")
    vec = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=seq_len,
        standardize=_standardize_text
    )
    ds = tf.data.Dataset.from_tensor_slices(np.array(texts)).batch(256)
    vec.adapt(ds)
    return vec

# =========================================
# Data loading & target prep
# =========================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'comment_text' not in df.columns:
        raise ValueError("CSV must include a 'comment_text' column.")
    # Ensure label columns exist; fill missing with 0
    for lab in LABELS:
        if lab not in df.columns:
            df[lab] = 0.0
    # Clean label values -> [0,1]
    for lab in LABELS:
        col = pd.to_numeric(df[lab], errors='coerce').fillna(0.0)
        # If values look like percentages (max > 1.5), scale down
        if col.max() > 1.5:
            col = col.clip(lower=0.0, upper=100.0) / 100.0
        else:
            col = col.clip(lower=0.0, upper=1.0)
        df[lab] = col
    # Drop insane empties/dupes
    df['comment_text'] = df['comment_text'].astype(str)
    df = df.drop_duplicates(subset=['comment_text']).reset_index(drop=True)
    # Remove ultra-short (often noise)
    df = df[df['comment_text'].str.strip().str.len() >= 2].reset_index(drop=True)
    return df

def train_val_split(df: pd.DataFrame, val_frac=0.15):
    n = len(df)
    if n < 10:
        raise ValueError("Not enough rows to train; need at least 10.")
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    val_n = max(1, int(n * val_frac))
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

# =========================================
# Model: Embedding + GAP + Dense(sigmoid)
# =========================================
def build_model(vocab_size: int, seq_len: int, emb_dim=128, hidden=256, dropout=0.3, num_labels=4):
    inputs = layers.Input(shape=(seq_len,), dtype='int32')
    x = layers.Embedding(vocab_size, emb_dim)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hidden, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden//2, activation='relu')(x)
    outputs = layers.Dense(num_labels, activation='sigmoid')(x)  # multi-label
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(curve='ROC', multi_label=True, num_labels=num_labels, name='auc'),
            tf.keras.metrics.Precision(name='precision_at_0.5', thresholds=0.5),
            tf.keras.metrics.Recall(name='recall_at_0.5', thresholds=0.5),
        ]
    )
    return model

def f1_from_pr(precision, recall, eps=1e-8):
    return (2*precision*recall) / max(eps, (precision+recall))

# =========================================
# Train + Predict
# =========================================
def train_and_predict(df: pd.DataFrame):
    # Ensure TensorFlow is available before attempting to train
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available in this environment. Cannot train model. Provide precomputed label columns in the CSV or install TensorFlow.")

    # Prepare data
    texts = df['comment_text'].astype(str).tolist()
    y = df[LABELS].values.astype('float32')
    vectorizer = build_vectorizer(texts)
    vocab_size = vectorizer.vocabulary_size()
    seq_len = vectorizer.get_config()["output_sequence_length"]


    # Vectorize
    X_all = vectorizer(np.array(texts)).numpy()

    # Split
    idx = np.arange(len(texts))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    val_size = max(1, int(0.15 * len(texts)))
    val_idx, train_idx = idx[:val_size], idx[val_size:]
    X_train, y_train = X_all[train_idx], y[train_idx]
    X_val, y_val = X_all[val_idx], y[val_idx]

    # Build model
    model = build_model(vocab_size, seq_len, num_labels=len(LABELS))

    # Callbacks
    cb = [
        EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', mode='max', patience=2, factor=0.5, min_lr=1e-5, verbose=1),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=cb,
        verbose=0
    )

    # Evaluate (basic)
    eval_res = model.evaluate(X_val, y_val, verbose=0)
    metrics = dict(zip(model.metrics_names, eval_res))

    # Per-label simple precision/recall at 0.5
    y_val_pred = (model.predict(X_val, verbose=0) >= 0.5).astype(int)
    per_label_pr = {}
    for i, lab in enumerate(LABELS):
        tp = int(((y_val[:, i] == 1) & (y_val_pred[:, i] == 1)).sum())
        fp = int(((y_val[:, i] == 0) & (y_val_pred[:, i] == 1)).sum())
        fn = int(((y_val[:, i] == 1) & (y_val_pred[:, i] == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        per_label_pr[lab] = {
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "f1": round(float(f1_from_pr(prec, rec)), 3),
            "support_pos": int((y_val[:, i] == 1).sum()),
        }

    # Predict full dataset (probabilities 0..1)
    y_probs = model.predict(X_all, verbose=0)
    pred_df = pd.DataFrame(y_probs, columns=LABELS)

    diagnostics = {
        "val_metrics": {k: (round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()},
        "per_label": per_label_pr,
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
    }
    return pred_df, diagnostics, vectorizer, model

# =========================================
# PDF Generation
# =========================================
def _add_chart(story, doc_width, averages):
    labels_chart = [l for l in LABELS]
    values_chart = [float(averages.get(l, 0.0)) for l in labels_chart]
    color_map = {
        'toxic': '#e24a4a',
        'constructive': '#58c96b',
        'positive': '#7ed07e',
        'neutral': '#bfbfbf',
    }
    colors_list = [color_map.get(l, '#bfbfbf') for l in labels_chart]
    fig, ax = plt.subplots(figsize=(doc_width/72, 2.6))
    bars = ax.barh(labels_chart, values_chart, color=colors_list)
    ax.set_xlim(0, max(100, max(values_chart) if values_chart else 100))
    ax.set_xlabel('Average %')
    ax.set_title('Average Scores by Category')
    for i, b in enumerate(bars):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2, f'{values_chart[i]:.1f}%', va='center')
    plt.tight_layout()
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    img_buf.seek(0)
    return Image(img_buf, width=doc_width, height=doc_width*0.32)

def generate_pdf(
    profile_url,
    csv_path="cyberprint_profile_data.csv",
    output_path=r"C:\Users\User\cyberprintproject\cyberprint\output\cyberprint_report.pdf",
    relevance_threshold=0.15  # show labels >= 15%
):
    print(f"[PDF] Start for {profile_url}")
    if not os.path.exists(csv_path):
        print(f"[PDF] CSV not found: {csv_path}")
        return

    df = load_data(csv_path)
    if df.empty:
        print("[PDF] No usable rows after cleaning.")
        return

    # If the CSV already contains precomputed category columns, use them and skip training
    use_precomputed = all(lab in df.columns for lab in LABELS)
    diag = None

    if use_precomputed and df[LABELS].notnull().any().any():
        # We have label columns -- assume they are either in 0..1 or 0..100 scale
        max_val = float(df[LABELS].max().max())
        if max_val <= 1.5:
            # Convert 0..1 -> 0..100
            for lab in LABELS:
                df[lab] = df[lab].astype(float) * 100.0
        else:
            # Ensure numeric and clipped
            for lab in LABELS:
                df[lab] = df[lab].astype(float).clip(lower=0.0, upper=100.0)

        print("[PDF] Using precomputed category columns from CSV. Skipping model training.")

        # Build a minimal diagnostics structure so the PDF table renders sensibly
        diag = {
            'val_metrics': {'auc': 'n/a'},
            'per_label': {},
            'vocab_size': 0,
            'seq_len': 0,
            'train_size': 0,
            'val_size': 0,
        }
        for lab in LABELS:
            support = int((df[lab].astype(float) > 0).sum())
            diag['per_label'][lab] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support_pos': support,
            }
    else:
        # No precomputed labels: train on the CSV (fallback)
        print("[PDF] Training multi-label model...")
        preds_df, diag, vectorizer, model = train_and_predict(df)
        for lab in LABELS:
            df[lab] = (preds_df[lab].astype(float).clip(0, 1) * 100.0)

    # Averages
    averages = df[LABELS].mean()

    # ----- Build PDF -----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=80, bottomMargin=60)
    styles = getSampleStyleSheet()
    normal = styles['Normal']; normal.fontName = 'Helvetica'; normal.fontSize = 10
    heading = ParagraphStyle('Heading', parent=styles['Heading1'], fontName='Helvetica', alignment=1, fontSize=18, spaceAfter=8)
    subheading = ParagraphStyle('Sub', parent=styles['Heading2'], fontName='Helvetica', fontSize=12, spaceAfter=6)

    story = []
    story.append(Paragraph('CyberPrint Profile Analysis', heading))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f'Profile URL: <a href="{profile_url}">{profile_url}</a>', normal))
    story.append(Spacer(1, 6))

    # Overall tone line: show all non-zero averages
    tone_bits = [f"{lab}: {averages.get(lab,0):.1f}%" for lab in LABELS if averages.get(lab,0) > 0.0]
    story.append(Paragraph(f'<b>Overall Tone</b> – ' + " | ".join(tone_bits), normal))
    story.append(Spacer(1, 8))

    # Chart
    story.append(_add_chart(story, doc.width, averages))
    story.append(Spacer(1, 12))

    # Diagnostics block (helps you verify quality before trusting)
    diag_tbl_rows = [['Metric', 'Value']]
    diag_tbl_rows.append(['Validation AUC', f"{diag['val_metrics'].get('auc', 'n/a')}"])
    diag_tbl_rows.append(['Train size', str(diag.get('train_size', 0))])
    diag_tbl_rows.append(['Val size', str(diag.get('val_size', 0))])
    for lab in LABELS:
        d = diag['per_label'].get(lab, {})
        diag_tbl_rows.append([f"{lab} (P/R/F1)", f"{d.get('precision',0):.3f}/{d.get('recall',0):.3f}/{d.get('f1',0):.3f} (n={d.get('support_pos',0)})"])
    t = Table(diag_tbl_rows, colWidths=[doc.width*0.55, doc.width*0.35])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d2d2d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.4, colors.lightgrey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(Paragraph('Model Diagnostics', subheading))
    story.append(t)
    story.append(Spacer(1, 12))

    # Averages table
    avg_rows = [['Category','Average %']]
    for lab in LABELS:
        avg_rows.append([lab, f"{averages.get(lab,0):.1f}%"])
    tbl = Table(avg_rows, colWidths=[doc.width*0.6, doc.width*0.3])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4a4a4a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.4, colors.lightgrey),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # Comment Analysis
    story.append(Paragraph('Comment Analysis', subheading))
    max_display = 50
    for idx, row in df.iterrows():
        if idx >= max_display:
            break
        text = str(row.get('comment_text',''))
        text_disp = (text[:240] + '...') if len(text) > 240 else text
        story.append(Paragraph(f'<b>{idx+1}.</b> {text_disp}', normal))
        story.append(Spacer(1, 4))

        comment_rows = [['Category','Score']]
        # Only show labels over threshold
        for lab in LABELS:
            score = float(row.get(lab, 0.0))
            if score >= relevance_threshold * 100.0:
                comment_rows.append([lab, f"{score:.1f}%"])

        if len(comment_rows) > 1:
            c_tbl = Table(comment_rows, colWidths=[doc.width*0.7, doc.width*0.2])
            c_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4a4a4a')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
                ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
            ]))
            story.append(c_tbl)
            story.append(Spacer(1, 8))
        else:
            story.append(Spacer(1, 4))

    # Build
    try:
        doc.build(story)
        print(f"[PDF] Saved to {output_path}")
    except Exception as e:
        print("[PDF] Build failed:", e)

# Entry (optional CLI usage)
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_url", type=str, required=True)
    ap.add_argument("--csv_path", type=str, default="cyberprint_profile_data.csv")
    ap.add_argument("--output_path", type=str, default=r"C:\Users\User\cyberprintproject\cyberprint\output\cyberprint_report.pdf")
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()
    generate_pdf(args.profile_url, args.csv_path, args.output_path, relevance_threshold=args.threshold)
