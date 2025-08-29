from cyberprint.label_mapping import FLAT_LABELS, LABELS_FLAT, CATEGORY_EMOTIONS

def remap_goemotions_labels(comment_labels):
    """
    Returns a dict with both flat category percentages and individual emotion percentages.
    """
    flat_result = {label: 0.0 for label in LABELS_FLAT}
    fine_result = {}  # new: keep each emotion separately

    for flat_label, emotions in FLAT_LABELS.items():
        count = sum(1 for e in comment_labels if e in emotions)
        flat_result[flat_label] = count
        for e in emotions:
            fine_result[e] = 1.0 if e in comment_labels else 0.0

    # normalize flat percentages
    total = sum(flat_result.values())
    if total > 0:
        for k in flat_result:
            flat_result[k] = (flat_result[k] / total) * 100

    return {**flat_result, **fine_result}

def process_predictions(predictions):
    """
    Process model predictions to generate detailed reports with category percentages
    and top emotion breakdowns.

    Accepts predictions where category values may be either in 0..1 or 0..100. The
    function normalizes internal representation to 0..1 and returns category values
    as percentages (0..100).
    """
    results = []

    for pred in predictions:
        # Detect scale of incoming category values (0..1 or 0..100)
        # Look across known category keys
        raw_vals = [float(pred.get(cat, 0.0)) for cat in LABELS_FLAT]
        max_raw = max(raw_vals) if raw_vals else 0.0
        is_percent_input = max_raw > 1.5  # if >1.5 assume input is already 0..100

        # Convert to 0..1 internal scores
        category_scores = {}
        for cat in LABELS_FLAT:
            raw = float(pred.get(cat, 0.0))
            if is_percent_input:
                score = max(0.0, min(raw / 100.0, 1.0))
            else:
                score = max(0.0, min(raw, 1.0))
            category_scores[cat] = score

        # Convert to percent for output (0..100) but keep independence (no renormalization)
        normalized_categories = {cat: round(score * 100.0, 1) for cat, score in category_scores.items()}

        # Find dominant category by raw score (highest independent probability)
        dominant_category = max(category_scores, key=category_scores.get)
        dominant_score = round(category_scores[dominant_category] * 100.0, 1)

        # Get top emotions in the dominant category
        emotions_in_dominant = CATEGORY_EMOTIONS.get(dominant_category, [])
        emotion_scores = {}
        total_emotion_score = 0.0
        for emotion in emotions_in_dominant:
            # Emotion values in `pred` may also be in 0..1 or 0..100; normalize accordingly
            raw_em = float(pred.get(emotion, 0.0))
            if is_percent_input:
                em_score = max(0.0, min(raw_em / 100.0, 1.0))
            else:
                em_score = max(0.0, min(raw_em, 1.0))
            emotion_scores[emotion] = em_score
            total_emotion_score += em_score

        # Normalize emotion percentages within the dominant category for a readable breakdown
        normalized_emotions = {}
        if total_emotion_score > 0:
            for emotion, score in emotion_scores.items():
                # show relative share of the emotion within the dominant category
                normalized_emotions[emotion] = round((score / total_emotion_score) * 100.0, 1)
        else:
            even_split = 100.0 / len(emotions_in_dominant) if emotions_in_dominant else 0
            normalized_emotions = {emotion: even_split for emotion in emotions_in_dominant}

        # Get top 3 emotions in the dominant category
        sorted_emotions = sorted(normalized_emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:3]

        # Format the result
        result = {
            'categories': normalized_categories,  # percents per-category (independent)
            'dominant_category': dominant_category,
            'dominant_category_score': dominant_score,
            'top_emotions': top_emotions,  # List of (emotion, percentage) tuples (relative within category)
            'overall_tone': 'Positive' if normalized_categories.get('positive', 0) > normalized_categories.get('toxic', 0) else 'Negative'
        }

        results.append(result)

    return results
