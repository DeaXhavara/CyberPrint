# main.py
import os
import pandas as pd
from generate_summary import generate_pdf
from cyberprint.models.ml.predict_toxicity import predict_toxicity
from cyberprint.pipeline import process_predictions
from cyberprint.data.scrapers.youtube_fetcher import fetch_youtube_comments
from cyberprint.data.scrapers.reddit_fetcher import fetch_reddit_comments_by_user
from cyberprint.data.scrapers.hackernews_fetcher import fetch_hackernews_comments_by_user
from cyberprint.label_mapping import LABELS_FLAT, CATEGORY_EMOTIONS

def detect_platform_and_id(url):
    if "reddit.com/user/" in url:
        return "reddit", url.split("reddit.com/user/")[-1].strip("/")
    elif "news.ycombinator.com/user?id=" in url:
        return "hackernews", url.split("id=")[-1]
    elif "youtube.com" in url:
        return "youtube", url
    else:
        return None, None

def extract_comment_text(comment):
    if isinstance(comment, dict) and "text" in comment:
        return comment["text"]
    elif isinstance(comment, str):
        return comment
    return ""

def analyze_profile(profile_url, limit=50):
    platform, identifier = detect_platform_and_id(profile_url)
    if not platform:
        print("[Error] Unsupported platform")
        return []

    comments = []
    try:
        if platform == "reddit":
            comments = fetch_reddit_comments_by_user(identifier, limit=limit)
        elif platform == "hackernews":
            comments = fetch_hackernews_comments_by_user(identifier, limit=limit)
        elif platform == "youtube":
            comments = fetch_youtube_comments(identifier, max_comments=limit)
    except Exception as e:
        print(f"[Fetcher Error] {e}")
        return []

    if not comments:
        print("No comments/posts found for this profile.")
        return []

    comment_texts = [extract_comment_text(c) for c in comments if extract_comment_text(c).strip()]
    if not comment_texts:
        print("No valid text found in fetched comments.")
        return []

    # Predict using pre-trained inference (do NOT train here)
    raw_predictions = predict_toxicity(comment_texts)

    # Ensure raw_predictions are normalized into a consistent format (0..1 floats)
    normalized_raw_preds = []
    for pred in raw_predictions:
        norm = {}
        # Determine if incoming preds are 0..100 percentages
        max_val = max([float(pred.get(k, 0.0)) for k in LABELS_FLAT]) if LABELS_FLAT else 0.0
        is_percent = max_val > 1.5
        for k, v in pred.items():
            try:
                fv = float(v)
            except:
                norm[k] = v
                continue
            if is_percent:
                norm[k] = max(0.0, min(fv / 100.0, 1.0))
            else:
                norm[k] = max(0.0, min(fv, 1.0))
        normalized_raw_preds.append(norm)

    processed_predictions = process_predictions(normalized_raw_preds)
    
    combined = []
    for idx, (comment, raw_pred, processed) in enumerate(zip(comment_texts, raw_predictions, processed_predictions)):
        # Create entry with comment text
        entry = {"comment_text": comment}
        
        # Add category percentages (use processed which are percent display values)
        categories = processed.get('categories', {})
        for cat, score in categories.items():
            entry[cat] = score
            
        # Add dominant category and score
        entry['dominant_category'] = processed.get('dominant_category', '')
        entry['dominant_category_score'] = processed.get('dominant_category_score', 0.0)
        
        # Add top emotions
        top_emotions = processed.get('top_emotions', [])
        for i, (emotion, score) in enumerate(top_emotions):
            entry[f'top_emotion_{i+1}'] = emotion
            entry[f'top_emotion_{i+1}_score'] = score
            
        # Add overall tone
        entry['overall_tone'] = processed.get('overall_tone', 'Neutral')
        
        # Add individual numeric scores using the normalized predictions (0..1 -> percent)
        norm_pred = normalized_raw_preds[idx] if idx < len(normalized_raw_preds) else {}
        for k, v in norm_pred.items():
            if k in entry:
                continue
            # numeric -> convert 0..1 to percent
            try:
                fv = float(v)
                entry[k] = round(max(0.0, min(fv, 1.0)) * 100.0, 1)
            except Exception:
                # non-numeric values (strings) -> keep as-is
                entry[k] = v
                
        combined.append(entry)
    return combined

def main():
    profile_url = input("Enter profile URL: ").strip()
    try:
        limit = int(input("Enter number of comments/posts to fetch (default 50): ").strip())
    except:
        limit = 50

    print(f"\nFetching data for {profile_url} ...\n")
    data = analyze_profile(profile_url, limit=limit)
    if not data:
        print("❌ No data fetched or unsupported platform.")
        return

    # Save CSV
    df = pd.DataFrame(data)
    csv_path = "cyberprint_profile_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Fetched {len(data)} items. Saved to {csv_path}\n")

    # Compute overall tone using mean of independent probabilities
    df_numeric = pd.DataFrame(data)
    category_averages = {}
    for category in LABELS_FLAT:
        if category in df_numeric.columns:
            # Values in CSV may be percents (0..100). Normalize back to 0..1 for averaging.
            vals = df_numeric[category].astype(float)
            if vals.max() > 1.5:
                vals = vals / 100.0
            category_averages[category] = vals.mean() * 100.0
        else:
            category_averages[category] = 0.0

    positive_categories = ['positive', 'constructive']
    negative_categories = ['toxic']

    avg_positive = sum(category_averages.get(cat, 0) for cat in positive_categories) / len(positive_categories) if positive_categories else 0
    avg_negative = sum(category_averages.get(cat, 0) for cat in negative_categories) / len(negative_categories) if negative_categories else 0

    overall_tone = "Positive" if avg_positive > avg_negative else "Negative"
    print(f"Overall Tone: {overall_tone} (avg positive: {avg_positive:.2f}%, avg toxic: {avg_negative:.2f}%)\n")

    # Generate PDF report
    print("Generating PDF report...")
    generate_pdf(profile_url=profile_url)
    print("✅ PDF generated successfully!")

if __name__ == "__main__":
    main()
