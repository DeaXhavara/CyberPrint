# data/preprocess.py

import re
import string
import emoji

def clean_text(text):
    """
    Cleans input text for ML predictions:
    - Lowercase
    - Remove URLs
    - Remove mentions (@username)
    - Remove hashtags (#tag)
    - Remove emojis
    - Strip punctuation
    - Remove extra spaces
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
