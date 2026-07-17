"""
preprocessing.py
-----------------
Shared text preprocessing utilities for the sentiment analysis pipeline.

Steps performed on every piece of text:
    1. Lowercasing
    2. Punctuation / digit removal
    3. Tokenization
    4. Stopword removal
    5. Lemmatization

Both baseline_model.py and tuning.py import `clean_text` and `load_dataset`
from this module so the exact same preprocessing is used everywhere
(this matters for a fair "before vs after tuning" comparison).
"""

import re
import string
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk_data():
    """
    Make sure the NLTK corpora we need are available.
    Tries to download them once; if there is no internet access,
    falls back to a small built-in stopword list so the pipeline
    still runs end-to-end.
    """
    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # handled by fallback stopword list below


_ensure_nltk_data()

# Fallback stopword list (used only if NLTK stopwords corpus is unavailable)
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "in", "on", "at", "to", "for", "of", "with",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
    "we", "they", "them", "his", "her", "their", "as", "so", "than", "then",
    "not", "no", "do", "does", "did", "have", "has", "had", "just", "very",
}

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = _FALLBACK_STOPWORDS

try:
    _lemmatizer = WordNetLemmatizer()
    _lemmatizer.lemmatize("test")  # trigger a lookup to check wordnet is present
    LEMMATIZER_AVAILABLE = True
except LookupError:
    LEMMATIZER_AVAILABLE = False


def clean_text(text: str) -> str:
    """
    Apply the full preprocessing pipeline to a single string and
    return the cleaned, space-joined string (ready for TfidfVectorizer).
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Tokenization
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()

    # 4. Stopword removal
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # 5. Lemmatization
    if LEMMATIZER_AVAILABLE:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset.csv (columns: text, label), drop empty rows,
    and add a `clean_text` column with fully preprocessed text.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


if __name__ == "__main__":
    # quick manual test
    sample = "The Acting was ABSOLUTELY brilliant!!! 10/10, would watch again."
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))
