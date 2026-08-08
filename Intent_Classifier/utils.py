"""
utils.py
--------
Text preprocessing utilities for the Intent Classification System.

Preprocessing strategy (bonus feature: "Use spaCy for better preprocessing"):
    1. Try to use spaCy (lemmatization + stopword removal + tokenization).
    2. If spaCy or its "en_core_web_sm" model isn't installed, fall back to
       NLTK (stopword removal + WordNet lemmatization).
    3. If NLTK data isn't available either, fall back to a plain regex
       cleaner so the project still runs with zero extra downloads.

This keeps the project runnable out-of-the-box while rewarding a fuller
install with better text normalization.
"""

import re

# --------------------------------------------------------------------------
# Try to set up spaCy
# --------------------------------------------------------------------------
_SPACY_NLP = None
try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except OSError:
        _SPACY_NLP = None
except ImportError:
    _SPACY_NLP = None

# --------------------------------------------------------------------------
# Try to set up NLTK (used only if spaCy is unavailable)
# --------------------------------------------------------------------------
_NLTK_READY = False
if _SPACY_NLP is None:
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        try:
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _STOPWORDS = set(stopwords.words("english"))

        try:
            _LEMMATIZER = WordNetLemmatizer()
            _LEMMATIZER.lemmatize("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            _LEMMATIZER = WordNetLemmatizer()

        _NLTK_READY = True
    except ImportError:
        _NLTK_READY = False


def _basic_clean(text: str) -> str:
    """Lowercase and strip anything that isn't a letter or space."""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# NOTE on stopword removal: for short intent-classification queries, words
# like "is", "what", or "do" often carry real signal ("is it raining?" vs.
# "what is your name?"). Blanket stopword lists (spaCy's included) can even
# strip out topical words by accident. So by default we lemmatize but keep
# stopwords, and let TF-IDF's own weighting handle common-word dilution.
# Set remove_stopwords=True if you want the more aggressive behavior.
def preprocess(text: str, remove_stopwords: bool = False) -> str:
    """
    Clean and normalize a piece of text for feature extraction.

    Returns a space-joined string of normalized tokens (lowercased,
    lemmatized where possible, punctuation/numbers stripped).
    """
    cleaned = _basic_clean(text)
    if not cleaned:
        return cleaned

    # 1) Preferred path: spaCy
    if _SPACY_NLP is not None:
        doc = _SPACY_NLP(cleaned)
        tokens = [
            tok.lemma_ for tok in doc
            if tok.lemma_.strip() and (not remove_stopwords or not tok.is_stop)
        ]
        return " ".join(tokens) if tokens else cleaned

    # 2) Fallback: NLTK
    if _NLTK_READY:
        words = cleaned.split()
        if remove_stopwords:
            words = [w for w in words if w not in _STOPWORDS]
        tokens = [_LEMMATIZER.lemmatize(w) for w in words]
        return " ".join(tokens) if tokens else cleaned

    # 3) Last resort: regex-only cleaning (no external deps)
    return cleaned


def preprocessing_backend() -> str:
    """Report which preprocessing backend is active (useful for the README/demo)."""
    if _SPACY_NLP is not None:
        return "spaCy (en_core_web_sm)"
    if _NLTK_READY:
        return "NLTK (stopwords + WordNet lemmatizer)"
    return "regex-only fallback"


if __name__ == "__main__":
    print("Active preprocessing backend:", preprocessing_backend())
    samples = ["Hi there!!", "I WANT to order 2 Pizzas!!", "What's the Weather like today?"]
    for s in samples:
        print(f"{s!r} -> {preprocess(s)!r}")
