"""
utils.py
--------
Text preprocessing and Named Entity Recognition (NER) utilities.

Primary path uses spaCy (bonus feature). If the spaCy model isn't
installed, the module falls back to NLTK, and finally to a pure-Python
regex tokenizer so the chatbot never crashes for lack of a model.

Run once before first use (bonus feature setup):
    pip install spacy nltk
    python -m spacy download en_core_web_sm
"""

import re
import string

# ---------------------------------------------------------------------
# Backend selection: spaCy > NLTK > plain regex
# ---------------------------------------------------------------------
_BACKEND = None
_nlp = None
_lemmatizer = None

try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
        _BACKEND = "spacy"
    except OSError:
        # Model not downloaded yet
        _BACKEND = None
except ImportError:
    _BACKEND = None

if _BACKEND is None:
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer

        # Make sure required corpora exist; download quietly if missing.
        for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4",
                    "averaged_perceptron_tagger", "maxent_ne_chunker", "words"):
            try:
                nltk.data.find(pkg)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

        _lemmatizer = WordNetLemmatizer()
        _BACKEND = "nltk"
    except ImportError:
        _BACKEND = "regex"


def get_backend():
    """Return which NLP backend is active: 'spacy', 'nltk', or 'regex'."""
    return _BACKEND


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def preprocess(text: str) -> str:
    """
    Lowercase, tokenize, strip punctuation, and lemmatize the input text.
    Used to normalize both training examples and live user input before
    they're fed into the TF-IDF vectorizer.
    """
    text = text.strip()
    if not text:
        return ""

    if _BACKEND == "spacy":
        doc = _nlp(text.lower())
        tokens = [t.lemma_ for t in doc if not t.is_punct and not t.is_space]
        return " ".join(tokens)

    if _BACKEND == "nltk":
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    # regex fallback: lowercase + strip punctuation + naive split
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


# ---------------------------------------------------------------------
# Named Entity Recognition (bonus feature)
# ---------------------------------------------------------------------
_KNOWN_CITIES = {
    "karachi", "lahore", "islamabad", "multan", "faisalabad", "peshawar",
    "quetta", "rawalpindi", "london", "new york", "paris", "dubai",
    "tokyo", "delhi", "mumbai", "beijing", "moscow", "cairo", "toronto",
}


def extract_entities(text: str):
    """
    Extract named entities from text. With spaCy, returns real NER
    labels (GPE, PERSON, DATE, ORG, ...). Without spaCy, falls back to
    a small gazetteer of well-known city names so the "weather" intent
    can still pick a city out of the sentence.

    Returns a list of dicts: [{"text": ..., "label": ...}, ...]
    """
    if _BACKEND == "spacy":
        doc = _nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    # Fallback gazetteer match (case-insensitive) for city names.
    found = []
    lowered = text.lower()
    for city in _KNOWN_CITIES:
        if city in lowered:
            found.append({"text": city.title(), "label": "GPE"})
    return found


def extract_city(text: str):
    """Convenience helper: return the first GPE/city entity found, or None."""
    for ent in extract_entities(text):
        if ent["label"] in ("GPE", "LOC"):
            return ent["text"]
    return None
