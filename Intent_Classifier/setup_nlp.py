"""
setup_nlp.py
------------
One-time helper to download the optional NLP resources (spaCy model,
NLTK corpora) used for better preprocessing. Safe to skip: utils.py
automatically falls back to a simpler method if these aren't present.

Usage:
    python setup_nlp.py
"""

import subprocess
import sys


def try_spacy():
    try:
        import spacy  # noqa: F401
    except ImportError:
        print("spaCy not installed — skipping spaCy model download.")
        return
    print("Downloading spaCy model 'en_core_web_sm' ...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)


def try_nltk():
    try:
        import nltk
    except ImportError:
        print("NLTK not installed — skipping NLTK corpora download.")
        return
    print("Downloading NLTK corpora (stopwords, wordnet, omw-1.4) ...")
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg)


if __name__ == "__main__":
    try_spacy()
    try_nltk()
    print("\nSetup complete.")
