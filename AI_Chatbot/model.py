"""
model.py
--------
Intent classification.

Two interchangeable engines:
  1. "tfidf"      - TF-IDF + Logistic Regression (fast, always available,
                     trained on intents.json at startup).
  2. "transformer" - Zero-shot classification using a pretrained Hugging
                     Face model (bonus feature). Falls back to "tfidf"
                     automatically if `transformers` isn't installed or
                     no internet access is available to fetch weights.

Usage:
    clf = IntentClassifier(engine="tfidf")
    clf.fit(intents_dict)
    label, confidence = clf.predict("hi there")
"""

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import preprocess

MODEL_PATH = Path(__file__).parent / "trained_model.pkl"


class IntentClassifier:
    def __init__(self, engine: str = "tfidf"):
        self.engine = engine
        self.vectorizer = None
        self.clf = None
        self.labels_ = []
        self._zero_shot = None  # lazily-loaded transformers pipeline

        if engine == "transformer":
            self._init_transformer()

    # ------------------------------------------------------------------
    # Transformer (bonus) engine
    # ------------------------------------------------------------------
    def _init_transformer(self):
        try:
            from transformers import pipeline
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )
        except Exception as e:
            print(f"[model.py] Could not load transformer pipeline ({e}). "
                  f"Falling back to TF-IDF engine.")
            self.engine = "tfidf"

    # ------------------------------------------------------------------
    # TF-IDF + Logistic Regression engine
    # ------------------------------------------------------------------
    def fit(self, intents: dict):
        """Train the TF-IDF classifier on the intents dataset."""
        self.labels_ = list(intents.keys())

        sentences, labels = [], []
        for intent, payload in intents.items():
            examples = payload["examples"] if isinstance(payload, dict) else payload
            for ex in examples:
                sentences.append(preprocess(ex))
                labels.append(intent)

        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(sentences)

        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, labels)
        return self

    def predict(self, text: str):
        """
        Predict the intent of `text`.
        Returns (label: str, confidence: float in [0, 1]).
        """
        if self.engine == "transformer" and self._zero_shot is not None:
            result = self._zero_shot(text, candidate_labels=self.labels_)
            return result["labels"][0], float(result["scores"][0])

        if self.clf is None:
            raise RuntimeError("Classifier not trained. Call .fit(intents) first.")

        processed = preprocess(text)
        vec = self.vectorizer.transform([processed])
        label = self.clf.predict(vec)[0]
        proba = self.clf.predict_proba(vec)[0]
        confidence = float(max(proba))
        return label, confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "clf": self.clf, "labels": self.labels_},
                f,
            )

    def load(self, path: Path = MODEL_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.clf = data["clf"]
        self.labels_ = data["labels"]
        return self


def load_intents(path: str = "intents.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_default_model(engine: str = "tfidf") -> IntentClassifier:
    """Convenience function used by chatbot.py / app.py at startup."""
    intents = load_intents(str(Path(__file__).parent / "intents.json"))
    clf = IntentClassifier(engine=engine)
    if clf.engine == "tfidf":
        clf.fit(intents)
    return clf


if __name__ == "__main__":
    # Quick sanity check when run directly: `python model.py`
    clf = train_default_model()
    for text in ["hi there", "bye for now", "what's your name", "weather in Lahore"]:
        label, conf = clf.predict(text)
        print(f"{text!r:35} -> {label:12} (confidence={conf:.2f})")
