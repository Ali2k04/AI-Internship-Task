"""
model.py
--------
Core training / evaluation logic for the Intent Classification System.

Bonus features implemented here:
    - Train/test split
    - Multiple models trained & compared (Logistic Regression, Naive Bayes,
      Linear SVM) with accuracy + classification_report
    - The best-performing model is kept as the "production" model
    - Model + vectorizer persisted to disk with joblib
    - predict_intent() returns both the predicted label and a confidence
      score (predict_proba when available)
"""

import os
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from utils import preprocess, preprocessing_backend

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def load_dataset(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """Load the intent dataset and add a cleaned text column."""
    df = pd.read_csv(csv_path)
    df["clean_text"] = df["text"].apply(preprocess)
    return df


def get_candidate_models() -> dict:
    """Models to train & compare (bonus feature: multiple models)."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
        "linear_svm": SVC(kernel="linear", probability=True),
    }


def train_and_compare(csv_path: str = "dataset.csv", test_size: float = 0.25, random_state: int = 42):
    """
    Train multiple candidate models on a train/test split, evaluate each,
    and return everything needed to inspect or persist the best one.
    """
    df = load_dataset(csv_path)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["intent"]

    # Stratify keeps every intent represented in both splits when possible.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Falls back to a non-stratified split if a class has too few samples.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    results = {}
    fitted_models = {}

    for name, clf in get_candidate_models().items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)

        results[name] = {"accuracy": acc, "report": report}
        fitted_models[name] = clf

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(report)

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.3f})")
    print(f"Preprocessing backend used: {preprocessing_backend()}")

    return {
        "vectorizer": vectorizer,
        "models": fitted_models,
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
    }


def save_model(vectorizer, model, results: dict, best_name: str) -> None:
    """Persist the vectorizer + best model + metrics with joblib (bonus feature)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    metrics_summary = {
        name: {"accuracy": r["accuracy"]} for name, r in results.items()
    }
    metrics_summary["best_model"] = best_name
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved model      -> {MODEL_PATH}")
    print(f"Saved metrics     -> {METRICS_PATH}")


def load_model():
    """Load a previously-saved vectorizer + model from disk."""
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
        raise FileNotFoundError(
            "No saved model found. Run `python model.py` (or main.py's train step) first."
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def predict_intent(text: str, vectorizer, model):
    """
    Predict the intent of a piece of text.

    Returns:
        (predicted_label, confidence)
        confidence is a float in [0, 1] from predict_proba when the model
        supports it, otherwise None.
    """
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        confidence = float(proba[classes.index(label)])

    return label, confidence


if __name__ == "__main__":
    outcome = train_and_compare()
    save_model(
        outcome["vectorizer"],
        outcome["best_model"],
        outcome["results"],
        outcome["best_name"],
    )
