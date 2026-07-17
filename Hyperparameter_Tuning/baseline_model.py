"""
baseline_model.py
-------------------
Trains a BASIC (untuned) Logistic Regression sentiment classifier so we
have a fair "before tuning" number to compare against tuning.py later.

Pipeline: TfidfVectorizer -> LogisticRegression (all default hyperparameters)

Run:
    python baseline_model.py
"""

import json
import time
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_dataset

RANDOM_STATE = 42
DATASET_PATH = "dataset.csv"


def main():
    # ---------------------------------------------------------
    # 1. Load + preprocess dataset
    # ---------------------------------------------------------
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} samples")
    print(df["label"].value_counts(), "\n")

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ---------------------------------------------------------
    # 2. Baseline pipeline: TF-IDF + Logistic Regression (default params)
    # ---------------------------------------------------------
    baseline_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    start = time.time()
    baseline_pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = baseline_pipeline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)

    print("===== BASELINE MODEL (untuned Logistic Regression) =====")
    print(f"Training time     : {train_time:.4f} sec")
    print(f"Baseline accuracy : {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Save the baseline accuracy + timing so tuning.py can build the
    # final "before vs after" comparison table without retraining.
    with open("baseline_results.json", "w") as f:
        json.dump({
            "accuracy": baseline_accuracy,
            "train_time_sec": train_time,
        }, f, indent=2)

    print("Saved baseline_results.json")


if __name__ == "__main__":
    main()
