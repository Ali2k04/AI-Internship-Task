"""
train_models.py
----------------
Run this once before launching the dashboard:

    python train_models.py

It will:
  1. Load dataset.csv (or generate a synthetic spam-detection dataset if
     dataset.csv doesn't exist yet)
  2. Train Logistic Regression, SVM, and Naive Bayes
  3. Evaluate them with Accuracy / Precision / Recall / F1-score
  4. Save everything under models/ so the Streamlit app loads instantly
     without retraining every time it starts
"""

from pathlib import Path
import pandas as pd

from utils import generate_spam_dataset, train_and_evaluate, save_artifacts

DATA_PATH = Path(__file__).parent / "dataset.csv"


def main():
    if DATA_PATH.exists():
        print(f"Loading existing dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("No dataset.csv found -- generating a synthetic spam-detection dataset")
        df = generate_spam_dataset(n_samples=400)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to {DATA_PATH} ({len(df)} rows)")

    print("\nTraining models: Logistic Regression, SVM (linear), Naive Bayes ...")
    metrics_df, vectorizer, artifacts, pos_label = train_and_evaluate(df)
    save_artifacts(metrics_df, vectorizer, artifacts, pos_label)

    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    print(f"\nPositive class used for metrics: '{pos_label}'")
    print("\nArtifacts saved under models/. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
