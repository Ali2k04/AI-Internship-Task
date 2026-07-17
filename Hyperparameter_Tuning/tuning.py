"""
tuning.py
----------
Hyperparameter tuning for an NLP sentiment classifier using
GridSearchCV and RandomizedSearchCV.

What this script does:
    1. Loads + preprocesses dataset.csv (same split as baseline_model.py)
    2. Builds a TF-IDF + Logistic Regression pipeline
    3. Tunes it with GridSearchCV      (exhaustive search)
    4. Tunes it with RandomizedSearchCV (efficient random search)
    5. BONUS: also tunes an SVM and a Naive Bayes pipeline, so we can
       compare across multiple model families, not just one
    6. Builds a full "before vs after" comparison table + bar chart
    7. Saves the single best performing model with joblib
    8. Runs an interactive prediction system using the tuned model

Run:
    python tuning.py
"""

import json
import time
import warnings

import joblib
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # silence sklearn's non-fatal FutureWarnings during grid search

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_dataset, clean_text

RANDOM_STATE = 42
DATASET_PATH = "dataset.csv"
CV_FOLDS = 5


def load_baseline_accuracy():
    """Read the accuracy recorded by baseline_model.py, if available."""
    try:
        with open("baseline_results.json") as f:
            data = json.load(f)
        return data["accuracy"], data["train_time_sec"]
    except FileNotFoundError:
        print("baseline_results.json not found - run baseline_model.py first.")
        return None, None


def main():
    results = []  # will hold dicts of {model, method, accuracy, train_time, best_params}

    # ---------------------------------------------------------
    # 1. Load + preprocess dataset (identical split to baseline_model.py
    #    so the comparison is apples-to-apples)
    # ---------------------------------------------------------
    df = load_dataset(DATASET_PATH)
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    baseline_accuracy, baseline_time = load_baseline_accuracy()
    if baseline_accuracy is not None:
        results.append({
            "model": "Logistic Regression",
            "method": "Before Tuning (default params)",
            "accuracy": baseline_accuracy,
            "train_time": baseline_time,
            "best_params": "-",
        })

    # ===========================================================
    # 2. GridSearchCV on Logistic Regression pipeline (exhaustive)
    # ===========================================================
    print("\n===== GridSearchCV: TF-IDF + Logistic Regression =====")

    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "tfidf__max_df": [0.85, 1.0],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1, 10],
        "clf__solver": ["liblinear", "lbfgs"],
    }

    grid_search = GridSearchCV(
        lr_pipeline, param_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start

    grid_best_model = grid_search.best_estimator_
    grid_test_accuracy = accuracy_score(y_test, grid_best_model.predict(X_test))

    print(f"Best CV score      : {grid_search.best_score_:.4f}")
    print(f"Best parameters    : {grid_search.best_params_}")
    print(f"Test accuracy      : {grid_test_accuracy:.4f}")
    print(f"Search time         : {grid_time:.2f} sec "
          f"({len(grid_search.cv_results_['params'])} combinations x {CV_FOLDS}-fold CV)")

    results.append({
        "model": "Logistic Regression",
        "method": "GridSearchCV",
        "accuracy": grid_test_accuracy,
        "train_time": grid_time,
        "best_params": grid_search.best_params_,
    })

    # ===========================================================
    # 3. RandomizedSearchCV on Logistic Regression pipeline (efficient)
    # ===========================================================
    print("\n===== RandomizedSearchCV: TF-IDF + Logistic Regression =====")

    param_dist = {
        "tfidf__max_df": [0.75, 0.85, 0.9, 1.0],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],  # liblinear supports both l1 and l2
    }

    random_search = RandomizedSearchCV(
        lr_pipeline, param_dist, n_iter=20, cv=CV_FOLDS,
        scoring="accuracy", n_jobs=-1, random_state=RANDOM_STATE,
    )

    start = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - start

    random_best_model = random_search.best_estimator_
    random_test_accuracy = accuracy_score(y_test, random_best_model.predict(X_test))

    print(f"Best CV score      : {random_search.best_score_:.4f}")
    print(f"Best parameters    : {random_search.best_params_}")
    print(f"Test accuracy      : {random_test_accuracy:.4f}")
    print(f"Search time         : {random_time:.2f} sec (20 sampled combinations x {CV_FOLDS}-fold CV)")

    results.append({
        "model": "Logistic Regression",
        "method": "RandomizedSearchCV",
        "accuracy": random_test_accuracy,
        "train_time": random_time,
        "best_params": random_search.best_params_,
    })

    # ===========================================================
    # 4. BONUS: tune two more model families for a broader comparison
    # ===========================================================
    print("\n===== BONUS: GridSearchCV on additional model families =====")

    # --- SVM ---
    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ])
    svm_param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.01, 0.1, 1, 10],
    }
    svm_search = GridSearchCV(svm_pipeline, svm_param_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
    start = time.time()
    svm_search.fit(X_train, y_train)
    svm_time = time.time() - start
    svm_best_model = svm_search.best_estimator_
    svm_test_accuracy = accuracy_score(y_test, svm_best_model.predict(X_test))
    print(f"[SVM] best params: {svm_search.best_params_} | test accuracy: {svm_test_accuracy:.4f}")

    results.append({
        "model": "SVM (LinearSVC)",
        "method": "GridSearchCV",
        "accuracy": svm_test_accuracy,
        "train_time": svm_time,
        "best_params": svm_search.best_params_,
    })

    # --- Naive Bayes ---
    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ])
    nb_param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.1, 0.5, 1.0, 2.0],
    }
    nb_search = GridSearchCV(nb_pipeline, nb_param_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
    start = time.time()
    nb_search.fit(X_train, y_train)
    nb_time = time.time() - start
    nb_best_model = nb_search.best_estimator_
    nb_test_accuracy = accuracy_score(y_test, nb_best_model.predict(X_test))
    print(f"[Naive Bayes] best params: {nb_search.best_params_} | test accuracy: {nb_test_accuracy:.4f}")

    results.append({
        "model": "Naive Bayes",
        "method": "GridSearchCV",
        "accuracy": nb_test_accuracy,
        "train_time": nb_time,
        "best_params": nb_search.best_params_,
    })

    # ===========================================================
    # 5. Build the final comparison table
    # ===========================================================
    comparison_df = pd.DataFrame(results)
    comparison_df["accuracy_pct"] = (comparison_df["accuracy"] * 100).round(2)
    comparison_df["train_time"] = comparison_df["train_time"].round(3)

    print("\n===== FINAL COMPARISON TABLE =====")
    print(comparison_df[["model", "method", "accuracy_pct", "train_time"]].to_string(index=False))

    comparison_df.to_csv("comparison_results.csv", index=False)
    print("\nSaved comparison_results.csv")

    # ===========================================================
    # 6. BONUS: bar chart of accuracy comparison
    # ===========================================================
    plt.figure(figsize=(10, 6))
    labels = [f"{r['model']}\n({r['method']})" for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    colors = ["#9e9e9e"] + ["#4c72b0"] * (len(results) - 1)
    bars = plt.bar(labels, accuracies, color=colors[:len(results)])
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Accuracy: Before vs After Hyperparameter Tuning")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.1f}%",
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=150)
    print("Saved comparison_plot.png")

    # ===========================================================
    # 7. Pick the single best model overall and save it with joblib
    # ===========================================================
    candidates = {
        "GridSearchCV (Logistic Regression)": (grid_test_accuracy, grid_best_model),
        "RandomizedSearchCV (Logistic Regression)": (random_test_accuracy, random_best_model),
        "GridSearchCV (SVM)": (svm_test_accuracy, svm_best_model),
        "GridSearchCV (Naive Bayes)": (nb_test_accuracy, nb_best_model),
    }
    best_name, (best_acc, best_model) = max(candidates.items(), key=lambda kv: kv[1][0])

    joblib.dump(best_model, "best_model.pkl")
    print(f"\nBest overall model: {best_name} (test accuracy: {best_acc:.4f})")
    print("Saved best_model.pkl")

    with open("best_model_info.json", "w") as f:
        json.dump({"name": best_name, "accuracy": best_acc}, f, indent=2)

    # ===========================================================
    # 8. Interactive prediction system using the tuned model
    # ===========================================================
    print("\n===== PREDICTION SYSTEM (tuned model) =====")
    print("Type a sentence to classify its sentiment, or press Enter to skip.")
    try:
        text = input("Enter text: ")
    except EOFError:
        text = ""

    if text.strip():
        cleaned = clean_text(text)
        prediction = best_model.predict([cleaned])[0]
        print(f"Prediction (Tuned Model - {best_name}): {prediction.capitalize()}")
    else:
        print("No input provided, skipping live prediction.")
        # demo prediction so the pipeline still shows expected output
        demo_text = "The acting was brilliant and I loved every minute of it."
        cleaned = clean_text(demo_text)
        prediction = best_model.predict([cleaned])[0]
        print(f"Demo input: \"{demo_text}\"")
        print(f"Prediction (Tuned Model - {best_name}): {prediction.capitalize()}")


if __name__ == "__main__":
    main()
