"""
utils.py
--------
Helper functions for the Model Evaluation Dashboard.

Responsibilities:
  1. Synthetic dataset generation (used if no dataset.csv is supplied)
  2. A shared train/evaluate pipeline used by BOTH train_models.py
     (offline training) and app.py (when a user uploads their own CSV)
  3. Persisting / loading trained models + metrics
  4. Confusion matrix & ROC curve helpers
"""

import random
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Synthetic dataset generation
# ---------------------------------------------------------------------------
def generate_spam_dataset(n_samples: int = 400, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Builds a synthetic SMS/email spam-detection dataset by mixing spam / ham
    sentence templates with randomised placeholders. This keeps the project
    fully self-contained (no external download required) while still giving
    a realistic binary text-classification problem.

    Swap this out for a real dataset (sentiment analysis, fake news, etc.)
    by simply replacing dataset.csv with your own text,label file.
    """
    rng = random.Random(seed)

    spam_templates = [
        "Congratulations! You have won a ${amount} gift card, click here to claim now",
        "URGENT: Your account has been suspended, verify immediately at the link below",
        "Free entry to win a brand new iPhone, text WIN to {number}",
        "Limited time offer! Buy now and get {amount}% discount on all items",
        "You have been selected for a cash prize of ${amount}, claim your reward today",
        "Your loan of ${amount} has been approved, reply now to receive funds",
        "Click this link to activate your free {amount}-day trial before it expires",
        "WINNER!! As a valued customer you have been selected to receive a prize",
        "Get rich quick! Invest ${amount} today and double your money in a week",
        "Your package could not be delivered, pay a ${amount} fee to reschedule",
        "Hot singles in your area want to chat with you right now, click here",
        "Claim your free {amount}GB data bundle now, offer ends today",
        "Act now! Your ${amount} refund is waiting, click to confirm your details",
        "Congratulations, your number has won {amount} in our monthly lottery draw",
    ]

    ham_templates = [
        "Hey, are we still meeting for lunch tomorrow at {number}?",
        "Can you send me the report before end of day?",
        "Happy birthday! Hope you have a great day.",
        "Reminder: team meeting at {number}am in conference room B",
        "Thanks for your help yesterday, really appreciate it",
        "Let's catch up this weekend, it's been a while",
        "Please review the attached document and share your feedback",
        "Running {number} minutes late, see you soon",
        "Can you pick up milk on your way home?",
        "The flight is scheduled to depart at {number} pm, don't be late",
        "Great job on the presentation today, the client loved it",
        "Don't forget to submit your assignment by {number}pm",
        "Mom said dinner is ready, come downstairs",
        "I've scheduled our call for {number} o'clock tomorrow",
    ]

    # A handful of deliberately ambiguous messages that blur the line between
    # spam and ham -- keeps the classification problem realistic instead of
    # trivially separable, so models don't all score a suspicious 100%.
    ambiguous_templates = [
        ("Reminder: your subscription payment of ${amount} is due tomorrow", "ham"),
        ("Your order confirmation: package arriving in {number} days", "ham"),
        ("Special offer for loyal customers: {amount}% off your next visit", "spam"),
        ("Your verification code is {number}{number}{number}{number}", "ham"),
        ("Don't miss out, only {number} spots left for the free webinar", "spam"),
        ("Your bank statement for this month is now available online", "ham"),
        ("Win big this weekend, deposit ${amount} and get a bonus", "spam"),
        ("Meeting moved to {number}pm, updated invite attached", "ham"),
    ]

    def fill(template: str) -> str:
        return template.format(
            amount=rng.choice([50, 100, 250, 500, 1000, 20, 5]),
            number=rng.choice([5, 7, 9, 10, 12, 3, 4, 6]),
        )

    rows = []
    for _ in range(n_samples // 2):
        rows.append({"text": fill(rng.choice(spam_templates)), "label": "spam"})
        rows.append({"text": fill(rng.choice(ham_templates)), "label": "ham"})

    # Sprinkle in ambiguous examples (~18% of the dataset)
    n_ambiguous = max(1, int(n_samples * 0.18))
    for _ in range(n_ambiguous):
        template, label = rng.choice(ambiguous_templates)
        rows.append({"text": fill(template), "label": label})

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add label noise (~9%) to mimic real-world annotation imperfection,
    # so no model achieves a suspicious perfect score and the three
    # algorithms show meaningfully different strengths/weaknesses.
    noise_rng = random.Random(seed + 1)
    n_noisy = max(1, int(len(df) * 0.09))
    noisy_idx = noise_rng.sample(range(len(df)), n_noisy)
    for i in noisy_idx:
        df.loc[i, "label"] = "ham" if df.loc[i, "label"] == "spam" else "spam"

    return df


# ---------------------------------------------------------------------------
# 2. Training pipeline (shared by train_models.py and app.py)
# ---------------------------------------------------------------------------
def get_model_zoo():
    """Dict of model name -> untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM (Linear Kernel)": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
    }


def train_and_evaluate(df: pd.DataFrame, text_col: str = "text", label_col: str = "label"):
    """
    Trains Logistic Regression, SVM, and Naive Bayes on the given dataframe
    and returns:
      - metrics_df : comparison table (Accuracy / Precision / Recall / F1)
      - vectorizer : the fitted TfidfVectorizer
      - artifacts  : dict[model_name] -> {model, y_test, y_pred, y_proba}
      - pos_label  : which class was treated as the "positive" class
    """
    labels = df[label_col].astype(str)
    classes = sorted(labels.unique())

    # Prefer an intuitively "positive" class name if present (spam/fake/etc.)
    positive_candidates = [c for c in classes if c.lower() in
                            ("spam", "fake", "positive", "1", "yes", "true")]
    pos_label = positive_candidates[0] if positive_candidates else classes[-1]
    y = (labels == pos_label).astype(int)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df[text_col].astype(str), y, test_size=0.25,
        random_state=RANDOM_STATE, stratify=y,
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    rows = []
    artifacts = {}

    for name, model in get_model_zoo().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
        })

        artifacts[name] = {
            "model": model,
            "y_test": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    metrics_df = pd.DataFrame(rows)
    return metrics_df, vectorizer, artifacts, pos_label


# ---------------------------------------------------------------------------
# 3. Persistence helpers
# ---------------------------------------------------------------------------
def save_artifacts(metrics_df, vectorizer, artifacts, pos_label):
    metrics_df.to_csv(MODELS_DIR / "metrics.csv", index=False)
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    joblib.dump({"artifacts": artifacts, "pos_label": pos_label}, MODELS_DIR / "artifacts.pkl")
    for name, art in artifacts.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(art["model"], MODELS_DIR / f"model_{safe_name}.pkl")


def load_artifacts():
    metrics_df = pd.read_csv(MODELS_DIR / "metrics.csv")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    data = joblib.load(MODELS_DIR / "artifacts.pkl")
    return metrics_df, vectorizer, data["artifacts"], data["pos_label"]


# ---------------------------------------------------------------------------
# 4. Confusion matrix & ROC curve helpers
# ---------------------------------------------------------------------------
def get_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)


def get_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
