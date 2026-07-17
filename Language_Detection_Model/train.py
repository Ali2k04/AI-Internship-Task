# ==========================================================
# Language Detection Model
# Train Script
# ==========================================================

import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully!")
print(df.head())


# ==========================================================
# Text Preprocessing
# ==========================================================

def clean_text(text):
    """
    Clean text while preserving language-specific characters.
    """

    text = str(text).lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["text"] = df["text"].apply(clean_text)


# ==========================================================
# Features & Labels
# ==========================================================

X = df["text"]
y = df["language"]


# ==========================================================
# Split Dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ==========================================================
# TF-IDF Character Features
# ==========================================================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4)
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# ==========================================================
# Train Model
# ==========================================================

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

print("\nModel Training Complete!")


# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(X_test_vectorized)


# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy :", round(accuracy * 100, 2), "%")


# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(classification_report(
    y_test,
    predictions
))


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Language")
plt.ylabel("Actual Language")
plt.title("Confusion Matrix")

plt.show()


# ==========================================================
# Save Model
# ==========================================================

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully!")

print("model.pkl")
print("vectorizer.pkl")