import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

from utils import preprocess

# -------------------------------------
# Load Dataset
# -------------------------------------

df = pd.read_csv("dataset.csv")

print(df.head())

# -------------------------------------
# Clean Text
# -------------------------------------

df["clean_text"] = df["text"].astype(str).apply(preprocess)

# -------------------------------------
# Features
# -------------------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])

y = df["label"]

# -------------------------------------
# Split
# -------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------
# Models
# -------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

best_model = None
best_score = 0

print("\n==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("\n", name)
    print("Accuracy :", acc)
    print("Precision:", precision_score(
        y_test,
        pred,
        pos_label="Real"
    ))

    print("Recall   :", recall_score(
        y_test,
        pred,
        pos_label="Real"
    ))

    print("F1 Score :", f1_score(
        y_test,
        pred,
        pos_label="Real"
    ))

    print(classification_report(y_test, pred))

    if acc > best_score:
        best_score = acc
        best_model = model
        best_prediction = pred

print("\n==============================")
print("Best Model Selected")
print(best_model)

# -------------------------------------
# Confusion Matrix
# -------------------------------------

cm = confusion_matrix(y_test, best_prediction)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# -------------------------------------
# Save Model
# -------------------------------------

joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully")