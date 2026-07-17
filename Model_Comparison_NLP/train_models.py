# ==========================================================
# NLP Model Comparison Project
# Models:
# 1. Multinomial Naive Bayes
# 2. Support Vector Machine (SVM)
# 3. Logistic Regression
#
# Features:
# ✔ Text Preprocessing
# ✔ TF-IDF Vectorization
# ✔ GridSearchCV
# ✔ Cross Validation
# ✔ Confusion Matrix
# ✔ Accuracy Comparison Graph
# ✔ Save Best Model
# ✔ User Prediction System
# ==========================================================

import re
import string
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ----------------------------------------------------------
# Download NLTK Data
# ----------------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("dataset.csv")

print("\nDataset Loaded Successfully")
print(df.head())

# ----------------------------------------------------------
# Text Preprocessing
# ----------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenization
    words = text.split()

    # Stopword Removal + Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["text"] = df["text"].apply(clean_text)

# ----------------------------------------------------------
# Split Dataset
# ----------------------------------------------------------

X = df["text"]

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
# ----------------------------------------------------------
# Machine Learning Models
# ----------------------------------------------------------

models = {

    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ]),

    "SVM": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LinearSVC())
    ]),

    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

}

# ----------------------------------------------------------
# Hyperparameter Tuning
# ----------------------------------------------------------

parameters = {

    "Naive Bayes": {

        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__alpha": [0.5, 1.0]

    },

    "SVM": {

        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.5, 1, 2]

    },

    "Logistic Regression": {

        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.5, 1, 2]

    }

}

# ----------------------------------------------------------
# Training Models
# ----------------------------------------------------------

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

print("\n====================================")
print("Training Models...")
print("====================================")

for model_name, pipeline in models.items():

    print(f"\nTraining {model_name}...")

    grid = GridSearchCV(

        estimator=pipeline,
        param_grid=parameters[model_name],
        cv=5,
        scoring="accuracy"

    )

    grid.fit(X_train, y_train)

    predictions = grid.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    cv_score = cross_val_score(

        grid.best_estimator_,
        X,
        y,
        cv=5,
        scoring="accuracy"

    ).mean()

    results.append([

        model_name,
        accuracy,
        precision,
        recall,
        f1,
        cv_score

    ])

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"CV Score : {cv_score:.4f}")

    # Save best model

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = grid.best_estimator_
        best_model_name = model_name
        # ----------------------------------------------------------
# Model Comparison Table
# ----------------------------------------------------------

results_df = pd.DataFrame(

    results,

    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Cross Validation"
    ]

)

print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df)

# ----------------------------------------------------------
# Plot Accuracy Graph
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Accuracy Comparison of NLP Models")

plt.xlabel("Models")

plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("accuracy_comparison.png")

plt.show()

print("\nAccuracy graph saved as accuracy_comparison.png")

# ----------------------------------------------------------
# Confusion Matrix + Classification Report
# ----------------------------------------------------------

print("\nGenerating Confusion Matrices...\n")

for model_name, pipeline in models.items():

    print("=" * 60)
    print(model_name)
    print("=" * 60)

    grid = GridSearchCV(

        estimator=pipeline,

        param_grid=parameters[model_name],

        cv=5,

        scoring="accuracy"

    )

    grid.fit(X_train, y_train)

    predictions = grid.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5,4))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    filename = model_name.replace(" ", "_") + "_confusion_matrix.png"

    plt.savefig(filename)

    plt.show()

    print("\nClassification Report\n")

    print(classification_report(y_test, predictions))

# ----------------------------------------------------------
# Save Best Model
# ----------------------------------------------------------

joblib.dump(best_model, "best_model.pkl")

print("\n========================================")
print("Best Model Saved Successfully")
print("File Name : best_model.pkl")
print("========================================")

print("\nBest Performing Model :", best_model_name)

print("Best Accuracy         :", round(best_accuracy * 100, 2), "%")
# ----------------------------------------------------------
# User Prediction System
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("          NLP TEXT CLASSIFICATION SYSTEM")
print("=" * 60)

while True:

    print("\nOptions")
    print("1. Predict Text")
    print("2. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":

        text = input("\nEnter text: ")

        cleaned_text = clean_text(text)

        prediction = best_model.predict([cleaned_text])[0]

        print("\n" + "=" * 45)
        print("Prediction Result")
        print("=" * 45)

        print(f"Best Model      : {best_model_name}")
        print(f"Input Text      : {text}")
        print(f"Prediction      : {prediction}")

        print("=" * 45)

    elif choice == "2":

        print("\nThank you for using the NLP Model Comparison System.")
        break

    else:

        print("\nInvalid choice! Please enter 1 or 2.")