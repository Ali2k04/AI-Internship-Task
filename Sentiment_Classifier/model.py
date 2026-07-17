import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils import preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("dataset.csv")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values")

print(df.isnull().sum())

# -------------------------------
# Text Preprocessing
# -------------------------------

df["Processed_Text"] = df["Text"].apply(preprocess_text)

print("\nProcessed Data")

print(df[["Text", "Processed_Text", "Sentiment"]].head())

# -------------------------------
# TF-IDF
# -------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["Processed_Text"])

y = df["Sentiment"]

print("\nFeature Matrix Shape")

print(X.shape)

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])

print("Testing Samples:", X_test.shape[0])

# -------------------------------
# Logistic Regression
# -------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy")

print(round(accuracy * 100, 2), "%")

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# -------------------------------
# Save Model
# -------------------------------

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("\nModel Saved Successfully")

# -------------------------------
# Prediction System
# -------------------------------

while True:

    print("\n--------------------------")

    print("1. Predict Sentiment")

    print("2. Exit")

    choice = input("Enter Choice : ")

    if choice == "1":

        sentence = input("\nEnter your sentence:\n")

        processed = preprocess_text(sentence)

        vector = vectorizer.transform(
            [processed]
        )

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        print("\nPredicted Sentiment:", prediction)

        print("\nPrediction Probabilities")

        for label, prob in zip(
            model.classes_,
            probability
        ):

            print(label, ":", round(prob*100,2), "%")

        result = pd.DataFrame({
            "Sentence":[sentence],
            "Prediction":[prediction]
        })

        result.to_csv(
            "predictions.csv",
            mode="a",
            header=False,
            index=False
        )

    elif choice == "2":

        print("Good Bye")

        break

    else:

        print("Invalid Choice")