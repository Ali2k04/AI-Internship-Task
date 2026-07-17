import pandas as pd
import joblib

from utils import preprocess

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

input_file = input("Enter CSV filename: ")

df = pd.read_csv(input_file)

df["clean_text"] = df["text"].astype(str).apply(preprocess)

X = vectorizer.transform(df["clean_text"])

df["Prediction"] = model.predict(X)

if hasattr(model, "predict_proba"):
    confidence = model.predict_proba(X).max(axis=1) * 100
    df["Confidence"] = confidence.round(2)

output = "predictions.csv"

df.to_csv(output, index=False)

print("\nPrediction completed.")
print("Saved as:", output)