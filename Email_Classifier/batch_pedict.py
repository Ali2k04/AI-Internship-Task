import pandas as pd
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

df = pd.read_csv("emails.csv")   # File containing email column

X = vectorizer.transform(df["email"])

df["Prediction"] = model.predict(X)

df.to_csv("predictions.csv", index=False)

print("Predictions saved successfully!")