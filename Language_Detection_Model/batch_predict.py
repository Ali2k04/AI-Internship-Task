import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Read CSV
df = pd.read_csv("input.csv")

# Convert text into vectors
X = vectorizer.transform(df["text"])

# Predict
df["Predicted Language"] = model.predict(X)

# Save output
df.to_csv("output.csv", index=False)

print("Prediction completed.")
print("Saved as output.csv")