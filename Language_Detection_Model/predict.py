# ==========================================
# Language Detection - Prediction Script
# ==========================================

import joblib
from langdetect import detect

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

while True:

    print("\n========== Language Detection ==========")
    print("Type 'exit' to quit.")

    text = input("\nEnter Text: ")

    if text.lower() == "exit":
        print("Goodbye!")
        break

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Predict language
    prediction = model.predict(text_vector)[0]

    print("\nPredicted Language:", prediction)

    # -------------------------------
    # Bonus 1
    # Top-2 Predictions
    # -------------------------------
    probabilities = model.predict_proba(text_vector)[0]

    results = list(zip(model.classes_, probabilities))
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 2 Predictions")

    for language, score in results[:2]:
        print(f"{language}: {score:.2%}")

    # -------------------------------
    # Bonus 2
    # Compare with langdetect
    # -------------------------------
    try:
        detected = detect(text)
        print("\nlangdetect Prediction:", detected)
    except:
        print("\nlangdetect could not detect the language.")