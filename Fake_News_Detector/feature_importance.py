import joblib
import matplotlib.pyplot as plt

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

if model.__class__.__name__ != "LogisticRegression":
    print("Feature importance is only available for Logistic Regression.")
    exit()

feature_names = vectorizer.get_feature_names_out()

coef = model.coef_[0]

top_positive = coef.argsort()[-20:]
top_negative = coef.argsort()[:20]

plt.figure(figsize=(12,6))

plt.barh(
    feature_names[top_positive],
    coef[top_positive]
)

plt.title("Top Words Predicting REAL News")

plt.show()

plt.figure(figsize=(12,6))

plt.barh(
    feature_names[top_negative],
    coef[top_negative]
)

plt.title("Top Words Predicting FAKE News")

plt.show()