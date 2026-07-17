import joblib
import numpy as np

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

feature_names = vectorizer.get_feature_names_out()

for i, category in enumerate(model.classes_):
    top = np.argsort(model.coef_[i])[-10:]

    print(f"\nTop Keywords for {category}:")
    for word in feature_names[top][::-1]:
        print(word)