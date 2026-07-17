import joblib
from utils import preprocess

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def predict_news(news):

    clean = preprocess(news)

    vector = vectorizer.transform([clean])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max() * 100

    print("\nPrediction :", prediction)
    print(f"Confidence : {probability:.2f}%")

    return prediction


if __name__ == "__main__":

    while True:

        print("\n==========================")

        news = input("Enter News (type exit to quit): ")

        if news.lower() == "exit":
            break

        predict_news(news)