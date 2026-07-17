import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()

    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

while True:

    email = input("\nEnter Email:\n")

    clean = preprocess(email)

    vec = vectorizer.transform([clean])

    prediction = model.predict(vec)[0]

    probability = model.predict_proba(vec).max()

    print("\nCategory:", prediction)

    print("Confidence:", round(probability*100,2),"%")

    choice = input("\nContinue? (y/n): ")

    if choice.lower()!="y":
        break