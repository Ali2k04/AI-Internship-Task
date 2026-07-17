import re
import string
import pandas as pd
import nltk

# Download required resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


class TextPreprocessor:

    def __init__(self):

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    # ----------------------------------------------------
    # Step 1: Cleaning
    # ----------------------------------------------------
    def clean_text(self, text):

        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # ----------------------------------------------------
    # Step 2: Custom Tokenization
    # ----------------------------------------------------
    def tokenize(self, text):

        tokens = re.findall(r'\b[a-zA-Z]+\b', text)

        return tokens

    # ----------------------------------------------------
    # Step 3: Stopword Removal
    # ----------------------------------------------------
    def remove_stopwords(self, tokens):

        return [
            word
            for word in tokens
            if word not in self.stop_words
        ]

    # ----------------------------------------------------
    # Step 4: Lemmatization
    # ----------------------------------------------------
    def lemmatize(self, tokens):

        return [
            self.lemmatizer.lemmatize(word, pos='v')
            for word in tokens
        ]

    # ----------------------------------------------------
    # Bonus: Stemming
    # ----------------------------------------------------
    def stem(self, tokens):

        return [
            self.stemmer.stem(word)
            for word in tokens
        ]

    # ----------------------------------------------------
    # Complete Pipeline
    # ----------------------------------------------------
    def preprocess(self, text, show_steps=True):

        if text.strip() == "":
            return []

        cleaned = self.clean_text(text)

        tokens = self.tokenize(cleaned)

        filtered = self.remove_stopwords(tokens)

        lemmatized = self.lemmatize(filtered)

        stemmed = self.stem(filtered)

        if show_steps:

            print("\n-----------------------------")
            print("RAW TEXT:")
            print(text)

            print("\nCLEANED:")
            print(cleaned)

            print("\nTOKENS:")
            print(tokens)

            print("\nSTOPWORDS REMOVED:")
            print(filtered)

            print("\nLEMMATIZED:")
            print(lemmatized)

            print("\nSTEMMED:")
            print(stemmed)

            print("-----------------------------")

        return lemmatized

    # ----------------------------------------------------
    # Batch Processing
    # ----------------------------------------------------
    def process_csv(self,
                    input_file,
                    output_file,
                    column_name):

        df = pd.read_csv(input_file)

        df["Processed"] = df[column_name].apply(
            lambda x: " ".join(self.preprocess(str(x), False))
        )

        df.to_csv(output_file, index=False)

        print(f"\nProcessed file saved as: {output_file}")