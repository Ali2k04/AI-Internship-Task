"""
extractive.py
--------------
Extractive Text Summarization using TF-IDF Sentence Ranking
"""
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
import nltk
import re

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download('punkt', quiet=True)


class ExtractiveSummarizer:

    def __init__(self):
        pass

    def preprocess(self, text):

        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def summarize(self, text, ratio=0.3):

        text = self.preprocess(text)

        sentences = sent_tokenize(text)

        if len(sentences) <= 2:
            return text, list(range(len(sentences)))

        tfidf = TfidfVectorizer(stop_words='english')

        matrix = tfidf.fit_transform(sentences)

        scores = np.asarray(matrix.sum(axis=1)).flatten()

        num_sentences = max(1, int(len(sentences) * ratio))

        ranked = np.argsort(scores)[::-1]

        selected = sorted(ranked[:num_sentences])

        summary = " ".join([sentences[i] for i in selected])

        return summary, selected

    def highlight(self, text, indexes):

        sentences = sent_tokenize(text)

        output = []

        for i, sentence in enumerate(sentences):

            if i in indexes:
                output.append(f">>> {sentence}")

            else:
                output.append(sentence)

        return "\n".join(output)