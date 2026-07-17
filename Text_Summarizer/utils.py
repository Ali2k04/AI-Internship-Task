"""
utils.py
-----------------------------------
Utility functions for the Text Summarizer Project
"""
nltk.download("punkt")
nltk.download("punkt_tab")
import re
import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

# Download tokenizer
nltk.download("punkt", quiet=True)


# --------------------------------------------------
# Read text file
# --------------------------------------------------
def load_text(file_path):
    """
    Reads a text file and returns its content.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# --------------------------------------------------
# Clean Text
# --------------------------------------------------
def preprocess_text(text):
    """
    Performs basic preprocessing.
    """

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special symbols
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# Sentence Tokenization
# --------------------------------------------------
def get_sentences(text):
    """
    Returns list of sentences.
    """
    return sent_tokenize(text)


# --------------------------------------------------
# Word Tokenization
# --------------------------------------------------
def get_words(text):
    """
    Returns list of words.
    """
    return word_tokenize(text)


# --------------------------------------------------
# Text Statistics
# --------------------------------------------------
def text_statistics(text):
    """
    Returns basic statistics.
    """

    sentences = get_sentences(text)
    words = get_words(text)

    stats = {
        "Characters": len(text),
        "Words": len(words),
        "Sentences": len(sentences)
    }

    return stats


# --------------------------------------------------
# Save Summary to TXT
# --------------------------------------------------
def save_to_txt(filename,
                original,
                extractive,
                abstractive):

    with open(filename, "w", encoding="utf-8") as file:

        file.write("=========== ORIGINAL TEXT ===========\n\n")
        file.write(original)

        file.write("\n\n=========== EXTRACTIVE SUMMARY ===========\n\n")
        file.write(extractive)

        file.write("\n\n=========== ABSTRACTIVE SUMMARY ===========\n\n")
        file.write(abstractive)

    print(f"\nSaved successfully to {filename}")


# --------------------------------------------------
# Save Summary to CSV
# --------------------------------------------------
def save_to_csv(filename,
                original,
                extractive,
                abstractive):

    df = pd.DataFrame({

        "Original Text": [original],
        "Extractive Summary": [extractive],
        "Abstractive Summary": [abstractive]

    })

    df.to_csv(filename,
              index=False,
              encoding="utf-8")

    print(f"\nSaved successfully to {filename}")


# --------------------------------------------------
# Print Statistics
# --------------------------------------------------
def print_statistics(title, text):

    stats = text_statistics(text)

    print("\n----------------------------")
    print(title)
    print("----------------------------")

    for key, value in stats.items():
        print(f"{key}: {value}")


# --------------------------------------------------
# Display Comparison
# --------------------------------------------------
def compare_lengths(original,
                    extractive,
                    abstractive):

    print("\n=========== SUMMARY LENGTHS ===========")

    print(f"Original Words     : {len(get_words(original))}")
    print(f"Extractive Words   : {len(get_words(extractive))}")
    print(f"Abstractive Words  : {len(get_words(abstractive))}")

    print("=======================================\n")