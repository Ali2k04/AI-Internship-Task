"""
abstractive.py
--------------------------
Abstractive Text Summarization
Uses Hugging Face Transformers (facebook/bart-large-cnn)

The model is downloaded automatically the first time you run it.
"""

from transformers import pipeline


class AbstractiveSummarizer:

    def __init__(self):
        print("\nLoading BART model (first run may take a few minutes)...")

        self.model = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

        print("Model Loaded Successfully!\n")

    def summarize(self, text, length="medium"):

        # Summary Length Settings
        settings = {
            "short": {
                "max_length": 60,
                "min_length": 20
            },
            "medium": {
                "max_length": 120,
                "min_length": 40
            },
            "long": {
                "max_length": 200,
                "min_length": 80
            }
        }

        config = settings.get(length.lower(), settings["medium"])

        # Hugging Face models work best with <=1024 tokens.
        # If the input is very long, truncate it.
        text = text[:3500]

        summary = self.model(
            text,
            max_length=config["max_length"],
            min_length=config["min_length"],
            do_sample=False
        )

        return summary[0]["summary_text"]