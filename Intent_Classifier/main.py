"""
main.py
-------
Entry point for the Intent Classification System.

Usage:
    python main.py            # trains (if needed) then starts an interactive prompt
    python main.py --retrain  # forces retraining even if a saved model exists

Type 'exit' at the prompt to quit.
"""

import argparse
import os

from model import (
    train_and_compare,
    save_model,
    load_model,
    predict_intent,
    VECTORIZER_PATH,
    MODEL_PATH,
)
from utils import preprocessing_backend


def ensure_model(retrain: bool = False):
    """Train + save a model if one doesn't already exist (or retrain is forced)."""
    if retrain or not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
        print("Training model...\n")
        outcome = train_and_compare()
        save_model(
            outcome["vectorizer"],
            outcome["best_model"],
            outcome["results"],
            outcome["best_name"],
        )
        return outcome["vectorizer"], outcome["best_model"]

    print("Loading existing saved model (use --retrain to train fresh)...")
    return load_model()


def interactive_loop(vectorizer, model):
    print(f"\nPreprocessing backend: {preprocessing_backend()}")
    print("Intent Classifier ready. Type a message (or 'exit' to quit).\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        label, confidence = predict_intent(user_input, vectorizer, model)
        if confidence is not None:
            print(f"Intent: {label}  (confidence: {confidence:.2%})")
        else:
            print(f"Intent: {label}")


def main():
    parser = argparse.ArgumentParser(description="Intent Classification System")
    parser.add_argument(
        "--retrain", action="store_true", help="Force retraining even if a saved model exists"
    )
    args = parser.parse_args()

    vectorizer, model = ensure_model(retrain=args.retrain)
    interactive_loop(vectorizer, model)


if __name__ == "__main__":
    main()
