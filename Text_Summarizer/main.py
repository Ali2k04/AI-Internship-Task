"""
main.py
-------------------------------------
Automatic Text Summarizer

Features
--------
1. Extractive Summarization
2. Abstractive Summarization
3. Compare both summaries
4. Highlight selected sentences
5. Save summaries to TXT
6. Save summaries to CSV
7. ROUGE Evaluation
8. Custom user input
9. File input
10. Summary Length Control
"""

import os

from extractive import ExtractiveSummarizer
from abstractive import AbstractiveSummarizer

from utils import (
    load_text,
    preprocess_text,
    print_statistics,
    compare_lengths,
    save_to_txt,
    save_to_csv
)

from evaluation import (
    RougeEvaluation,
    manual_comparison
)

# ----------------------------------------------------
# Initialize Models
# ----------------------------------------------------

extractive = ExtractiveSummarizer()
abstractive = AbstractiveSummarizer()
rouge = RougeEvaluation()

# ----------------------------------------------------
# Summary Length Menu
# ----------------------------------------------------

def choose_length():

    print("\nChoose Summary Length")

    print("1. Short")
    print("2. Medium")
    print("3. Long")

    choice = input("\nEnter choice : ")

    if choice == "1":
        return "short"

    elif choice == "2":
        return "medium"

    elif choice == "3":
        return "long"

    else:
        print("Invalid Choice.")
        print("Using Medium.")

        return "medium"

# ----------------------------------------------------
# Read Input Text
# ----------------------------------------------------

def read_text():

    print("\nInput Options")

    print("1. Enter Custom Text")

    print("2. Load Text File")

    choice = input("\nChoose : ")

    if choice == "1":

        print("\nPaste your text below.")
        print("Press ENTER twice when finished.\n")

        lines = []

        while True:

            line = input()

            if line == "":
                break

            lines.append(line)

        text = "\n".join(lines)

        return preprocess_text(text)

    elif choice == "2":

        path = input("\nEnter file path : ")

        if not os.path.exists(path):

            print("File Not Found.")

            return None

        text = load_text(path)

        return preprocess_text(text)

    else:

        print("Invalid Choice")

        return None

# ----------------------------------------------------
# Generate Summaries
# ----------------------------------------------------

def generate(text):

    print("\nGenerating Extractive Summary...")

    extractive_summary, indexes = extractive.summarize(
        text,
        ratio=0.30
    )

    print("Done.")

    length = choose_length()

    print("\nGenerating Abstractive Summary...")

    abstractive_summary = abstractive.summarize(
        text,
        length
    )

    print("Done.")

    return (
        extractive_summary,
        abstractive_summary,
        indexes
    )

# ----------------------------------------------------
# Show Results
# ----------------------------------------------------

def display_results(
        original,
        extractive_summary,
        abstractive_summary,
        indexes):

    print("\n")
    print("=" * 80)

    print("ORIGINAL TEXT")

    print("=" * 80)

    print(original)

    print("\n")

    print("=" * 80)

    print("EXTRACTIVE SUMMARY")

    print("=" * 80)

    print(extractive_summary)

    print("\n")

    print("=" * 80)

    print("ABSTRACTIVE SUMMARY")

    print("=" * 80)

    print(abstractive_summary)

    print("\n")

    print("=" * 80)

    print("HIGHLIGHTED SENTENCES")

    print("=" * 80)

    highlighted = extractive.highlight(
        original,
        indexes
    )

    print(highlighted)

    print("\n")

    compare_lengths(
        original,
        extractive_summary,
        abstractive_summary
    )

    print_statistics(
        "Original Text",
        original
    )

    print_statistics(
        "Extractive Summary",
        extractive_summary
    )

    print_statistics(
        "Abstractive Summary",
        abstractive_summary
    )
    # ----------------------------------------------------
# Save Results
# ----------------------------------------------------

def save_results(
        original,
        extractive_summary,
        abstractive_summary):

    print("\nDo you want to save the summaries?")

    print("1. Save as TXT")
    print("2. Save as CSV")
    print("3. Save Both")
    print("4. Skip")

    choice = input("\nEnter choice: ")

    if choice == "1":

        filename = input("Enter TXT filename (example: summary.txt): ")

        save_to_txt(
            filename,
            original,
            extractive_summary,
            abstractive_summary
        )

    elif choice == "2":

        filename = input("Enter CSV filename (example: summary.csv): ")

        save_to_csv(
            filename,
            original,
            extractive_summary,
            abstractive_summary
        )

    elif choice == "3":

        txt_name = input("TXT filename: ")
        csv_name = input("CSV filename: ")

        save_to_txt(
            txt_name,
            original,
            extractive_summary,
            abstractive_summary
        )

        save_to_csv(
            csv_name,
            original,
            extractive_summary,
            abstractive_summary
        )

    else:

        print("\nResults were not saved.")


# ----------------------------------------------------
# ROUGE Evaluation
# ----------------------------------------------------

def evaluate_results(
        original,
        extractive_summary,
        abstractive_summary):

    print("\n")
    print("=" * 80)
    print("ROUGE EVALUATION")
    print("=" * 80)

    print(
        "\nUsing the ORIGINAL TEXT as reference.\n"
        "(For research papers, a human-written summary is preferred.)"
    )

    rouge.compare(
        original,
        extractive_summary,
        abstractive_summary
    )

    manual_comparison()


# ----------------------------------------------------
# Main Menu
# ----------------------------------------------------

def menu():

    print("\n")
    print("=" * 80)
    print("AUTOMATIC TEXT SUMMARIZER")
    print("=" * 80)

    print("1. Summarize Text")
    print("2. Exit")

    return input("\nEnter choice: ")


# ----------------------------------------------------
# Main Function
# ----------------------------------------------------

def main():

    while True:

        choice = menu()

        if choice == "2":

            print("\nThank you for using the Text Summarizer!")

            break

        elif choice != "1":

            print("\nInvalid Choice.\n")

            continue

        # Read text
        text = read_text()

        if text is None:

            continue

        if len(text.strip()) < 50:

            print("\nPlease enter a longer text.")

            continue

        # Generate summaries
        extractive_summary, abstractive_summary, indexes = generate(text)

        # Display
        display_results(
            text,
            extractive_summary,
            abstractive_summary,
            indexes
        )

        # ROUGE
        evaluate_results(
            text,
            extractive_summary,
            abstractive_summary
        )

        # Save
        save_results(
            text,
            extractive_summary,
            abstractive_summary
        )

        print("\n")
        print("=" * 80)
        print("Summarization Completed Successfully.")
        print("=" * 80)

        again = input("\nSummarize another text? (y/n): ")

        if again.lower() != "y":

            print("\nGoodbye!")

            break


# ----------------------------------------------------
# Program Entry
# ----------------------------------------------------

if __name__ == "__main__":

    main()