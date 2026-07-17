# Automatic Text Summarizer (Extractive + Abstractive)

## Project Overview

This project automatically generates summaries from long text using two different Natural Language Processing (NLP) approaches:

1. Extractive Summarization
2. Abstractive Summarization

The system compares both summaries based on readability, length, factual accuracy, and ROUGE evaluation scores.

---

## Features

- Extractive Summarization using TF-IDF Sentence Ranking
- Abstractive Summarization using Facebook BART
- Text Preprocessing
- Sentence Tokenization
- Word Tokenization
- User Input Support
- Load Text From File
- Summary Length Control
    - Short
    - Medium
    - Long
- Highlight Selected Sentences
- Save Summary to TXT
- Save Summary to CSV
- ROUGE Score Evaluation
- Text Statistics
- Comparison between Extractive and Abstractive Summaries
- Command Line Interface (CLI)

---

## Technologies Used

- Python
- NLTK
- Scikit-Learn
- Hugging Face Transformers
- Torch
- Pandas
- ROUGE Score
- Tabulate

---

## Folder Structure

```
Text_Summarizer/
│
├── main.py
├── extractive.py
├── abstractive.py
├── utils.py
├── evaluation.py
├── requirements.txt
├── sample_texts.txt
└── README.md
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---

## Workflow

1. Enter custom text OR load text file.
2. Choose summary length.
3. Generate Extractive Summary.
4. Generate Abstractive Summary.
5. Compare summaries.
6. View highlighted sentences.
7. Calculate ROUGE Scores.
8. Save summaries.

---

## Extractive Summarization

Technique Used:

- TF-IDF Vectorization
- Sentence Ranking

Advantages:

- Preserves original wording
- High factual accuracy
- Fast

Disadvantages:

- Less fluent
- Can sound repetitive

---

## Abstractive Summarization

Model Used:

facebook/bart-large-cnn

Advantages:

- Human-like summaries
- Better readability
- Better compression

Disadvantages:

- Slower
- May slightly change wording

---

## ROUGE Evaluation

The project calculates:

- ROUGE-1
- ROUGE-2
- ROUGE-L

---

## Example Output

Original Text

```
Artificial Intelligence is transforming industries by automating
tasks and improving efficiency.
```

Extractive Summary

```
Artificial Intelligence is transforming industries.
It improves efficiency.
```

Abstractive Summary

```
AI is revolutionizing industries by making processes
more efficient.
```

---

## Bonus Features

✔ Custom User Input

✔ Summary Length Control

✔ Save TXT

✔ Save CSV

✔ CLI Menu

✔ Highlight Selected Sentences

✔ ROUGE Evaluation

✔ Statistics

---

## Future Improvements

- GUI using Tkinter
- Web Application using Flask
- Streamlit Interface
- Pegasus Summarizer
- GPT-based Summarization
- Multi-document Summarization

---

## Author

Automatic Text Summarizer Project

NLP Assignment