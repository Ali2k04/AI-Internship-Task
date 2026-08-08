# Intent Classifier

A machine-learning system that detects user intent from short text messages —
the core building block behind chatbots and virtual assistants.

```
"hi"                    -> greeting
"order pizza"            -> food_order
"what's the weather?"    -> weather
"how much does this cost" -> price_query
```

## Folder Structure

```
Intent_Classifier/
│── dataset.csv        # Labeled training data (text, intent)
│── utils.py            # Text preprocessing (spaCy / NLTK / regex fallback)
│── model.py             # Training, evaluation, saving, prediction logic
│── main.py               # CLI entry point (train + interactive chat loop)
│── setup_nlp.py           # Optional one-time downloader for spaCy/NLTK data
│── requirements.txt
│── README.md
└── saved_model/            # Created after training (vectorizer + model + metrics)
```

## How It Works

1. **Dataset** — `dataset.csv` holds ~150 labeled examples across 9 intents:
   `greeting`, `goodbye`, `food_order`, `weather`, `name`, `thanks`, `help`,
   `booking`, `price_query`.
2. **Preprocessing** (`utils.py`) — lowercases text, strips punctuation/numbers,
   and lemmatizes words. It prefers **spaCy**, falls back to **NLTK**, and
   falls back again to plain regex cleaning if neither is installed — so the
   project always runs, and gets better as more of the stack is available.
3. **Feature extraction** — `TfidfVectorizer` turns cleaned text into numeric
   vectors.
4. **Model training & comparison** (`model.py`) — the dataset is split into
   train/test sets, and three models are trained and compared:
   Logistic Regression, Multinomial Naive Bayes, and a Linear SVM. Each is
   scored with accuracy + a full `classification_report`. The
   highest-accuracy model is kept as the "production" model.
5. **Persistence** — the winning vectorizer + model are saved to
   `saved_model/` with `joblib`, alongside a `metrics.json` summary, so you
   don't have to retrain every run.
6. **Prediction** — `predict_intent()` returns both the predicted label and a
   **confidence score** (from `predict_proba`).

## Setup

```bash
pip install -r requirements.txt

# Optional but recommended — downloads the spaCy English model and NLTK
# corpora used for better preprocessing. Safe to skip.
python setup_nlp.py
```

## Usage

Train the model and evaluate it:

```bash
python model.py
```

This prints an accuracy + classification report for each candidate model and
saves the best one to `saved_model/`.

Run the interactive chat demo (trains automatically the first time):

```bash
python main.py
```

```
You: hello there
Intent: greeting  (confidence: 42.10%)

You: I want to order pizza
Intent: food_order  (confidence: 55.30%)

You: is it going to rain today
Intent: weather  (confidence: 48.70%)

You: exit
Goodbye!
```

Force a fresh retrain (e.g. after editing `dataset.csv`):

```bash
python main.py --retrain
```

## Using It Programmatically

```python
from model import load_model, predict_intent

vectorizer, model = load_model()
label, confidence = predict_intent("book a table for two", vectorizer, model)
print(label, confidence)   # booking 0.41
```

## Bonus Features Implemented

- ✅ Multiple models trained & compared (Logistic Regression, Naive Bayes, Linear SVM)
- ✅ Train/test split (`train_test_split`, stratified where possible)
- ✅ Full evaluation with `classification_report`
- ✅ spaCy-based preprocessing with automatic NLTK / regex fallback
- ✅ Confidence scores via `predict_proba`
- ✅ Model + vectorizer persistence with `joblib` (`saved_model/`)
- ✅ Clean, modular code (`utils.py` / `model.py` / `main.py`)
- ⏭️ Chatbot integration — this module exposes `load_model()` +
  `predict_intent()` so it can be dropped straight into a larger
  chatbot/assistant pipeline as the intent-detection component.

## Extending the Project

- Add more examples per intent in `dataset.csv` for better accuracy —
  short-text intent classifiers benefit a lot from more labeled data.
- Add new intents by adding new `(text, intent)` rows — no code changes
  needed.
- Swap in a transformer-based model (e.g. a fine-tuned DistilBERT) by
  replacing the vectorizer/model in `model.py` while keeping the same
  `predict_intent()` interface.
