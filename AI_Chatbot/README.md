# Context-Aware AI Chatbot

A context-aware chatbot that classifies user intent, remembers conversation
state per user, and generates relevant responses — built with a classic
NLP pipeline and several optional upgrades (spaCy, NER, transformers,
a Flask web UI, voice I/O, and SQLite-backed conversation history).

## Folder Structure

```
AI_Chatbot/
│── intents.json          # Intent dataset (examples + responses)
│── utils.py               # Preprocessing + Named Entity Recognition
│── model.py                # TF-IDF/LogReg classifier + transformer option
│── database.py             # SQLite conversation history & context storage
│── voice.py                 # Optional voice input/output
│── chatbot.py                # Core chatbot engine (context-aware chat loop)
│── app.py                     # Flask web UI
│── templates/index.html        # Chat web page
│── requirements.txt
│── README.md
```

## Quick Start

### 1. Install dependencies

Minimum (terminal chatbot only):
```bash
pip install scikit-learn nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Recommended (spaCy backend + NER, bonus feature):
```bash
pip install spacy scikit-learn flask
python -m spacy download en_core_web_sm
```

Full bonus set (transformers + voice + web UI):
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> The code auto-detects what's installed: spaCy is used if available,
> otherwise it falls back to NLTK, otherwise a plain regex tokenizer.
> Nothing crashes if you skip the optional installs — you just lose that
> particular upgrade.

### 2. Run in the terminal

```bash
python chatbot.py
```

```
Chatbot ready (NLP backend: spacy). Type 'exit' to quit.
You: hi
Bot: Hello! How can I help you today?
You: weather today
Bot: Please tell me your city and I will look up the weather.
You: Karachi
Bot: Here's the weather info for Karachi: (live lookup not wired up in this demo, plug in a weather API here)
```

### 3. Run the web UI (bonus feature)

```bash
python app.py
```
Open **http://127.0.0.1:5000** — a chat bubble UI with a microphone
button (server-side voice capture, if `voice.py` deps are installed).

## How It Works

1. **Preprocessing** (`utils.py`) — lowercases, tokenizes, strips
   punctuation, and lemmatizes text via spaCy (preferred) or NLTK.
2. **Feature extraction** — TF-IDF vectorizes preprocessed examples from
   `intents.json`.
3. **Intent classification** (`model.py`) — a Logistic Regression model
   trained on the TF-IDF features. A `confidence` score (max class
   probability) is returned with every prediction; low-confidence
   predictions fall back to a "didn't understand" reply instead of
   guessing.
4. **Named Entity Recognition** (`utils.py`) — spaCy's NER pulls out
   entities like city names (`GPE`) from user text; without spaCy, a
   small city gazetteer is used as a fallback so the weather follow-up
   still works.
5. **Context management** (`chatbot.py` + `database.py`) — each user's
   last intent is cached in memory and persisted to SQLite. If the
   previous turn was "weather", the next turn is treated as a follow-up
   (e.g. just a city name) instead of being reclassified from scratch —
   this is the actual "memory" that makes the bot context-aware.
6. **Response generation** — a response is chosen (randomly, among a few
   templates) for the predicted intent from `intents.json`.
7. **History storage** (`database.py`) — every user and bot message is
   logged to `chatbot_history.db` (SQLite) with timestamp, intent, and
   extracted entities, so history survives restarts.

## Bonus Features Implemented

| Feature | Where | Notes |
|---|---|---|
| spaCy instead of NLTK | `utils.py` | Auto-detected; NLTK/regex fallback if spaCy isn't installed |
| Named Entity Recognition | `utils.py` | Real spaCy NER, or gazetteer fallback |
| Pretrained transformer model | `model.py` | `IntentClassifier(engine="transformer")` uses a zero-shot BART model from Hugging Face |
| Web UI | `app.py`, `templates/index.html` | Flask app with a styled chat interface |
| Voice input/output | `voice.py`, wired into `app.py`'s `/voice` route and the mic button in the UI | Optional deps; degrades gracefully if not installed |
| Conversation history in a database | `database.py` | SQLite, two tables: `messages` and `context` |

## Extending It

- **Add real weather data**: replace the placeholder string in
  `chatbot.py`'s `weather` follow-up branch with a call to a weather API
  (e.g. OpenWeatherMap), using the extracted city.
- **Add more intents**: extend `intents.json` with new keys — no code
  changes needed, `model.py` retrains from the file automatically.
- **Swap in the transformer engine**: `Chatbot(engine="transformer")` in
  `chatbot.py` or `app.py` (requires `transformers` + `torch`, and
  downloads model weights on first run).
- **Multi-turn memory beyond one intent**: `database.py`'s `messages`
  table already stores full history per `user_id` — you can feed recent
  turns back into the transformer engine as a running conversation
  window.
