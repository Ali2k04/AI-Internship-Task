"""
chatbot.py
----------
Core context-aware chatbot engine. Ties together:
  - intent classification (model.py)
  - preprocessing + NER (utils.py)
  - persistent context + history (database.py)

This module has no UI of its own - it's imported by app.py (Flask web UI)
and by the __main__ block below (terminal chat), so the same logic
powers both interfaces.
"""

import json
import random
from pathlib import Path

from model import train_default_model, load_intents
from utils import preprocess, extract_entities, extract_city, get_backend
import database as db

INTENTS_PATH = Path(__file__).parent / "intents.json"

# In-memory cache mirrors the DB so we don't hit SQLite on every single
# turn; database.py is the source of truth across restarts.
_context_cache = {}


class Chatbot:
    def __init__(self, engine: str = "tfidf", confidence_threshold: float = 0.25):
        self.intents = load_intents(str(INTENTS_PATH))
        self.classifier = train_default_model(engine=engine)
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    def _respond_for(self, intent: str) -> str:
        payload = self.intents.get(intent, {})
        responses = payload.get("responses", []) if isinstance(payload, dict) else []
        if responses:
            return random.choice(responses)
        return "Sorry, I didn't quite understand that. Could you rephrase?"

    # ------------------------------------------------------------------
    def get_context(self, user_id: str):
        if user_id in _context_cache:
            return _context_cache[user_id]
        ctx = db.get_context(user_id)
        _context_cache[user_id] = ctx
        return ctx

    def set_context(self, user_id: str, intent: str):
        _context_cache[user_id] = intent
        db.set_context(user_id, intent)

    # ------------------------------------------------------------------
    def chat(self, user_input: str, user_id: str = "user1") -> dict:
        """
        Process one turn of conversation.
        Returns a dict with the bot's reply plus debugging metadata
        (intent, confidence, entities) so the web UI / caller can show it.
        """
        user_input = user_input.strip()
        entities = extract_entities(user_input)
        prev_intent = self.get_context(user_id)

        # ---- Context-aware branch --------------------------------------
        # If the previous turn was "weather" and this turn looks like a
        # follow-up (e.g. just a city name), answer using that context
        # instead of re-classifying from scratch. This is the same idea
        # as the sample chatbot() in the task brief, generalized a bit.
        if prev_intent == "weather":
            city = extract_city(user_input) or user_input
            reply = f"Here's the weather info for {city}: (live lookup not wired up in this demo, plug in a weather API here)."
            intent, confidence = "weather_followup", 1.0
            # Clear the "waiting for city" context now that we've answered.
            self.set_context(user_id, None)
        else:
            intent, confidence = self.classifier.predict(user_input)
            if confidence < self.confidence_threshold:
                reply = "Sorry, I didn't understand that. Could you rephrase?"
                intent = "unknown"
            else:
                reply = self._respond_for(intent)
                self.set_context(user_id, intent)

        # ---- Persist turn to DB (bonus: conversation history storage) --
        db.log_message(user_id, "user", user_input, intent, json.dumps(entities))
        db.log_message(user_id, "bot", reply, intent, None)

        return {
            "reply": reply,
            "intent": intent,
            "confidence": round(confidence, 3),
            "entities": entities,
            "backend": get_backend(),
        }

    def history(self, user_id: str = "user1"):
        return db.get_history(user_id)


# ------------------------------------------------------------------------
# Terminal chat loop (matches the step-by-step brief's example usage)
# ------------------------------------------------------------------------
if __name__ == "__main__":
    bot = Chatbot()
    print(f"Chatbot ready (NLP backend: {get_backend()}). Type 'exit' to quit.\n")
    user_id = "cli_user"
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        result = bot.chat(user_input, user_id=user_id)
        print("Bot:", result["reply"])
