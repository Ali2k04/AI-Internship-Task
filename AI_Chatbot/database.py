"""
database.py
------------
Persist conversation history and context to a local SQLite database
(bonus feature: "store conversation history in database").

No external dependencies - uses Python's built-in sqlite3 module.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "chatbot_history.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            sender TEXT NOT NULL,          -- 'user' or 'bot'
            message TEXT NOT NULL,
            intent TEXT,
            entities TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS context (
            user_id TEXT PRIMARY KEY,
            last_intent TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_message(user_id: str, sender: str, message: str, intent: str = None, entities: str = None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (user_id, sender, message, intent, entities, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, sender, message, intent, entities, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(user_id: str, limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT sender, message, intent, timestamp FROM messages "
        "WHERE user_id = ? ORDER BY id ASC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_context(user_id: str, intent: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO context (user_id, last_intent, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET last_intent = excluded.last_intent, "
        "updated_at = excluded.updated_at",
        (user_id, intent, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_context(user_id: str):
    conn = get_connection()
    row = conn.execute(
        "SELECT last_intent FROM context WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return row["last_intent"] if row else None


# Initialize the DB as soon as this module is imported.
init_db()
