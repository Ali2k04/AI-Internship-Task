"""
database.py
-----------
Thin SQLite persistence layer for the Personalized Feed Engine.

Tables
------
interactions       raw event log (user_id, content_id, interaction_type, timestamp)
user_profiles       aggregated per-user/content weighted score (recomputed periodically)
ab_assignments      which A/B test group each user has been assigned to
feed_impressions     every feed the engine has ever generated, for offline analysis
"""

import sqlite3
from contextlib import contextmanager

import pandas as pd

DB_PATH = "data/feed_engine.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content_id INTEGER NOT NULL,
    interaction_type TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id INTEGER NOT NULL,
    content_id INTEGER NOT NULL,
    score REAL NOT NULL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (user_id, content_id)
);

CREATE TABLE IF NOT EXISTS ab_assignments (
    user_id INTEGER PRIMARY KEY,
    group_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feed_impressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    content_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    group_name TEXT,
    strategy TEXT,
    generated_at TEXT NOT NULL
);
"""


@contextmanager
def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def load_interactions_csv(csv_path, db_path=DB_PATH):
    """Load a user_behavior.csv into the interactions table (replaces existing rows)."""
    df = pd.read_csv(csv_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM interactions")
        df[["user_id", "content_id", "interaction_type", "timestamp"]].to_sql(
            "interactions", conn, if_exists="append", index=False
        )


def fetch_interactions(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        return pd.read_sql("SELECT * FROM interactions", conn)


def insert_interaction(user_id, content_id, interaction_type, timestamp, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO interactions (user_id, content_id, interaction_type, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (user_id, content_id, interaction_type, str(timestamp)),
        )


def save_user_profile(user_id, content_scores: dict, db_path=DB_PATH):
    """content_scores: {content_id: score}. Upserts one user's aggregated profile."""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        for content_id, score in content_scores.items():
            conn.execute(
                """INSERT INTO user_profiles (user_id, content_id, score, last_updated)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, content_id)
                   DO UPDATE SET score = excluded.score, last_updated = excluded.last_updated""",
                (user_id, content_id, float(score), now),
            )


def get_ab_assignment(user_id, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT group_name FROM ab_assignments WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else None


def set_ab_assignment(user_id, group_name, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO ab_assignments (user_id, group_name) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET group_name = excluded.group_name""",
            (user_id, group_name),
        )


def log_feed_impressions(user_id, feed, group_name, strategy, db_path=DB_PATH):
    """feed: list of dicts with content_id, score (as returned by FeedEngine.get_feed)."""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection(db_path) as conn:
        for rank, item in enumerate(feed, start=1):
            conn.execute(
                """INSERT INTO feed_impressions
                   (user_id, content_id, rank, score, group_name, strategy, generated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, item["content_id"], rank, item["score"], group_name, strategy, now),
            )


def fetch_feed_impressions(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        return pd.read_sql("SELECT * FROM feed_impressions", conn)
