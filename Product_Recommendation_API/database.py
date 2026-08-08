"""
database.py
Handles SQLite database setup, seeding from CSV files, and basic CRUD
helpers used by the API (products, interactions).
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "recommendation.db"
PRODUCTS_CSV = BASE_DIR / "data" / "products.csv"
INTERACTIONS_CSV = BASE_DIR / "data" / "interactions.csv"


def get_connection():
    """Return a new SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(reset: bool = False):
    """
    Create tables and seed them from the CSV files if the database
    doesn't already exist (or if reset=True).
    """
    first_run = reset or not DB_PATH.exists()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            features TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
        """
    )
    conn.commit()

    if first_run:
        # Seed products
        products_df = pd.read_csv(PRODUCTS_CSV)
        products_df.to_sql("products", conn, if_exists="replace", index=False)

        # Seed interactions
        interactions_df = pd.read_csv(INTERACTIONS_CSV)
        interactions_df.to_sql("interactions_seed", conn, if_exists="replace", index=False)
        cur.execute("DELETE FROM interactions")
        cur.execute(
            """
            INSERT INTO interactions (user_id, product_id, rating)
            SELECT user_id, product_id, rating FROM interactions_seed
            """
        )
        cur.execute("DROP TABLE interactions_seed")
        conn.commit()

    conn.close()


def get_all_products(category: str = None, min_price: float = None, max_price: float = None):
    conn = get_connection()
    query = "SELECT * FROM products WHERE 1=1"
    params = []

    if category:
        query += " AND LOWER(category) = LOWER(?)"
        params.append(category)
    if min_price is not None:
        query += " AND price >= ?"
        params.append(min_price)
    if max_price is not None:
        query += " AND price <= ?"
        params.append(max_price)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_product(product_id: int):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM products WHERE product_id = ?", (product_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_products_dataframe():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    return df


def get_interactions_dataframe():
    conn = get_connection()
    df = pd.read_sql_query("SELECT user_id, product_id, rating FROM interactions", conn)
    conn.close()
    return df


def add_interaction(user_id: int, product_id: int, rating: float):
    conn = get_connection()
    conn.execute(
        "INSERT INTO interactions (user_id, product_id, rating) VALUES (?, ?, ?)",
        (user_id, product_id, rating),
    )
    conn.commit()
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return new_id


def get_user_interactions(user_id: int):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM interactions WHERE user_id = ?", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
