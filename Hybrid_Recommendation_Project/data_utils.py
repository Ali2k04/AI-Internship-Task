"""
data_utils.py
-------------
Small shared helpers for loading the items/ratings CSVs and, if they don't
exist yet, generating them automatically so the project runs out-of-the-box.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"

ITEMS_PATH = DATA_DIR / "items.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"


def ensure_dataset_exists():
    """Generate the synthetic dataset the first time the project is run."""
    if ITEMS_PATH.exists() and RATINGS_PATH.exists():
        return
    print("No dataset found -- generating synthetic dataset...")
    subprocess.run(
        [sys.executable, str(DATA_DIR / "generate_dataset.py")],
        check=True,
    )


def load_data():
    """Return (ratings_df, items_df), generating data if needed."""
    ensure_dataset_exists()
    ratings_df = pd.read_csv(RATINGS_PATH)
    items_df = pd.read_csv(ITEMS_PATH)
    return ratings_df, items_df


def ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR
