"""
train.py
--------
Trains the content-based model and the collaborative-filtering (SVD) model
on the full dataset, then saves both to `saved_models/` so they can be
reloaded instantly by main.py / app.py without retraining every time
(bonus: "save trained model for reuse").

Run directly:
    python train.py
"""

from pathlib import Path

from data_utils import load_data, ensure_models_dir
from model import ContentBasedRecommender, CollaborativeRecommender

CONTENT_MODEL_PATH = "content_model.joblib"
COLLAB_MODEL_PATH = "collab_model.surprise"


def train_all():
    ratings_df, items_df = load_data()
    models_dir = ensure_models_dir()

    print(f"Loaded {len(ratings_df)} ratings and {len(items_df)} items.")

    print("Training content-based model (TF-IDF + cosine similarity)...")
    content_model = ContentBasedRecommender(items_df)
    content_model.fit()
    content_model.save(models_dir / CONTENT_MODEL_PATH)
    print(f"  saved -> {models_dir / CONTENT_MODEL_PATH}")

    print("Training collaborative-filtering model (SVD)...")
    collab_model = CollaborativeRecommender()
    collab_model.fit(ratings_df)
    collab_model.save(models_dir / COLLAB_MODEL_PATH)
    print(f"  saved -> {models_dir / COLLAB_MODEL_PATH}")

    print("Done.")
    return content_model, collab_model, ratings_df, items_df


def load_trained(models_dir: Path = None):
    """Load previously-trained models, training fresh if missing."""
    from data_utils import MODELS_DIR
    models_dir = models_dir or MODELS_DIR
    content_path = models_dir / CONTENT_MODEL_PATH
    collab_path = models_dir / COLLAB_MODEL_PATH

    ratings_df, items_df = load_data()

    if content_path.exists():
        content_model = ContentBasedRecommender.load(content_path)
    else:
        content_model, _, _, _ = train_all()
        return load_trained(models_dir)

    if collab_path.exists():
        collab_model = CollaborativeRecommender.load(collab_path)
    else:
        _, collab_model, _, _ = train_all()
        return load_trained(models_dir)

    return content_model, collab_model, ratings_df, items_df


if __name__ == "__main__":
    train_all()
