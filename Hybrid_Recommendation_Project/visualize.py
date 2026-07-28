"""
visualize.py
------------
Optional (tech-stack bonus): a couple of quick exploratory charts using
matplotlib / seaborn. Not required for the recommender to work -- just
useful for a report / README screenshot.

Run:
    python visualize.py

Outputs PNGs into reports/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import load_data

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    ratings_df, items_df = load_data()

    sns.set_theme(style="whitegrid")

    # 1. Rating distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="rating", data=ratings_df, hue="rating", palette="viridis", legend=False)
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "rating_distribution.png", dpi=120)
    plt.close()

    # 2. Ratings per user (sparsity check)
    plt.figure(figsize=(6, 4))
    counts = ratings_df.groupby("user_id").size()
    sns.histplot(counts, bins=20, color="steelblue")
    plt.title("Number of ratings per user")
    plt.xlabel("Ratings given")
    plt.ylabel("Number of users")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ratings_per_user.png", dpi=120)
    plt.close()

    # 3. Average rating by genre
    exploded = items_df.assign(genres=items_df["genres"].str.split("|")).explode("genres")
    merged = ratings_df.merge(exploded[["item_id", "genres"]], on="item_id")
    genre_avg = merged.groupby("genres")["rating"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=genre_avg.values, y=genre_avg.index, hue=genre_avg.index,
                palette="magma", legend=False)
    plt.title("Average rating by genre")
    plt.xlabel("Average rating")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "avg_rating_by_genre.png", dpi=120)
    plt.close()

    print(f"Saved 3 charts to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
