"""
generate_dataset.py
--------------------
Generates a synthetic (but behaviourally realistic) movie-recommendation
dataset so the whole project runs end-to-end without needing an external
download.

Why synthetic data?
    - No external dataset link needed (works fully offline).
    - We can control the "signal" in the data: users have hidden genre
      preferences, so both content-based and collaborative filtering
      have real patterns to learn from (and the two methods will actually
      *disagree* sometimes -- which is exactly when a hybrid helps).

Output:
    data/items.csv    -> item_id, title, genres, tags, combined_features
    data/ratings.csv  -> user_id, item_id, rating, timestamp
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_ITEMS = 60
N_USERS = 300
MIN_RATINGS_PER_USER = 15
MAX_RATINGS_PER_USER = 45

GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror",
    "Mystery", "Romance", "SciFi", "Thriller", "Animation",
    "Documentary", "Crime", "Family", "Musical",
]

TAGS = [
    "dark", "emotional", "fast-paced", "witty", "epic", "twist-ending",
    "feel-good", "visually-stunning", "slow-burn", "underrated",
    "based-on-true-story", "cult-classic", "award-winning", "indie",
    "blockbuster",
]

ADJECTIVES = ["Lost", "Silent", "Broken", "Golden", "Last", "Hidden",
              "Distant", "Eternal", "Forgotten", "Midnight", "Crimson",
              "Shattered", "Frozen", "Sacred", "Wandering"]
NOUNS = ["Horizon", "Kingdom", "Journey", "Shadow", "Echo", "Legacy",
         "Storm", "Garden", "River", "City", "Dream", "Flame", "Empire",
         "Signal", "Harbor"]


def make_items(n_items: int) -> pd.DataFrame:
    rows = []
    used_titles = set()
    for item_id in range(1, n_items + 1):
        # invented (non-copyrighted) title
        while True:
            title = f"The {random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
            if title not in used_titles:
                used_titles.add(title)
                break

        n_genres = random.choice([1, 1, 2, 2, 3])
        item_genres = sorted(random.sample(GENRES, n_genres))

        n_tags = random.choice([2, 3, 3, 4])
        item_tags = sorted(random.sample(TAGS, n_tags))

        year = random.randint(1995, 2025)

        combined = " ".join(item_genres + item_tags + [str(year)])

        rows.append({
            "item_id": item_id,
            "title": title,
            "year": year,
            "genres": "|".join(item_genres),
            "tags": "|".join(item_tags),
            "combined_features": combined,
        })
    return pd.DataFrame(rows)


def make_ratings(items_df: pd.DataFrame, n_users: int) -> pd.DataFrame:
    rows = []
    base_ts = 1_700_000_000  # arbitrary unix timestamp anchor

    item_genre_lists = items_df["genres"].str.split("|").tolist()
    item_ids = items_df["item_id"].tolist()

    for user_id in range(1, n_users + 1):
        # each user has 2-3 "true" favourite genres (hidden taste profile)
        n_pref = random.choice([2, 2, 3])
        preferred_genres = set(random.sample(GENRES, n_pref))

        n_ratings = random.randint(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER)
        rated_items = random.sample(item_ids, min(n_ratings, len(item_ids)))

        for item_id in rated_items:
            idx = item_id - 1
            genres_for_item = set(item_genre_lists[idx])
            overlap = len(genres_for_item & preferred_genres)

            if overlap >= 2:
                mean_rating = 4.6
            elif overlap == 1:
                mean_rating = 3.7
            else:
                mean_rating = 2.4

            noisy = np.random.normal(loc=mean_rating, scale=0.7)
            rating = int(np.clip(round(noisy), 1, 5))

            ts = base_ts + random.randint(0, 60 * 60 * 24 * 365 * 3)
            rows.append({
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating,
                "timestamp": ts,
            })

    return pd.DataFrame(rows)


def main():
    out_dir = Path(__file__).resolve().parent
    items_df = make_items(N_ITEMS)
    ratings_df = make_ratings(items_df, N_USERS)

    items_df.to_csv(out_dir / "items.csv", index=False)
    ratings_df.to_csv(out_dir / "ratings.csv", index=False)

    print(f"Generated {len(items_df)} items -> {out_dir / 'items.csv'}")
    print(f"Generated {len(ratings_df)} ratings -> {out_dir / 'ratings.csv'}")
    print(f"Unique users: {ratings_df['user_id'].nunique()}")


if __name__ == "__main__":
    main()
