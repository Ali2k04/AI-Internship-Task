"""
generate_data.py
----------------
Creates realistic sample datasets for the Personalized Content Feed Engine:

  data/content_data.csv    -> content_id, title, category, tags, content_type
  data/user_behavior.csv   -> user_id, content_id, interaction_type, timestamp

The simulation gives each user a hidden "taste profile" (1-2 favourite
categories) so that the recommendation engine has real signal to learn from,
plus some random noise so it isn't trivially easy.

Run directly to (re)generate the CSVs:
    python generate_data.py
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORIES = {
    "Technology": ["ai", "gadgets", "programming", "startups"],
    "Sports": ["football", "cricket", "olympics", "fitness"],
    "Entertainment": ["movies", "music", "celebrity", "tv-shows"],
    "Health": ["nutrition", "mental-health", "fitness", "medicine"],
    "Finance": ["stocks", "crypto", "budgeting", "startups"],
    "Education": ["online-courses", "programming", "career", "exams"],
}

CONTENT_TYPES = ["article", "video", "short-post"]
INTERACTION_TYPES = ["view", "click", "like", "search"]
# Rough probability of each interaction type occurring (views most common)
INTERACTION_PROBS = [0.55, 0.25, 0.12, 0.08]

N_CONTENT = 60
N_USERS = 30
N_DAYS_HISTORY = 30
MIN_INTERACTIONS_PER_USER = 15
MAX_INTERACTIONS_PER_USER = 40


def generate_content(n_content=N_CONTENT):
    rows = []
    cid = 1
    categories = list(CATEGORIES.keys())
    for i in range(n_content):
        category = categories[i % len(categories)]
        tag_pool = CATEGORIES[category]
        n_tags = random.randint(1, min(3, len(tag_pool)))
        tags = "|".join(random.sample(tag_pool, n_tags))
        content_type = random.choice(CONTENT_TYPES)
        title = f"{category} {content_type.title()} #{cid}"
        rows.append(
            {
                "content_id": cid,
                "title": title,
                "category": category,
                "tags": tags,
                "content_type": content_type,
            }
        )
        cid += 1
    return pd.DataFrame(rows)


def generate_behavior(content_df, n_users=N_USERS):
    categories = content_df["category"].unique().tolist()
    rows = []
    now = datetime.now()

    for user_id in range(1, n_users + 1):
        # Each user has 1-2 favourite categories (their "taste")
        n_fav = random.choice([1, 1, 2])
        fav_categories = random.sample(categories, n_fav)

        n_interactions = random.randint(MIN_INTERACTIONS_PER_USER, MAX_INTERACTIONS_PER_USER)
        for _ in range(n_interactions):
            # 80% chance the interaction is with a favourite-category item, 20% exploration/noise
            if random.random() < 0.8:
                candidate_content = content_df[content_df["category"].isin(fav_categories)]
            else:
                candidate_content = content_df
            content_row = candidate_content.sample(1).iloc[0]

            interaction_type = np.random.choice(INTERACTION_TYPES, p=INTERACTION_PROBS)

            # More recent timestamps are slightly more likely (recency bias),
            # which lets the time-decay feature show a visible effect.
            days_ago = np.random.exponential(scale=N_DAYS_HISTORY / 3)
            days_ago = min(days_ago, N_DAYS_HISTORY)
            timestamp = now - timedelta(
                days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59)
            )

            rows.append(
                {
                    "user_id": user_id,
                    "content_id": int(content_row["content_id"]),
                    "interaction_type": interaction_type,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    behavior_df = pd.DataFrame(rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return behavior_df


def main():
    import os

    os.makedirs("data", exist_ok=True)

    content_df = generate_content()
    behavior_df = generate_behavior(content_df)

    content_df.to_csv("data/content_data.csv", index=False)
    behavior_df.to_csv("data/user_behavior.csv", index=False)

    print(f"Generated {len(content_df)} content items -> data/content_data.csv")
    print(f"Generated {len(behavior_df)} interactions across {behavior_df['user_id'].nunique()} users "
          f"-> data/user_behavior.csv")


if __name__ == "__main__":
    main()
