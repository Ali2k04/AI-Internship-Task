"""
ab_testing.py
-------------
A/B testing framework for comparing recommendation strategies.

Rather than faking click data, this evaluates strategies the way real
recommender-system teams do offline before running a live test:

  1. Split each user's interaction history by time: their most recent
     interactions become a held-out "test set", everything earlier is
     "train".
  2. Build a FeedEngine on the train set only.
  3. Deterministically assign each user to a group (A/B/C...), each
     mapped to a different strategy/alpha.
  4. Generate that user's feed using their group's strategy, then check
     how many of the recommended items the user *actually went on to
     interact with* in the held-out test set.
  5. Aggregate Precision@K and Recall@K per group.

Whichever group comes out on top on held-out data is the one you'd
promote to more real traffic in a live A/B test.
"""

import hashlib

import pandas as pd

from engine import FeedEngine

# Each group maps to a strategy + alpha the engine understands.
GROUP_CONFIG = {
    "A_content_only": {"strategy": "content", "alpha": 1.0},
    "B_collaborative_only": {"strategy": "collaborative", "alpha": 0.0},
    "C_hybrid": {"strategy": "hybrid", "alpha": 0.5},
}


def assign_group(user_id, groups=tuple(GROUP_CONFIG.keys())):
    """Deterministic hash-based bucketing so a user always lands in the same group."""
    h = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
    return groups[h % len(groups)]


def time_split(interactions_df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Per user, holds out the most recent `test_fraction` of interactions as the
    test set and returns (train_df, test_df).
    """
    interactions_df = interactions_df.copy()
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])

    train_parts, test_parts = [], []
    for user_id, group in interactions_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        n_test = max(1, int(len(group) * test_fraction))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def run_ab_evaluation(interactions_df, content_df, top_n=5, test_fraction=0.2, half_life_days=14.0):
    """
    Runs the full offline A/B evaluation and returns a per-group summary
    DataFrame with mean Precision@K / Recall@K, plus the detailed per-user
    results DataFrame.
    """
    train_df, test_df = time_split(interactions_df, test_fraction=test_fraction)

    # Reference time = latest timestamp in train, so decay is computed consistently
    reference_time = pd.to_datetime(train_df["timestamp"]).max()
    engine = FeedEngine(train_df, content_df, half_life_days=half_life_days, reference_time=reference_time)

    test_items_by_user = test_df.groupby("user_id")["content_id"].apply(set).to_dict()

    rows = []
    for user_id in test_items_by_user:
        if user_id not in engine.user_index:
            continue  # user had no train interactions (all went to test) -> can't recommend
        group = assign_group(user_id)
        config = GROUP_CONFIG[group]

        feed = engine.get_feed(
            user_id, top_n=top_n, strategy=config["strategy"], alpha=config["alpha"], exclude_seen=True
        )
        recommended_ids = {item["content_id"] for item in feed}
        relevant_ids = test_items_by_user[user_id]

        hits = len(recommended_ids & relevant_ids)
        precision = hits / top_n if top_n else 0.0
        recall = hits / len(relevant_ids) if relevant_ids else 0.0

        rows.append(
            {
                "user_id": user_id,
                "group": group,
                "strategy": config["strategy"],
                "hits": hits,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "n_recommended": len(recommended_ids),
                "n_relevant": len(relevant_ids),
            }
        )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        summary_df = pd.DataFrame(
            columns=["group", "strategy", "n_users", "avg_precision_at_k", "avg_recall_at_k"]
        )
    else:
        summary_df = (
            results_df.groupby(["group", "strategy"])
            .agg(
                n_users=("user_id", "count"),
                avg_precision_at_k=("precision_at_k", "mean"),
                avg_recall_at_k=("recall_at_k", "mean"),
            )
            .reset_index()
            .sort_values("avg_precision_at_k", ascending=False)
        )

    return summary_df, results_df, engine
