"""
main.py
-------
End-to-end demo of the Personalized Content Feed Engine.

Run:
    python main.py

What it does:
  1. Generates sample data if data/*.csv don't exist yet
  2. Loads everything into a SQLite database (data/feed_engine.db)
  3. Builds the FeedEngine (content-based + collaborative + time-decay)
  4. Prints a personalized feed for a few sample users
  5. Simulates a brand-new interaction and shows the feed adapt in real time
  6. Runs the offline A/B test comparing content-only vs. collaborative-only
     vs. hybrid strategies and prints the results table
  7. Persists user profiles and feed impressions to the database
"""

import os

import pandas as pd

import database
from ab_testing import GROUP_CONFIG, assign_group, run_ab_evaluation
from engine import FeedEngine

DATA_DIR = "data"
CONTENT_CSV = os.path.join(DATA_DIR, "content_data.csv")
BEHAVIOR_CSV = os.path.join(DATA_DIR, "user_behavior.csv")


def ensure_data_exists():
    if not (os.path.exists(CONTENT_CSV) and os.path.exists(BEHAVIOR_CSV)):
        print("Sample data not found -- generating it now...")
        import generate_data

        generate_data.main()


def print_feed(title, feed):
    print(f"\n{title}")
    if not feed:
        print("  (no recommendations available)")
    for item in feed:
        print(f"  - [{item['content_id']:>3}] {item['title']:<32} "
              f"({item['category']:<13}) score={item['score']}")


def main():
    ensure_data_exists()

    content_df = pd.read_csv(CONTENT_CSV)
    behavior_df = pd.read_csv(BEHAVIOR_CSV)

    # --- 1. Persist raw interactions to SQLite -----------------------------
    database.init_db()
    database.load_interactions_csv(BEHAVIOR_CSV)
    print(f"Loaded {len(behavior_df)} interactions and {len(content_df)} content items into "
          f"{database.DB_PATH}")

    # --- 2. Build the live engine over ALL data -----------------------------
    engine = FeedEngine(behavior_df, content_df, half_life_days=14.0)

    sample_users = behavior_df["user_id"].drop_duplicates().head(3).tolist()

    print("\n" + "=" * 60)
    print("PERSONALIZED FEEDS (hybrid strategy)")
    print("=" * 60)
    for user_id in sample_users:
        group = assign_group(user_id)
        database.set_ab_assignment(user_id, group)
        config = GROUP_CONFIG[group]

        feed = engine.get_feed(user_id, top_n=5, strategy=config["strategy"], alpha=config["alpha"])
        print_feed(f"User {user_id}  (A/B group: {group} -> strategy={config['strategy']})", feed)
        database.log_feed_impressions(user_id, feed, group, config["strategy"])

        # keep an aggregated profile snapshot in the DB too
        database.save_user_profile(user_id, engine.get_user_profile(user_id))

    # --- 3. Real-time adaptation demo --------------------------------------
    # A brand-new user (cold start) makes the adaptation obvious: with zero
    # history there's nothing to recommend from, but after a handful of live
    # interactions the feed immediately reflects the new taste.
    demo_user = "new_user_demo"
    print("\n" + "=" * 60)
    print(f"REAL-TIME UPDATE DEMO -- brand-new user '{demo_user}' (cold start)")
    print("=" * 60)

    before_feed = engine.get_feed(demo_user, top_n=5, strategy="content")
    print_feed("Feed BEFORE any interactions (cold start)", before_feed)

    tech_items = content_df[content_df["category"] == "Technology"].head(3)
    print(f"\n>>> Simulating: '{demo_user}' just viewed, clicked, and liked "
          f"{len(tech_items)} Technology items right now...")
    for _, row in tech_items.iterrows():
        for interaction_type in ["view", "click", "like"]:
            engine.add_interaction(demo_user, int(row["content_id"]), interaction_type)
            database.insert_interaction(
                demo_user, int(row["content_id"]), interaction_type, pd.Timestamp.now()
            )

    after_feed = engine.get_feed(demo_user, top_n=5, strategy="content")
    print_feed("Feed AFTER live interactions (updated instantly, no restart needed)", after_feed)

    # --- 4. Offline A/B evaluation ------------------------------------------
    print("\n" + "=" * 60)
    print("A/B TEST: content-only vs. collaborative-only vs. hybrid")
    print("(evaluated offline on held-out recent interactions per user)")
    print("=" * 60)

    summary_df, results_df, _eval_engine = run_ab_evaluation(
        behavior_df, content_df, top_n=5, test_fraction=0.2
    )
    if summary_df.empty:
        print("Not enough held-out data to evaluate (try generating more interactions).")
    else:
        print(summary_df.to_string(index=False))
        best = summary_df.iloc[0]
        print(f"\nBest performing group on held-out data: {best['group']} "
              f"(strategy={best['strategy']}, avg precision@5={best['avg_precision_at_k']:.3f})")

    print(f"\nAll interactions, profiles, A/B assignments, and feed impressions are stored in "
          f"{database.DB_PATH}")
    print("Run 'streamlit run dashboard.py' for an interactive view.")


if __name__ == "__main__":
    main()
