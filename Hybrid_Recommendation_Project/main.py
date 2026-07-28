"""
main.py
-------
Command-line demo of the whole pipeline:

    1. Load (or generate) the dataset.
    2. Load trained models from saved_models/, training + saving them the
       first time this is run.
    3. Print Top-N recommendations for a few example users, including:
         - a normal existing user
         - a cold-start "brand-new" user (identified only by favourite
           genres, no ratings at all)
    4. Run the RMSE / Precision@K evaluation.

Usage:
    python main.py                  # demo with default user ids
    python main.py --user 101        # recommend for a specific user_id
    python main.py --alpha 0.5       # override the collaborative/content weight
    python main.py --evaluate        # also run evaluate.py's metrics
"""

import argparse

from train import load_trained
from model import HybridRecommender


def print_recommendations(title, recs):
    print(f"\n{title}")
    print("-" * len(title))
    if not recs:
        print("  (no recommendations)")
        return
    for r in recs:
        print(f"  Item {r['item_id']:>3}  {r['title']:<28} "
              f"[{r['genres']:<25}]  score={r['score']:.2f}   ({r['reason']})")


def main():
    parser = argparse.ArgumentParser(description="Hybrid recommender demo")
    parser.add_argument("--user", type=int, default=None,
                         help="user_id to generate recommendations for")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7,
                         help="weight on collaborative score (0-1). Default 0.7 "
                              "= 70%% collaborative / 30%% content.")
    parser.add_argument("--evaluate", action="store_true",
                         help="also compute RMSE / Precision@K")
    args = parser.parse_args()

    print("Loading / training models (this uses saved artifacts if present)...")
    content_model, collab_model, ratings_df, items_df = load_trained()

    hybrid = HybridRecommender(content_model, collab_model, ratings_df, default_alpha=args.alpha)

    if args.user is not None:
        recs = hybrid.recommend(args.user, top_n=args.top_n, alpha=args.alpha)
        print_recommendations(f"Recommended items for User {args.user} (alpha={args.alpha})", recs)
    else:
        sample_existing_user = int(ratings_df["user_id"].iloc[0])
        recs = hybrid.recommend(sample_existing_user, top_n=args.top_n, alpha=args.alpha)
        print_recommendations(
            f"Recommended items for existing User {sample_existing_user} (alpha={args.alpha})",
            recs,
        )

        cold_user_id = int(ratings_df["user_id"].max()) + 999  # guaranteed unseen id
        cold_recs = hybrid.recommend(
            cold_user_id, top_n=args.top_n, alpha=args.alpha,
            new_user_genres=["SciFi", "Adventure"],
        )
        print_recommendations(
            f"Cold-start User {cold_user_id} who says they like SciFi/Adventure",
            cold_recs,
        )

        cold_user_no_profile = int(ratings_df["user_id"].max()) + 1000
        cold_recs_pop = hybrid.recommend(cold_user_no_profile, top_n=args.top_n, alpha=args.alpha)
        print_recommendations(
            f"Cold-start User {cold_user_no_profile} with no info at all (popularity fallback)",
            cold_recs_pop,
        )

    if args.evaluate:
        print("\nRunning evaluation (RMSE / Precision@K)...")
        import evaluate
        evaluate.main()


if __name__ == "__main__":
    main()
