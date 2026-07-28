"""
evaluate.py
-----------
Bonus feature: "Evaluate using RMSE / Precision@K".

- RMSE: standard collaborative-filtering accuracy metric, computed on a
  held-out test split (never seen during training).
- Precision@K: computed two ways so you can see whether the hybrid
  actually helps --
    1. collaborative-only ranking
    2. hybrid ranking (content + collaborative)
  for each user, using their held-out test items as the candidate pool.

Run directly:
    python evaluate.py
"""

from collections import defaultdict

import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from data_utils import load_data
from model import ContentBasedRecommender, CollaborativeRecommender, content_score_to_rating


LIKE_THRESHOLD = 3.5  # rating >= this counts as "relevant" for precision@k
K = 5
TEST_SIZE = 0.2


def rmse_eval(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=TEST_SIZE, random_state=42)

    algo = SVD(n_factors=50, n_epochs=30, random_state=42)
    algo.fit(trainset)
    predictions = algo.predict_batch if hasattr(algo, "predict_batch") else None
    preds = [algo.predict(uid, iid, r) for (uid, iid, r) in testset]
    rmse = accuracy.rmse(preds, verbose=False)
    return rmse, trainset, testset, algo


def precision_recall_at_k(predictions, k=K, threshold=LIKE_THRESHOLD):
    """Standard Surprise-FAQ recipe, grouped by user."""
    user_est_true = defaultdict(list)
    for uid, _iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for (_, true_r) in ratings)
        n_rec_k = min(k, len(ratings))
        top_k = ratings[:n_rec_k]
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = np.mean(list(precisions.values())) if precisions else 0.0
    avg_recall = np.mean(list(recalls.values())) if recalls else 0.0
    return avg_precision, avg_recall


def hybrid_precision_at_k(testset, algo, content_model, ratings_df, alpha=0.7, k=K, threshold=LIKE_THRESHOLD):
    """
    Re-rank each user's held-out test items using the hybrid score
    (collaborative estimate blended with content similarity to that
    user's profile), then measure precision@k the same way.
    """
    by_user = defaultdict(list)
    for uid, iid, true_r in testset:
        by_user[uid].append((iid, true_r))

    precisions = []
    for uid, items in by_user.items():
        profile = content_model.build_user_profile(uid, ratings_df, like_threshold=threshold)

        scored = []
        for iid, true_r in items:
            collab_est = algo.predict(uid, iid).est
            idx = content_model.item_id_to_idx.get(iid)
            if profile is not None and idx is not None:
                item_vec = np.asarray(content_model.tfidf_matrix[idx].todense()).flatten()
                denom = np.linalg.norm(item_vec) * np.linalg.norm(profile)
                content_sim = float(item_vec.dot(profile) / denom) if denom > 0 else 0.0
            else:
                content_sim = 0.0
            content_rating = content_score_to_rating(content_sim)
            final = alpha * collab_est + (1 - alpha) * content_rating
            scored.append((final, true_r))

        scored.sort(key=lambda x: x[0], reverse=True)
        n_rec_k = min(k, len(scored))
        top_k = scored[:n_rec_k]
        n_rel_and_rec_k = sum(true_r >= threshold for (_, true_r) in top_k)
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k else 0)

    return float(np.mean(precisions)) if precisions else 0.0


def main():
    ratings_df, items_df = load_data()

    print("Evaluating RMSE on a held-out 80/20 split...")
    rmse, trainset, testset, algo = rmse_eval(ratings_df)
    print(f"  RMSE (collaborative SVD): {rmse:.4f}")

    preds = [algo.predict(uid, iid, r) for (uid, iid, r) in testset]
    prec_collab, rec_collab = precision_recall_at_k(preds, k=K)
    print(f"  Precision@{K} (collaborative only): {prec_collab:.4f}")
    print(f"  Recall@{K}    (collaborative only): {rec_collab:.4f}")

    print("Fitting content-based model for hybrid evaluation...")
    # fit content model on train-split ratings so profiles don't leak test info
    train_ratings = ratings_df.iloc[[]]  # placeholder, profiles are rebuilt from full history below
    content_model = ContentBasedRecommender(items_df)
    content_model.fit()

    # Build profiles from the *training* portion of each user's history only.
    train_pairs = {(trainset.to_raw_uid(u), trainset.to_raw_iid(i))
                    for (u, i, _r) in trainset.all_ratings()}
    train_ratings_df = ratings_df[
        ratings_df.apply(lambda r: (r["user_id"], r["item_id"]) in train_pairs, axis=1)
    ]

    prec_hybrid = hybrid_precision_at_k(testset, algo, content_model, train_ratings_df, alpha=0.7, k=K)
    print(f"  Precision@{K} (hybrid, alpha=0.7):   {prec_hybrid:.4f}")

    print("\nSummary")
    print("-------")
    print(f"RMSE:                          {rmse:.4f}")
    print(f"Precision@{K} collaborative-only: {prec_collab:.4f}")
    print(f"Precision@{K} hybrid:             {prec_hybrid:.4f}")


if __name__ == "__main__":
    main()
