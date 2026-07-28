"""
model.py
--------
Three classes:

    ContentBasedRecommender  - TF-IDF + cosine similarity over item features,
                                plus a "user profile" vector built from the
                                items a user has liked (bonus: user-profile
                                based recommendations).

    CollaborativeRecommender - thin wrapper around Surprise's SVD
                                (matrix factorization) model.

    HybridRecommender         - combines both with an adjustable weight,
                                and falls back gracefully for cold-start
                                users / items.

Both individual scores are normalised onto the same 1-5 rating scale
before being combined -- this matters, because averaging a raw cosine
similarity (0-1) directly with a predicted rating (1-5) as in a naive
implementation quietly biases every score toward the collaborative
number. Here `content_score_to_rating()` fixes that.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise import dump as surprise_dump


# --------------------------------------------------------------------------
# Content-based
# --------------------------------------------------------------------------
class ContentBasedRecommender:
    def __init__(self, items_df: pd.DataFrame, feature_col: str = "combined_features"):
        self.items_df = items_df.reset_index(drop=True)
        self.feature_col = feature_col
        self.item_id_to_idx = {
            item_id: idx for idx, item_id in enumerate(self.items_df["item_id"])
        }
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None

    def fit(self):
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.items_df[self.feature_col])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)
        return self

    def similar_items(self, item_id: int, top_n: int = 5):
        """Item-to-item similarity: 'because you liked X'."""
        idx = self.item_id_to_idx.get(item_id)
        if idx is None:
            return []
        sims = list(enumerate(self.cosine_sim[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        sims = [s for s in sims if s[0] != idx][:top_n]
        return [
            (self.items_df.iloc[i]["item_id"], self.items_df.iloc[i]["title"], score)
            for i, score in sims
        ]

    def build_user_profile(self, user_id: int, ratings_df: pd.DataFrame, like_threshold: float = 3.5):
        """
        Build a taste vector for a user: the (rating-weighted) average of the
        TF-IDF vectors of items they rated at/above `like_threshold`.
        This is the "user profile based recommendation" bonus feature --
        it works even for a user with zero collaborative-filtering history,
        as long as we know a few items they liked.
        """
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= like_threshold]
        if liked.empty:
            return None

        idxs, weights = [], []
        for _, row in liked.iterrows():
            idx = self.item_id_to_idx.get(row["item_id"])
            if idx is not None:
                idxs.append(idx)
                weights.append(row["rating"])

        if not idxs:
            return None

        weights = np.array(weights, dtype=float)
        vectors = self.tfidf_matrix[idxs]
        profile = np.asarray(vectors.T.dot(weights) / weights.sum()).flatten()
        return profile

    def profile_from_genre_list(self, genre_list):
        """
        Build a synthetic 'profile' vector directly from a list of genres a
        brand-new user says they like (cold-start entry point when there is
        no ratings history at all).
        """
        pseudo_doc = " ".join(genre_list)
        vec = self.tfidf.transform([pseudo_doc])
        return np.asarray(vec.todense()).flatten()

    def score_all_items_for_profile(self, profile_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of a user profile vector against every item."""
        if profile_vector is None or np.linalg.norm(profile_vector) == 0:
            return np.zeros(len(self.items_df))
        item_matrix = np.asarray(self.tfidf_matrix.todense())
        norms = np.linalg.norm(item_matrix, axis=1) * np.linalg.norm(profile_vector)
        norms[norms == 0] = 1e-9
        sims = item_matrix.dot(profile_vector) / norms
        return sims  # range roughly [0, 1]

    def save(self, path: Path):
        joblib.dump(
            {
                "tfidf": self.tfidf,
                "tfidf_matrix": self.tfidf_matrix,
                "cosine_sim": self.cosine_sim,
                "item_id_to_idx": self.item_id_to_idx,
                "items_df": self.items_df,
                "feature_col": self.feature_col,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path):
        state = joblib.load(path)
        obj = cls(state["items_df"], state["feature_col"])
        obj.tfidf = state["tfidf"]
        obj.tfidf_matrix = state["tfidf_matrix"]
        obj.cosine_sim = state["cosine_sim"]
        obj.item_id_to_idx = state["item_id_to_idx"]
        return obj


def content_score_to_rating(score: float) -> float:
    """Map a 0..1 cosine similarity onto the 1..5 rating scale."""
    score = max(0.0, min(1.0, score))
    return 1.0 + 4.0 * score


# --------------------------------------------------------------------------
# Collaborative filtering (Surprise SVD)
# --------------------------------------------------------------------------
class CollaborativeRecommender:
    def __init__(self, rating_scale=(1, 5)):
        self.rating_scale = rating_scale
        self.model = SVD(n_factors=50, n_epochs=30, random_state=42)
        self.trainset = None
        self.global_mean = None
        self.known_users = set()
        self.known_items = set()

    def fit(self, ratings_df: pd.DataFrame, trainset=None):
        if trainset is None:
            reader = Reader(rating_scale=self.rating_scale)
            data = Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)
            trainset = data.build_full_trainset()
        self.model.fit(trainset)
        self.trainset = trainset
        self.global_mean = trainset.global_mean
        self.known_users = set(ratings_df["user_id"].unique())
        self.known_items = set(ratings_df["item_id"].unique())
        return self

    def predict(self, user_id, item_id) -> float:
        return self.model.predict(user_id, item_id).est

    def is_known_user(self, user_id) -> bool:
        return user_id in self.known_users

    def is_known_item(self, item_id) -> bool:
        return item_id in self.known_items

    def save(self, path: Path):
        surprise_dump.dump(str(path), algo=self.model)
        joblib.dump(
            {
                "known_users": self.known_users,
                "known_items": self.known_items,
                "global_mean": self.global_mean,
                "rating_scale": self.rating_scale,
            },
            str(path) + ".meta",
        )

    @classmethod
    def load(cls, path: Path):
        _, algo = surprise_dump.load(str(path))
        meta = joblib.load(str(path) + ".meta")
        obj = cls(meta["rating_scale"])
        obj.model = algo
        obj.known_users = meta["known_users"]
        obj.known_items = meta["known_items"]
        obj.global_mean = meta["global_mean"]
        return obj


# --------------------------------------------------------------------------
# Hybrid
# --------------------------------------------------------------------------
class HybridRecommender:
    """
    final_score = alpha * collaborative_estimate + (1 - alpha) * content_estimate

    Default alpha = 0.7  -> 70% collaborative / 30% content (bonus request).

    Cold-start handling:
      - Brand-new user, no ratings at all:
            -> pure content-based (from stated genre preferences) if given,
               otherwise falls back to global popularity.
      - Existing user, but scoring an item with (almost) no ratings yet:
            -> pure content-based score for that item (collaborative signal
               for an unrated item is meaningless, so alpha is temporarily
               forced to 0 for that single item).
    """

    def __init__(self, content_model: ContentBasedRecommender,
                 collab_model: CollaborativeRecommender,
                 ratings_df: pd.DataFrame,
                 default_alpha: float = 0.7):
        self.content_model = content_model
        self.collab_model = collab_model
        self.ratings_df = ratings_df
        self.default_alpha = default_alpha
        self.item_rating_counts = ratings_df.groupby("item_id").size().to_dict()
        self.popularity = (
            ratings_df.groupby("item_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )

    # -- scoring for a single (user, item) pair -------------------------------
    def score(self, user_id, item_id, alpha=None, user_profile_vector=None):
        alpha = self.default_alpha if alpha is None else alpha

        user_known = self.collab_model.is_known_user(user_id)
        item_known = self.collab_model.is_known_item(item_id)
        item_rating_count = self.item_rating_counts.get(item_id, 0)

        # content score (0..1 -> 1..5)
        idx = self.content_model.item_id_to_idx.get(item_id)
        if user_profile_vector is not None and idx is not None:
            item_vector = np.asarray(self.content_model.tfidf_matrix[idx].todense()).flatten()
            denom = (np.linalg.norm(item_vector) * np.linalg.norm(user_profile_vector))
            content_sim = float(item_vector.dot(user_profile_vector) / denom) if denom > 0 else 0.0
        else:
            content_sim = 0.0
        content_rating = content_score_to_rating(content_sim)

        # Cold-start: unknown user -> content only (or popularity fallback)
        if not user_known:
            if user_profile_vector is not None:
                return content_rating, "content-only (new user)"
            pop_row = self.popularity[self.popularity["item_id"] == item_id]
            fallback = float(pop_row["mean"].iloc[0]) if not pop_row.empty else self.collab_model.global_mean
            return fallback, "popularity fallback (new user, no profile)"

        # Cold-start: item barely/never rated -> content only for this item
        if not item_known or item_rating_count < 3:
            return content_rating, "content-only (cold item)"

        # Normal case: blended score
        collab_est = self.collab_model.predict(user_id, item_id)
        final = alpha * collab_est + (1 - alpha) * content_rating
        return final, f"hybrid (alpha={alpha})"

    # -- top-N recommendations -------------------------------------------------
    def recommend(self, user_id, top_n=5, alpha=None, exclude_rated=True,
                   new_user_genres=None):
        """
        new_user_genres: optional list of genre strings, used only when
        user_id has no ratings history at all (cold-start entry point).
        """
        items_df = self.content_model.items_df

        user_profile_vector = self.content_model.build_user_profile(user_id, self.ratings_df)
        if user_profile_vector is None and new_user_genres:
            user_profile_vector = self.content_model.profile_from_genre_list(new_user_genres)

        already_rated = set()
        if exclude_rated:
            already_rated = set(
                self.ratings_df.loc[self.ratings_df["user_id"] == user_id, "item_id"]
            )

        results = []
        for item_id in items_df["item_id"]:
            if item_id in already_rated:
                continue
            score, reason = self.score(user_id, item_id, alpha=alpha,
                                        user_profile_vector=user_profile_vector)
            results.append((item_id, score, reason))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_n]

        out = []
        for item_id, score, reason in top:
            title = items_df.loc[items_df["item_id"] == item_id, "title"].iloc[0]
            genres = items_df.loc[items_df["item_id"] == item_id, "genres"].iloc[0]
            out.append({
                "item_id": int(item_id),
                "title": title,
                "genres": genres,
                "score": round(float(score), 3),
                "reason": reason,
            })
        return out
