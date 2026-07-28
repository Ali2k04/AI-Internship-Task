"""
engine.py
---------
Core recommendation logic for the Personalized Content Feed Engine.

Implements:
  - Weighted interaction scoring (view / click / like / search)
  - Time-decay so recent activity matters more than old activity
  - Content-based filtering (cosine similarity over category + tags)
  - Collaborative filtering (matrix factorization via NMF over the
    user-item interaction matrix)
  - A hybrid scorer that blends both signals with a tunable alpha
  - Real-time incremental updates (add_interaction) so the feed adapts
    immediately to new behavior without a full offline rebuild every time
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "like": 3.0,
    "search": 1.5,
}


class FeedEngine:
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        content_df: pd.DataFrame,
        half_life_days: float = 14.0,
        n_cf_factors: int = 8,
        reference_time: datetime | None = None,
    ):
        """
        interactions_df columns: user_id, content_id, interaction_type, timestamp
        content_df columns:      content_id, title, category, tags, content_type
        half_life_days:          how many days until an interaction's weight halves
        n_cf_factors:            latent dimensions for the collaborative filtering model
        reference_time:          "now" for decay purposes (defaults to current time;
                                  pass a fixed timestamp for reproducible offline evaluation)
        """
        self.content_df = content_df.reset_index(drop=True)
        self.content_ids = self.content_df["content_id"].tolist()
        self.content_index = {cid: i for i, cid in enumerate(self.content_ids)}
        self.half_life_days = half_life_days
        self.n_cf_factors = n_cf_factors
        self.reference_time = pd.Timestamp(reference_time) if reference_time else pd.Timestamp.now()

        self.interactions = interactions_df.copy()
        self._pending_updates = 0

        self._prepare_content_similarity()
        self._rebuild_from_interactions()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _prepare_content_similarity(self):
        """Builds an item-item similarity matrix from content metadata (category + tags)."""
        cat_dummies = pd.get_dummies(self.content_df["category"])
        tag_dummies = self.content_df["tags"].fillna("").str.get_dummies(sep="|")
        features = pd.concat([cat_dummies, tag_dummies], axis=1)
        self.content_similarity = cosine_similarity(features.values)

    def _time_decay(self, timestamps) -> pd.Series:
        """Exponential half-life decay: weight halves every `half_life_days`."""
        ts = pd.to_datetime(pd.Series(timestamps).values)
        age_days = (self.reference_time - ts).total_seconds() / 86400.0
        age_days = np.clip(age_days, 0, None)
        return pd.Series(0.5 ** (age_days / self.half_life_days))

    def _rebuild_from_interactions(self):
        """Full rebuild of the user-item matrix and both models from the interaction log."""
        df = self.interactions.copy()
        df["base_weight"] = df["interaction_type"].map(INTERACTION_WEIGHTS).fillna(1.0)
        df["decay"] = self._time_decay(df["timestamp"]).values
        df["score"] = df["base_weight"] * df["decay"]

        self.user_ids = sorted(df["user_id"].unique().tolist())
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}

        n_users = len(self.user_ids)
        n_items = len(self.content_ids)
        matrix = np.zeros((n_users, n_items))

        grouped = df.groupby(["user_id", "content_id"])["score"].sum().reset_index()
        for _, row in grouped.iterrows():
            u = self.user_index.get(row["user_id"])
            i = self.content_index.get(row["content_id"])
            if u is not None and i is not None:
                matrix[u, i] = row["score"]

        self.user_item_matrix = matrix
        self._fit_collaborative_model()

    def _fit_collaborative_model(self):
        """Fits an NMF matrix-factorization model over the user-item matrix."""
        if self.user_item_matrix.shape[0] < 2 or self.user_item_matrix.sum() == 0:
            self.user_factors = None
            self.item_factors = None
            return
        k = max(1, min(self.n_cf_factors, min(self.user_item_matrix.shape) - 1))
        model = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=42)
        self.user_factors = model.fit_transform(self.user_item_matrix)
        self.item_factors = model.components_

    # ------------------------------------------------------------------ #
    # Real-time updates
    # ------------------------------------------------------------------ #
    def add_interaction(self, user_id, content_id, interaction_type, timestamp=None, refit_every=5):
        """
        Records a brand-new interaction and folds it into the live model immediately
        (the user-item matrix updates on every call, so content-based scores and
        already-seen exclusion reflect it instantly). The collaborative-filtering
        factorization is comparatively expensive, so it is refit every `refit_every`
        new interactions rather than on every single call -- a common production
        pattern (incremental signal now, periodic full refresh in the background).
        """
        timestamp = timestamp or datetime.now()
        new_row = {
            "user_id": user_id,
            "content_id": content_id,
            "interaction_type": interaction_type,
            "timestamp": timestamp,
        }
        self.interactions = pd.concat(
            [self.interactions, pd.DataFrame([new_row])], ignore_index=True
        )

        weight = INTERACTION_WEIGHTS.get(interaction_type, 1.0)
        decay = self._time_decay([timestamp]).iloc[0]
        score = weight * decay

        if user_id not in self.user_index:
            self.user_index[user_id] = len(self.user_ids)
            self.user_ids.append(user_id)
            self.user_item_matrix = np.vstack(
                [self.user_item_matrix, np.zeros((1, self.user_item_matrix.shape[1]))]
            )

        if content_id in self.content_index:
            u = self.user_index[user_id]
            i = self.content_index[content_id]
            self.user_item_matrix[u, i] += score

        self._pending_updates += 1
        if self._pending_updates >= refit_every:
            self._fit_collaborative_model()
            self._pending_updates = 0

    def force_refit(self):
        """Manually trigger a full collaborative-filtering refit (e.g. on a schedule/cron)."""
        self._fit_collaborative_model()
        self._pending_updates = 0

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _content_based_scores(self, user_id) -> np.ndarray:
        n_items = len(self.content_ids)
        scores = np.zeros(n_items)
        if user_id not in self.user_index:
            return scores
        u = self.user_index[user_id]
        user_vector = self.user_item_matrix[u]
        interacted = np.nonzero(user_vector)[0]
        for i in interacted:
            scores += user_vector[i] * self.content_similarity[i]
        return scores

    def _collaborative_scores(self, user_id) -> np.ndarray:
        n_items = len(self.content_ids)
        if self.user_factors is None or user_id not in self.user_index:
            return np.zeros(n_items)
        u = self.user_index[user_id]
        return self.user_factors[u].dot(self.item_factors)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0 or (arr.max() - arr.min()) < 1e-9:
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    def get_user_profile(self, user_id) -> dict:
        """Returns {content_id: aggregated_score} for whatever the user has interacted with."""
        if user_id not in self.user_index:
            return {}
        u = self.user_index[user_id]
        vector = self.user_item_matrix[u]
        return {
            self.content_ids[i]: float(vector[i])
            for i in np.nonzero(vector)[0]
        }

    def get_feed(self, user_id, top_n=5, strategy="hybrid", alpha=0.5, exclude_seen=True):
        """
        strategy: "content"       -> pure content-based (metadata similarity)
                  "collaborative" -> pure collaborative filtering (NMF latent factors)
                  "hybrid"        -> alpha * content + (1 - alpha) * collaborative
        Returns a list of dicts: content_id, title, category, score (ranked, highest first).
        """
        content_scores = self._normalize(self._content_based_scores(user_id))
        cf_scores = self._normalize(self._collaborative_scores(user_id))

        if strategy == "content":
            final = content_scores
        elif strategy == "collaborative":
            final = cf_scores
        elif strategy == "hybrid":
            final = alpha * content_scores + (1 - alpha) * cf_scores
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        final = final.copy()
        if exclude_seen and user_id in self.user_index:
            seen = np.nonzero(self.user_item_matrix[self.user_index[user_id]])[0]
            final[seen] = -np.inf

        ranked_idx = np.argsort(final)[::-1]
        results = []
        for idx in ranked_idx:
            if final[idx] == -np.inf:
                continue
            if len(results) >= top_n:
                break
            row = self.content_df.iloc[idx]
            results.append(
                {
                    "content_id": int(row["content_id"]),
                    "title": row["title"],
                    "category": row["category"],
                    "score": round(float(final[idx]), 4),
                }
            )
        return results
