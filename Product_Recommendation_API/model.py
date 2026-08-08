"""
model.py
Recommendation engine with two strategies:

1. Content-based filtering — TF-IDF over product `features` text,
   ranked by cosine similarity. Used for "products similar to X".

2. User-based collaborative filtering — builds a user-item rating
   matrix from interactions, finds the most similar users (cosine
   similarity over rating vectors), and recommends items those
   similar users rated highly that the target user hasn't seen yet.

Both matrices are built once at startup and can be refreshed via
`refresh()` after new interactions are added.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import database as db


class RecommendationEngine:
    def __init__(self):
        self.products_df = pd.DataFrame()
        self.content_sim = None
        self.product_index = {}  # product_id -> row index

        self.interactions_df = pd.DataFrame()
        self.user_item_matrix = pd.DataFrame()
        self.user_sim = pd.DataFrame()

        self.refresh()

    # ------------------------------------------------------------------
    # Build / rebuild the underlying similarity matrices
    # ------------------------------------------------------------------
    def refresh(self):
        self._build_content_model()
        self._build_collaborative_model()

    def _build_content_model(self):
        self.products_df = db.get_products_dataframe()
        if self.products_df.empty:
            self.content_sim = None
            self.product_index = {}
            return

        self.products_df = self.products_df.reset_index(drop=True)
        self.product_index = {
            pid: idx for idx, pid in enumerate(self.products_df["product_id"])
        }

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.products_df["features"].fillna(""))
        self.content_sim = cosine_similarity(tfidf_matrix)

    def _build_collaborative_model(self):
        self.interactions_df = db.get_interactions_dataframe()
        if self.interactions_df.empty:
            self.user_item_matrix = pd.DataFrame()
            self.user_sim = pd.DataFrame()
            return

        self.user_item_matrix = self.interactions_df.pivot_table(
            index="user_id", columns="product_id", values="rating", aggfunc="mean"
        ).fillna(0)

        if self.user_item_matrix.shape[0] < 2:
            self.user_sim = pd.DataFrame()
            return

        sim = cosine_similarity(self.user_item_matrix)
        self.user_sim = pd.DataFrame(
            sim, index=self.user_item_matrix.index, columns=self.user_item_matrix.index
        )

    # ------------------------------------------------------------------
    # Content-based: "products similar to this product"
    # ------------------------------------------------------------------
    def recommend_by_product(
        self,
        product_id: int,
        top_n: int = 5,
        category: str = None,
        min_price: float = None,
        max_price: float = None,
    ):
        if product_id not in self.product_index or self.content_sim is None:
            return None

        idx = self.product_index[product_id]
        scores = list(enumerate(self.content_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        results = []
        for i, score in scores:
            if i == idx:
                continue
            row = self.products_df.iloc[i]

            if category and str(row["category"]).lower() != category.lower():
                continue
            if min_price is not None and row["price"] < min_price:
                continue
            if max_price is not None and row["price"] > max_price:
                continue

            results.append(
                {
                    "product_id": int(row["product_id"]),
                    "name": row["name"],
                    "category": row["category"],
                    "price": float(row["price"]),
                    "similarity_score": round(float(score), 4),
                }
            )
            if len(results) >= top_n:
                break

        return results

    # ------------------------------------------------------------------
    # Collaborative: "products recommended for this user"
    # ------------------------------------------------------------------
    def recommend_by_user(self, user_id: int, top_n: int = 5):
        if self.user_item_matrix.empty or user_id not in self.user_item_matrix.index:
            return None

        already_rated = set(
            self.user_item_matrix.columns[self.user_item_matrix.loc[user_id] > 0]
        )

        # No other users to compare against -> fall back to top-rated items overall
        if self.user_sim.empty:
            candidate_scores = self.user_item_matrix.mean(axis=0)
        else:
            sims = self.user_sim[user_id].drop(index=user_id)
            sims = sims[sims > 0]

            if sims.empty:
                candidate_scores = self.user_item_matrix.mean(axis=0)
            else:
                weighted = self.user_item_matrix.loc[sims.index].T.dot(sims)
                sim_sums = sims.sum()
                candidate_scores = weighted / sim_sums if sim_sums > 0 else weighted

        candidate_scores = candidate_scores.drop(
            labels=[p for p in already_rated if p in candidate_scores.index],
            errors="ignore",
        )
        candidate_scores = candidate_scores.sort_values(ascending=False)

        results = []
        for product_id, score in candidate_scores.items():
            if score <= 0:
                continue
            product = db.get_product(int(product_id))
            if not product:
                continue
            results.append(
                {
                    "product_id": product["product_id"],
                    "name": product["name"],
                    "category": product["category"],
                    "price": product["price"],
                    "predicted_score": round(float(score), 4),
                }
            )
            if len(results) >= top_n:
                break

        return results


# Ensure the database exists and is seeded before building the engine,
# since this module can be imported before FastAPI's startup event fires.
db.init_db()

# Singleton engine instance used by the API
engine = RecommendationEngine()
