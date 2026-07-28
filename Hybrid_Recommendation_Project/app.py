"""
app.py
------
Bonus feature: Streamlit web app.

Run with:
    streamlit run app.py

Lets you:
  - Pick an existing user and see their hybrid Top-N recommendations
  - Adjust the collaborative/content weighting live with a slider
  - Simulate a brand-new ("cold-start") user by picking favourite genres
  - View item similarity ("because you liked X")
  - View RMSE / Precision@K for the current model (cached, computed once)
"""

import streamlit as st
import pandas as pd

from train import load_trained
from model import HybridRecommender

st.set_page_config(page_title="Hybrid Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading / training models...")
def get_models():
    return load_trained()


content_model, collab_model, ratings_df, items_df = get_models()
hybrid = HybridRecommender(content_model, collab_model, ratings_df)

st.title("🎬 Hybrid Recommendation System")
st.caption("Content-Based Filtering + Collaborative Filtering (SVD), blended with an adjustable weight.")

with st.sidebar:
    st.header("Settings")

    mode = st.radio("User type", ["Existing user", "New user (cold start)"])

    alpha = st.slider(
        "Weight: Collaborative ↔ Content",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="1.0 = pure collaborative filtering, 0.0 = pure content-based. Default 0.7 = 70% collaborative / 30% content.",
    )
    top_n = st.slider("Number of recommendations", 3, 15, 5)

    if mode == "Existing user":
        user_ids = sorted(ratings_df["user_id"].unique().tolist())
        user_id = st.selectbox("User ID", user_ids, index=0)
        new_user_genres = None
    else:
        user_id = int(ratings_df["user_id"].max()) + 12345
        all_genres = sorted({g for gs in items_df["genres"].str.split("|") for g in gs})
        new_user_genres = st.multiselect("Favourite genres", all_genres, default=all_genres[:2])

    st.divider()
    show_eval = st.checkbox("Show evaluation metrics (RMSE / Precision@K)", value=False)


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Top {top_n} recommendations")
    recs = hybrid.recommend(
        user_id, top_n=top_n, alpha=alpha,
        new_user_genres=new_user_genres,
    )
    if recs:
        rec_df = pd.DataFrame(recs)[["title", "genres", "score", "reason"]]
        rec_df.columns = ["Title", "Genres", "Score (1-5)", "How it was scored"]
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        st.bar_chart(rec_df.set_index("Title")["Score (1-5)"])
    else:
        st.info("No recommendations available for this selection.")

with col2:
    st.subheader("User snapshot")
    if mode == "Existing user":
        user_hist = ratings_df[ratings_df["user_id"] == user_id].merge(items_df, on="item_id")
        st.metric("Ratings given", len(user_hist))
        if not user_hist.empty:
            st.metric("Average rating", round(user_hist["rating"].mean(), 2))
            top_liked = user_hist.sort_values("rating", ascending=False).head(5)
            st.write("Top rated items:")
            st.dataframe(
                top_liked[["title", "genres", "rating"]].rename(
                    columns={"title": "Title", "genres": "Genres", "rating": "Rating"}
                ),
                hide_index=True, use_container_width=True,
            )
    else:
        st.info("This is a simulated brand-new user with no rating history — "
                 "recommendations come purely from the selected genres "
                 "(content-based cold-start).")

st.divider()

with st.expander("🔍 Explore item-to-item similarity ('because you liked...')"):
    item_titles = items_df.set_index("item_id")["title"].to_dict()
    chosen_item = st.selectbox(
        "Pick an item", items_df["item_id"],
        format_func=lambda x: f"{x} - {item_titles[x]}",
    )
    similar = content_model.similar_items(chosen_item, top_n=5)
    sim_df = pd.DataFrame(similar, columns=["item_id", "Title", "Similarity"])
    st.dataframe(sim_df[["Title", "Similarity"]], hide_index=True, use_container_width=True)

if show_eval:
    st.divider()
    st.subheader("Evaluation metrics")
    with st.spinner("Running RMSE / Precision@K evaluation on a held-out split..."):
        import evaluate as ev
        rmse, trainset, testset, algo = ev.rmse_eval(ratings_df)
        preds = [algo.predict(uid, iid, r) for (uid, iid, r) in testset]
        prec_collab, rec_collab = ev.precision_recall_at_k(preds, k=5)

        train_pairs = {(trainset.to_raw_uid(u), trainset.to_raw_iid(i))
                        for (u, i, _r) in trainset.all_ratings()}
        train_ratings_df = ratings_df[
            ratings_df.apply(lambda r: (r["user_id"], r["item_id"]) in train_pairs, axis=1)
        ]
        prec_hybrid = ev.hybrid_precision_at_k(testset, algo, content_model, train_ratings_df, alpha=alpha, k=5)

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE (collaborative)", f"{rmse:.3f}")
    m2.metric("Precision@5 (collaborative only)", f"{prec_collab:.3f}")
    m3.metric(f"Precision@5 (hybrid, α={alpha})", f"{prec_hybrid:.3f}")

st.divider()
st.caption(
    "Cold-start handling: new users are scored purely on content similarity to their "
    "stated genres (or overall popularity if no preferences are given); items with "
    "fewer than 3 ratings are scored purely on content similarity, since a "
    "collaborative-filtering estimate is not yet reliable for them."
)
