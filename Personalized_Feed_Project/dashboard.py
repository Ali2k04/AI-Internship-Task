"""
dashboard.py
------------
Interactive Streamlit UI for the Personalized Content Feed Engine.

Run:
    streamlit run dashboard.py

Lets you:
  - Pick any user and see their personalized feed under any strategy
  - Fire off a brand-new interaction and watch the feed update in real time
  - See the user's raw interaction history
  - View the offline A/B test comparison across strategies
"""

import os

import pandas as pd
import streamlit as st

import database
from ab_testing import GROUP_CONFIG, assign_group, run_ab_evaluation
from engine import FeedEngine

DATA_DIR = "data"
CONTENT_CSV = os.path.join(DATA_DIR, "content_data.csv")
BEHAVIOR_CSV = os.path.join(DATA_DIR, "user_behavior.csv")

st.set_page_config(page_title="Personalized Feed Engine", layout="wide")


@st.cache_data
def load_data():
    if not (os.path.exists(CONTENT_CSV) and os.path.exists(BEHAVIOR_CSV)):
        import generate_data

        generate_data.main()
    content_df = pd.read_csv(CONTENT_CSV)
    behavior_df = pd.read_csv(BEHAVIOR_CSV)
    return content_df, behavior_df


def get_engine():
    """Engine lives in session_state so live interactions persist across reruns."""
    if "engine" not in st.session_state:
        content_df, behavior_df = load_data()
        st.session_state["engine"] = FeedEngine(behavior_df, content_df, half_life_days=14.0)
        database.init_db()
        database.load_interactions_csv(BEHAVIOR_CSV)
    return st.session_state["engine"]


def render_feed(feed):
    if not feed:
        st.info("No recommendations yet -- interact with some content first.")
        return
    for rank, item in enumerate(feed, start=1):
        st.markdown(
            f"**{rank}. {item['title']}**  \n"
            f"`{item['category']}` &nbsp;·&nbsp; content_id={item['content_id']} "
            f"&nbsp;·&nbsp; score={item['score']}"
        )


def main():
    st.title("🎯 Personalized Content Feed Engine")
    st.caption("User-behavior-based recommendations: clicks, likes, views, and search history.")

    content_df, behavior_df = load_data()
    engine = get_engine()

    tab_feed, tab_history, tab_ab, tab_data = st.tabs(
        ["📰 Personalized Feed", "🕘 Interaction History", "🧪 A/B Test Results", "📊 Dataset Overview"]
    )

    # ---------------------------------------------------------------- #
    # Tab 1: Personalized Feed + real-time interaction simulation
    # ---------------------------------------------------------------- #
    with tab_feed:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Choose a user")
            existing_users = sorted(engine.user_ids, key=lambda x: str(x))
            user_choice_mode = st.radio("User", ["Existing user", "New user"], horizontal=True)
            if user_choice_mode == "Existing user":
                user_id = st.selectbox("User ID", existing_users)
            else:
                user_id = st.text_input("New user ID", value="new_user")

            strategy = st.selectbox("Strategy", ["hybrid", "content", "collaborative"], index=0)
            alpha = 0.5
            if strategy == "hybrid":
                alpha = st.slider(
                    "Blend (1.0 = pure content-based, 0.0 = pure collaborative filtering)",
                    0.0, 1.0, 0.5, 0.05,
                )
            top_n = st.slider("Number of recommendations", 3, 10, 5)

            st.divider()
            st.subheader("Simulate a live interaction")
            content_choice = st.selectbox(
                "Content item",
                content_df["content_id"],
                format_func=lambda cid: f"{cid} - "
                f"{content_df.loc[content_df['content_id'] == cid, 'title'].values[0]}",
            )
            interaction_choice = st.selectbox("Interaction type", ["view", "click", "like", "search"])
            if st.button("▶ Record interaction now", use_container_width=True):
                engine.add_interaction(user_id, int(content_choice), interaction_choice)
                database.insert_interaction(
                    user_id, int(content_choice), interaction_choice, pd.Timestamp.now()
                )
                st.success(
                    f"Recorded: user '{user_id}' {interaction_choice}d content {content_choice}. "
                    f"Feed updated below."
                )

        with col_right:
            group = assign_group(user_id)
            st.subheader(f"Feed for user `{user_id}`")
            st.caption(f"A/B group (if using default hybrid rules): **{group}** "
                       f"→ default strategy `{GROUP_CONFIG[group]['strategy']}` "
                       f"(you can override with the controls on the left)")
            feed = engine.get_feed(user_id, top_n=top_n, strategy=strategy, alpha=alpha)
            render_feed(feed)

            with st.expander("Raw user profile (weighted, time-decayed scores)"):
                profile = engine.get_user_profile(user_id)
                if profile:
                    profile_df = pd.DataFrame(
                        [{"content_id": cid, "score": round(score, 3)} for cid, score in profile.items()]
                    ).merge(content_df[["content_id", "title", "category"]], on="content_id")
                    st.dataframe(profile_df.sort_values("score", ascending=False), hide_index=True)
                else:
                    st.write("No interactions yet.")

    # ---------------------------------------------------------------- #
    # Tab 2: Raw interaction history
    # ---------------------------------------------------------------- #
    with tab_history:
        st.subheader("Interaction log")
        history_df = database.fetch_interactions()
        st.dataframe(history_df.sort_values("timestamp", ascending=False), hide_index=True, height=500)

    # ---------------------------------------------------------------- #
    # Tab 3: A/B test comparison
    # ---------------------------------------------------------------- #
    with tab_ab:
        st.subheader("Offline A/B evaluation: content-only vs. collaborative-only vs. hybrid")
        st.caption(
            "Each user's most recent interactions are held out as a test set. The engine is "
            "trained on everything earlier, then we check how many held-out items each "
            "strategy actually recommended."
        )
        top_n_eval = st.slider("Top-K for evaluation", 3, 10, 5, key="ab_topk")
        summary_df, results_df, _ = run_ab_evaluation(behavior_df, content_df, top_n=top_n_eval)

        if summary_df.empty:
            st.warning("Not enough held-out data to evaluate.")
        else:
            st.dataframe(summary_df, hide_index=True)
            st.bar_chart(summary_df.set_index("group")[["avg_precision_at_k", "avg_recall_at_k"]])

            best = summary_df.iloc[0]
            st.success(
                f"Best strategy on held-out data: **{best['strategy']}** "
                f"(group {best['group']}), avg precision@{top_n_eval} = "
                f"{best['avg_precision_at_k']:.3f}"
            )

            with st.expander("Per-user detail"):
                st.dataframe(results_df, hide_index=True)

    # ---------------------------------------------------------------- #
    # Tab 4: Dataset overview
    # ---------------------------------------------------------------- #
    with tab_data:
        st.subheader("Content catalog")
        st.dataframe(content_df, hide_index=True, height=250)

        st.subheader("Interaction volume by type")
        st.bar_chart(behavior_df["interaction_type"].value_counts())

        st.subheader("Interactions by category")
        merged = behavior_df.merge(content_df[["content_id", "category"]], on="content_id")
        st.bar_chart(merged["category"].value_counts())


if __name__ == "__main__":
    main()
