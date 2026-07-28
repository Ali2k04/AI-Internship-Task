# Personalized Content Feed Engine

A user-behavior-based recommendation engine that learns from **clicks, likes, views,
and search history**, and adapts the feed in real time as new behavior comes in.

Built with Python, Pandas, NumPy, scikit-learn, SQLite, and Streamlit.

---

## Folder Structure

```
Personalized_Feed_Project/
│── data/
│   ├── content_data.csv       # sample content catalog (generated)
│   ├── user_behavior.csv      # sample interaction log (generated)
│   └── feed_engine.db         # SQLite database (created on first run)
│── generate_data.py           # creates the sample CSVs above
│── database.py                # SQLite persistence layer
│── engine.py                  # core recommendation engine (the heart of the project)
│── ab_testing.py              # A/B test framework + offline evaluation
│── main.py                    # end-to-end CLI demo
│── dashboard.py                # interactive Streamlit UI
│── requirements.txt
│── README.md
```

---

## Quick Start

```bash
cd Personalized_Feed_Project
pip install -r requirements.txt

# Run the full CLI demo (generates sample data automatically on first run)
python main.py

# Or launch the interactive dashboard
streamlit run dashboard.py
```

No dataset needs to be supplied manually — `generate_data.py` creates a realistic
sample `content_data.csv` and `user_behavior.csv` the first time either `main.py`
or `dashboard.py` is run. You can delete `data/*.csv` at any time to regenerate a
fresh sample, or replace them with your own data as long as the columns match
(see **Data Schema** below).

---

## How It Works

### 1. Interaction weighting

Not all behavior signals the same strength of interest. Each raw event is
assigned a base weight in `engine.py`:

| Interaction | Weight |
|---|---|
| view | 1.0 |
| click | 2.0 |
| search | 1.5 |
| like | 3.0 |

*(Search history is modeled as an interaction against the content the search
surfaced/engaged — a common simplification when a dedicated search-log/query
index isn't available. If you have raw search queries, you can extend
`_prepare_content_similarity` to fold query terms into the content feature
space directly.)*

### 2. Time decay (recency matters)

A raw weight alone doesn't distinguish "the user loved this a month ago" from
"the user loved this an hour ago." Every interaction's weight is multiplied by
an exponential half-life decay:

```
decayed_score = base_weight × 0.5 ^ (days_since_interaction / half_life_days)
```

With the default 14-day half-life, an interaction from two weeks ago counts
half as much as one from today. This is computed in `FeedEngine._time_decay`.

### 3. Content-based filtering

Content items are featurized from their `category` and `tags` (one-hot /
multi-hot encoded), and an item-item cosine similarity matrix is built with
`sklearn.metrics.pairwise.cosine_similarity`. A user's content-based score for
every item is the sum of their weighted interactions with similar items —
i.e., "recommend more of what's similar to what you already engaged with."

### 4. Collaborative filtering

Independently, a **user-item interaction matrix** (rows = users, columns =
content, values = decayed weighted scores) is factorized with
**Non-negative Matrix Factorization (NMF)** from scikit-learn into latent
user and item factor matrices. This captures patterns content metadata can't
— e.g., "users who liked these Technology posts also tend to like these
specific Finance posts," even if the category tags don't overlap.

### 5. Hybrid scoring

`get_feed(user_id, strategy="hybrid", alpha=0.5)` blends both signals:

```
final_score = alpha × content_based_score + (1 - alpha) × collaborative_score
```

`alpha=1.0` is pure content-based, `alpha=0.0` is pure collaborative
filtering, and everything in between is a tunable hybrid. Both component
scores are min-max normalized first so neither dominates purely due to scale.

### 6. Real-time adaptation

`FeedEngine.add_interaction(user_id, content_id, interaction_type)` immediately
folds a new event into the live user-item matrix — the very next call to
`get_feed()` reflects it, with no restart or full offline rebuild required.
The expensive part (refitting the NMF collaborative model) is batched every
`refit_every` new interactions (default 5) — a standard production pattern:
cheap incremental signal immediately, periodic heavier refresh in the
background/on a schedule (`engine.force_refit()` can also be called manually,
e.g. from a cron job).

### 7. A/B testing (`ab_testing.py`)

Rather than faking click data, strategies are compared the way real
recommender teams evaluate offline before a live test:

1. Each user's most recent interactions are held out as a **test set**; the
   engine is trained only on what came before.
2. Users are deterministically hash-bucketed into groups
   (`A_content_only`, `B_collaborative_only`, `C_hybrid`), each mapped to a
   different strategy.
3. Each group's feed is generated from the train-only engine, and we check
   how many recommended items the user *actually went on to interact with*
   in the held-out set.
4. **Precision@K** and **Recall@K** are aggregated per group.

Whichever group wins on held-out historical data is the one you'd promote to
more live traffic in a real A/B test. Run it standalone:

```python
from ab_testing import run_ab_evaluation
import pandas as pd

behavior_df = pd.read_csv("data/user_behavior.csv")
content_df = pd.read_csv("data/content_data.csv")
summary_df, results_df, engine = run_ab_evaluation(behavior_df, content_df, top_n=5)
print(summary_df)
```

### 8. Persistence (SQLite)

`database.py` stores everything so nothing is lost between runs:
- `interactions` — the raw event log
- `user_profiles` — aggregated per-user/content weighted scores
- `ab_assignments` — which group each user is bucketed into
- `feed_impressions` — every feed ever generated, for later analysis

---

## Data Schema

**`content_data.csv`**

| column | description |
|---|---|
| content_id | unique integer id |
| title | display title |
| category | e.g. Technology, Sports, Health |
| tags | pipe-separated, e.g. `ai\|startups` |
| content_type | article / video / short-post |

**`user_behavior.csv`**

| column | description |
|---|---|
| user_id | integer or string user identifier |
| content_id | matches `content_data.csv` |
| interaction_type | view / click / like / search |
| timestamp | `YYYY-MM-DD HH:MM:SS` |

---

## Example Output

```
User 3  (A/B group: B_collaborative_only -> strategy=collaborative)
  - [ 11] Finance Article #11              (Finance      ) score=1.0
  - [ 53] Finance Article #53              (Finance      ) score=0.676
  - [ 32] Sports Short-Post #32            (Sports       ) score=0.6441
  - [  8] Sports Short-Post #8             (Sports       ) score=0.5754
  - [ 14] Sports Article #14               (Sports       ) score=0.5658

REAL-TIME UPDATE DEMO -- brand-new user 'new_user_demo' (cold start)
Feed BEFORE any interactions: all scores 0.0 (nothing to personalize on yet)
>>> Simulating: viewed/clicked/liked 3 Technology items right now...
Feed AFTER live interactions:
  - [ 37] Technology Video #37             (Technology   ) score=0.9414
  - [ 25] Technology Article #25           (Technology   ) score=0.9414
  - [ 31] Technology Article #31           (Technology   ) score=0.8153
  ...

A/B TEST RESULTS
               group      strategy  n_users  avg_precision_at_k  avg_recall_at_k
B_collaborative_only collaborative        8              0.150            0.217
      A_content_only       content        9              0.111            0.126
            C_hybrid        hybrid       13              0.108            0.147
```

---

## Requirements Checklist

**Core requirements**
- [x] Process and analyze user behavior data (clicks, likes, views, search)
- [x] Weighted interaction scoring mechanism
- [x] Personalized recommendation generation per user
- [x] Efficient, vectorized (NumPy/Pandas) scoring logic

**Bonus features**
- [x] Real-time feed updates (`add_interaction`, no rebuild needed)
- [x] Collaborative filtering layer (NMF matrix factorization)
- [x] Time-decay factor (exponential half-life on recency)
- [x] Streamlit dashboard UI (`dashboard.py`)
- [x] User profiles stored in a database (SQLite, `database.py`)
- [x] A/B testing across recommendation strategies with proper offline
      Precision@K / Recall@K evaluation (`ab_testing.py`)

---

## Scaling Notes (for taking this to production)

- **Cold start**: new users/content with no history get zero-signal scores.
  A production system would fall back to trending/popularity-based
  recommendations until enough behavior accumulates (a `most_popular()`
  fallback would be a natural next addition to `engine.py`).
- **NMF refit cost**: refitting scales with matrix size. At real scale, this
  would move to a scheduled batch job (e.g. nightly) with incremental
  approximate updates in between, or an online factorization method (e.g.
  implicit ALS with incremental updates).
- **Storage**: SQLite is fine for a prototype/demo; a production deployment
  would use a proper OLTP store for the event log and a feature
  store / vector index (e.g. FAISS, pgvector) for similarity lookups at scale.
- **Search history**: this project models search as an interaction with the
  content it surfaced. A richer implementation would maintain a
  query-embedding index and blend query-similarity into the content-based
  score directly.
