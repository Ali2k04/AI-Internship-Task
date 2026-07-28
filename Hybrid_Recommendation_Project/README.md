# Hybrid Recommendation System (Content-Based + Collaborative Filtering)

A complete, runnable hybrid recommender that blends **content-based
filtering** (TF-IDF + cosine similarity over item features) with
**collaborative filtering** (SVD matrix factorization via the `Surprise`
library), plus a Streamlit web app, cold-start handling, and evaluation
metrics.

No external dataset download is required — a synthetic-but-realistic
dataset (60 items, 300 users, ~9,000 ratings) is generated automatically
the first time you run the project, with hidden per-user genre
preferences so both algorithms have real signal to learn from.

---

## 📁 Folder Structure

```
Hybrid_Recommendation_Project/
│
├── data/
│   ├── generate_dataset.py   # creates the synthetic items.csv / ratings.csv
│   ├── items.csv             # item_id, title, genres, tags, combined_features
│   └── ratings.csv           # user_id, item_id, rating, timestamp
│
├── saved_models/             # trained models, created by train.py
│   ├── content_model.joblib
│   ├── collab_model.surprise
│   └── collab_model.surprise.meta
│
├── reports/                  # optional EDA charts (from visualize.py)
│
├── data_utils.py              # loads/generates the dataset
├── model.py                   # ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
├── train.py                   # trains + saves both models
├── evaluate.py                # RMSE / Precision@K evaluation
├── visualize.py               # optional matplotlib/seaborn EDA charts
├── main.py                    # CLI demo (recommendations + evaluation)
├── app.py                     # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

```bash
cd Hybrid_Recommendation_Project
pip install -r requirements.txt
```

### Run the CLI demo

```bash
python main.py
```

The first run trains both models and saves them to `saved_models/`
(subsequent runs load the saved models instantly). It prints Top-5
recommendations for:
- an existing user,
- a cold-start user who only states favourite genres,
- a cold-start user with **no** information at all (popularity fallback).

Useful flags:

```bash
python main.py --user 42 --top_n 10 --alpha 0.5   # specific user, more items, 50/50 weighting
python main.py --evaluate                          # also print RMSE / Precision@K
```

### Run the Streamlit web app

```bash
streamlit run app.py
```

- Pick any existing user, or switch to "New user (cold start)" and choose
  favourite genres.
- Live slider to adjust the collaborative ↔ content weight.
- "Because you liked X" item-similarity explorer.
- Optional evaluation panel (RMSE / Precision@K) computed on demand.

### (Optional) Generate EDA charts

```bash
python visualize.py     # saves 3 PNGs to reports/
```

### Retrain from scratch

```bash
rm -rf saved_models/
python train.py
```

---

## 🧠 How It Works

### 1. Content-Based Filtering (`ContentBasedRecommender`)
Each item's genres + descriptive tags are combined into one text field and
vectorized with **TF-IDF**. Cosine similarity between items gives
item-to-item recommendations ("because you liked X"). A **user profile
vector** is also built as the rating-weighted average of the TF-IDF
vectors of items a user rated highly — this is what powers the
"user-profile-based recommendations" bonus feature, and also what makes
cold-start-by-genre possible.

### 2. Collaborative Filtering (`CollaborativeRecommender`)
Wraps `Surprise`'s **SVD** (matrix factorization) algorithm, trained on
the full user–item rating matrix, to predict a rating a user would give
an item they haven't rated yet.

### 3. Hybrid Combination (`HybridRecommender`)
```
final_score = alpha * collaborative_estimate + (1 - alpha) * content_estimate
```
Default `alpha = 0.7` → **70% collaborative / 30% content**, adjustable
per call (or live via the Streamlit slider).

> **Note on the starter pseudocode:** the original spec combined a raw
> cosine similarity (0–1 range) directly with a predicted rating (1–5
> range). That silently biases every hybrid score toward whichever term
> uses the bigger scale. This implementation normalizes the content
> similarity onto the same 1–5 scale (`content_score_to_rating()`)
> *before* blending, so the `alpha` weight behaves the way you'd expect.

### 4. Cold-Start Handling
- **Brand-new user, genres known:** pure content-based score against
  their stated genres.
- **Brand-new user, nothing known:** falls back to item popularity
  (average rating).
- **Item with fewer than 3 ratings:** scored purely on content
  similarity, since a collaborative estimate would be unreliable.

---

## 📊 Evaluation

Run `python main.py --evaluate` or `python evaluate.py` directly. Example
output from the included synthetic dataset (80/20 train/test split,
`k=5`, "relevant" = rating ≥ 3.5):

```
RMSE (collaborative SVD):                1.00
Precision@5 (collaborative only):        0.23
Precision@5 (hybrid, alpha=0.7):         0.24
```

Exact numbers vary slightly by machine/library version, but the pattern
— hybrid ranking matching or slightly beating collaborative-only — is
the expected result: content features add the most value precisely
where collaborative data is thin (new users/items), which is also where
Precision@K on a fixed test split has less room to move.

---

## 🔧 Tech Stack

- Python, Pandas, NumPy
- scikit-learn (TF-IDF, cosine similarity)
- Surprise (SVD collaborative filtering)
- Streamlit (web app)
- matplotlib / seaborn (optional EDA)
- joblib (model persistence)

## ✅ Bonus Features Implemented

- [x] Adjustable collaborative/content weighting (default 70/30, live slider in app)
- [x] User-profile-based recommendations
- [x] Cold-start handling (new users and new/low-rated items)
- [x] Streamlit web app
- [x] Trained models saved to disk for reuse (`saved_models/`)
- [x] Evaluation via RMSE and Precision@K (collaborative-only vs. hybrid)

---

## 📦 Dataset

Included (`data/items.csv`, `data/ratings.csv`), auto-generated by
`data/generate_dataset.py` if missing. All titles are invented so the
project runs fully offline with no license/attribution concerns. To use
your own data, replace the two CSVs, keeping the columns:
- `items.csv`: `item_id, combined_features` (any extra descriptive columns are fine)
- `ratings.csv`: `user_id, item_id, rating`
