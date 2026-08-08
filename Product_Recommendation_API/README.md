# Product Recommendation API

A FastAPI-based product recommendation service combining:

- **Content-based filtering** — TF-IDF + cosine similarity over product features, for "products similar to this one."
- **User-based collaborative filtering** — cosine similarity over a user-item rating matrix, for "products this user would like."
- **SQLite database** — products and interactions are persisted, seeded from CSV on first run.
- **Filtering** — restrict recommendations/catalog by category and price range.
- **Streamlit frontend** — a small UI that consumes the API.

## Folder Structure

```
Product_Recommendation_API/
│── data/
│   ├── products.csv          # sample product catalog
│   ├── interactions.csv      # sample user ratings
│   └── recommendation.db     # created automatically on first run
│── main.py                   # FastAPI app and endpoints
│── model.py                  # recommendation engine (content + collaborative)
│── database.py                # SQLite setup and query helpers
│── streamlit_app.py          # optional frontend
│── requirements.txt
│── README.md
```

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the API

```bash
uvicorn main:app --reload
```

The first run automatically creates `data/recommendation.db` and seeds it from the CSV files.

Open the interactive docs at:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## 3. Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/products` | List products. Optional query params: `category`, `min_price`, `max_price` |
| GET | `/products/{product_id}` | Get one product |
| GET | `/recommend/product/{product_id}` | Content-based: similar products. Params: `top_n`, `category`, `min_price`, `max_price` |
| GET | `/recommend/user/{user_id}` | Collaborative: personalized recommendations. Params: `top_n` |
| POST | `/interactions` | Log a new rating `{user_id, product_id, rating}` — updates the model live |

### Example requests

```bash
curl http://127.0.0.1:8000/recommend/product/101?top_n=5

curl http://127.0.0.1:8000/recommend/user/1?top_n=5

curl -X POST http://127.0.0.1:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "product_id": 106, "rating": 5}'
```

### Example response

```json
{
  "product_id": 101,
  "strategy": "content-based",
  "recommended_products": [
    {"product_id": 102, "name": "Noise Cancelling Earbuds", "category": "Electronics", "price": 79.99, "similarity_score": 0.42}
  ]
}
```

## 4. Run the Streamlit frontend (optional)

With the API running in one terminal, in another terminal:

```bash
streamlit run streamlit_app.py
```

## 5. Swap in your own data

Replace `data/products.csv` and `data/interactions.csv` with your own (same column names), delete `data/recommendation.db` if it already exists, and restart the API — it will reseed automatically.

## 6. Deploying

Any ASGI-friendly host works (Render, Railway, Fly.io, AWS). General steps:

1. Push this folder to a GitHub repo.
2. On Render/Railway, create a new **Web Service** from the repo.
3. Set the start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set the build command: `pip install -r requirements.txt`
5. Deploy — the platform will provide a public URL, and `/docs` will work there too.

For Streamlit, deploy `streamlit_app.py` separately (e.g. Streamlit Community Cloud) and update `API_URL` in that file to point at your deployed API's URL.

## Tech Stack

Python · FastAPI · Pandas · NumPy · Scikit-learn · SQLite · Uvicorn · Streamlit
