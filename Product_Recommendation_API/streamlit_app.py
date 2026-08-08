"""
streamlit_app.py
Simple frontend for the Product Recommendation API.

Run the FastAPI backend first (uvicorn main:app --reload), then:
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Product Recommender", page_icon="🛒", layout="centered")
st.title("🛒 Product Recommendation Demo")

mode = st.sidebar.radio("Recommendation type", ["Similar products", "For a user", "Browse catalog"])

if mode == "Similar products":
    st.subheader("Find products similar to a given product")
    product_id = st.number_input("Product ID", min_value=1, step=1, value=101)
    top_n = st.slider("Number of recommendations", 1, 20, 5)
    category = st.text_input("Filter by category (optional)")
    col1, col2 = st.columns(2)
    min_price = col1.number_input("Min price (optional)", min_value=0.0, value=0.0)
    max_price = col2.number_input("Max price (optional)", min_value=0.0, value=0.0)

    if st.button("Get recommendations"):
        params = {"top_n": top_n}
        if category:
            params["category"] = category
        if min_price > 0:
            params["min_price"] = min_price
        if max_price > 0:
            params["max_price"] = max_price

        resp = requests.get(f"{API_URL}/recommend/product/{int(product_id)}", params=params)
        if resp.status_code == 200:
            data = resp.json()
            st.write(f"Recommendations similar to product **{data['product_id']}**:")
            st.table(data["recommended_products"])
        else:
            st.error(resp.json().get("detail", "Something went wrong"))

elif mode == "For a user":
    st.subheader("Personalized recommendations for a user")
    user_id = st.number_input("User ID", min_value=1, step=1, value=1)
    top_n = st.slider("Number of recommendations", 1, 20, 5)

    if st.button("Get recommendations"):
        resp = requests.get(f"{API_URL}/recommend/user/{int(user_id)}", params={"top_n": top_n})
        if resp.status_code == 200:
            data = resp.json()
            st.write(f"Recommendations for user **{data['user_id']}**:")
            st.table(data["recommended_products"])
        else:
            st.error(resp.json().get("detail", "Something went wrong"))

else:
    st.subheader("Browse the product catalog")
    category = st.text_input("Filter by category (optional)")
    params = {}
    if category:
        params["category"] = category

    resp = requests.get(f"{API_URL}/products", params=params)
    if resp.status_code == 200:
        st.table(resp.json()["products"])
    else:
        st.error("Could not reach the API. Is it running?")
