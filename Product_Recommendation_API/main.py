"""
main.py
FastAPI application exposing the recommendation engine.

Endpoints:
    GET  /                              -> health check
    GET  /products                      -> list products (with filters)
    GET  /products/{product_id}         -> get a single product
    GET  /recommend/product/{product_id}-> content-based "similar products"
    GET  /recommend/user/{user_id}      -> user-based collaborative recommendations
    POST /interactions                  -> log a new user-product rating
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database as db
from model import engine

# ----------------------------------------------------------------------
# App + Swagger/OpenAPI customization
# ----------------------------------------------------------------------
app = FastAPI(
    title="Product Recommendation API",
    description=(
        "A content-based and collaborative-filtering product recommendation "
        "service. Supply a product ID to get similar items, or a user ID to "
        "get personalized suggestions based on rating history."
    ),
    version="1.0.0",
    contact={"name": "Product Recommendation API"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "General", "description": "Health check and metadata."},
        {"name": "Products", "description": "Browse and filter the product catalog."},
        {"name": "Recommendations", "description": "Content-based and collaborative recommendations."},
        {"name": "Interactions", "description": "Log user activity used by the recommender."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    db.init_db()
    engine.refresh()


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class InteractionIn(BaseModel):
    user_id: int = Field(..., example=1)
    product_id: int = Field(..., example=101)
    rating: float = Field(..., ge=1, le=5, example=4.5)


# ----------------------------------------------------------------------
# General
# ----------------------------------------------------------------------
@app.get("/", tags=["General"])
def home():
    return {"message": "Product Recommendation API is running"}


# ----------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------
@app.get("/products", tags=["Products"])
def list_products(
    category: Optional[str] = Query(None, description="Filter by category, e.g. Electronics"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
):
    products = db.get_all_products(category=category, min_price=min_price, max_price=max_price)
    return {"count": len(products), "products": products}


@app.get("/products/{product_id}", tags=["Products"])
def get_product(product_id: int):
    product = db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    return product


# ----------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------
@app.get("/recommend/product/{product_id}", tags=["Recommendations"])
def recommend_by_product(
    product_id: int,
    top_n: int = Query(5, ge=1, le=20, description="Number of recommendations"),
    category: Optional[str] = Query(None, description="Restrict results to a category"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
):
    if not db.get_product(product_id):
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    results = engine.recommend_by_product(
        product_id, top_n=top_n, category=category, min_price=min_price, max_price=max_price
    )
    return {
        "product_id": product_id,
        "strategy": "content-based",
        "recommended_products": results,
    }


@app.get("/recommend/user/{user_id}", tags=["Recommendations"])
def recommend_by_user(
    user_id: int,
    top_n: int = Query(5, ge=1, le=20, description="Number of recommendations"),
):
    results = engine.recommend_by_user(user_id, top_n=top_n)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"No interaction history found for user {user_id}",
        )
    return {
        "user_id": user_id,
        "strategy": "user-based collaborative filtering",
        "recommended_products": results,
    }


# ----------------------------------------------------------------------
# Interactions (feeds the collaborative model)
# ----------------------------------------------------------------------
@app.post("/interactions", tags=["Interactions"])
def create_interaction(interaction: InteractionIn):
    if not db.get_product(interaction.product_id):
        raise HTTPException(
            status_code=404, detail=f"Product {interaction.product_id} not found"
        )

    new_id = db.add_interaction(interaction.user_id, interaction.product_id, interaction.rating)
    engine.refresh()  # keep the collaborative model up to date

    return {
        "message": "Interaction recorded",
        "interaction_id": new_id,
        "user_id": interaction.user_id,
        "product_id": interaction.product_id,
        "rating": interaction.rating,
    }
