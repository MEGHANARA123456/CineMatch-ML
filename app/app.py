"""
app.py
Streamlit dashboard for the Personalized Recommendation System.

Run with:
    streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import load_movielens
from src.hybrid.hybrid_recommender import HybridRecommender

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Personalized Movie Recommendation System")
st.markdown("Hybrid model combining **Collaborative Filtering (SVD)** and **Content-Based (TF-IDF)** filtering.")
st.divider()


# ── Load data & models ─────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    ratings_df, movies_df = load_movielens(
        ratings_path="../data/raw/u.data",
        movies_path="../data/raw/u.item",
    )
    cf_model = joblib.load("../models/svd_model.pkl")
    content_model = joblib.load("../models/content_model.pkl")
    hybrid = HybridRecommender(cf_model=cf_model, content_model=content_model, alpha=0.7)
    return ratings_df, movies_df, hybrid


try:
    ratings_df, movies_df, hybrid = load_resources()
    models_loaded = True
except Exception as e:
    st.warning(f"⚠️ Models not found. Train models first by running the notebooks. ({e})")
    models_loaded = False


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
n_recommendations = st.sidebar.slider("Number of recommendations", 5, 20, 10)
alpha = st.sidebar.slider("CF weight (alpha)", 0.0, 1.0, 0.7, 0.1,
    help="Higher alpha = more collaborative, lower = more content-based")

st.sidebar.divider()
st.sidebar.markdown("**Dataset:** MovieLens 100K")
st.sidebar.markdown("**Models:** SVD + TF-IDF Hybrid")


# ── Main interface ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Get Recommendations")
    user_id = st.number_input("Enter User ID (1–943)", min_value=1, max_value=943, value=1)

    if st.button("🔍 Recommend", type="primary", disabled=not models_loaded):
        with st.spinner("Generating recommendations..."):
            hybrid.alpha = alpha
            recommendations = hybrid.recommend(user_id, movies_df, n=n_recommendations)
        st.session_state["recommendations"] = recommendations
        st.session_state["user_id"] = user_id

with col2:
    st.subheader("🎥 Top Recommendations")
    if "recommendations" in st.session_state:
        recs = st.session_state["recommendations"]
        uid = st.session_state["user_id"]
        st.markdown(f"**Top {len(recs)} recommendations for User {uid}:**")
        for i, row in recs.iterrows():
            score_pct = int(row["hybrid_score"] * 100)
            st.markdown(f"**{i+1}.** {row['title']}  &nbsp; `{score_pct}% match`")
        st.divider()
        st.dataframe(recs[["title", "hybrid_score"]].rename(
            columns={"title": "Movie", "hybrid_score": "Score"}),
            use_container_width=True
        )
    else:
        st.info("Enter a user ID and click Recommend to see results.")


# ── Dataset stats ──────────────────────────────────────────────────────────────
if models_loaded:
    st.divider()
    st.subheader("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Users", ratings_df["user_id"].nunique())
    c2.metric("Total Movies", ratings_df["item_id"].nunique())
    c3.metric("Total Ratings", len(ratings_df))
    c4.metric("Avg Rating", f"{ratings_df['rating'].mean():.2f}")
