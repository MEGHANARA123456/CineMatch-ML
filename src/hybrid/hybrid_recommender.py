"""
hybrid_recommender.py
Combines collaborative filtering (SVD) and content-based (TF-IDF) 
into a weighted hybrid recommender.
"""

import pandas as pd
import numpy as np


class HybridRecommender:
    """
    Weighted hybrid recommender:
        final_score = alpha * cf_score + (1 - alpha) * content_score
    """

    def __init__(self, cf_model, content_model, alpha=0.7):
        """
        Args:
            cf_model: trained Surprise SVD model
            content_model: fitted ContentBasedRecommender instance
            alpha: weight for collaborative filtering (0–1)
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.alpha = alpha

    def recommend(self, user_id: int, movies_df: pd.DataFrame, n: int = 10):
        """
        Generate hybrid recommendations for a given user.

        Args:
            user_id: target user ID
            movies_df: full movies DataFrame
            n: number of top recommendations to return

        Returns:
            DataFrame with top-N recommendations and hybrid scores
        """
        all_item_ids = movies_df["item_id"].tolist()
        cf_scores = {}
        for item_id in all_item_ids:
            pred = self.cf_model.predict(user_id, item_id)
            cf_scores[item_id] = pred.est

        # Normalize CF scores to [0, 1]
        cf_values = np.array(list(cf_scores.values()))
        cf_min, cf_max = cf_values.min(), cf_values.max()
        norm_cf = {
            iid: (s - cf_min) / (cf_max - cf_min + 1e-8)
            for iid, s in cf_scores.items()
        }

        # Build content scores from cosine similarity matrix (avg per item)
        cosine_sim = self.content_model.cosine_sim
        content_scores = {}
        for idx, row in movies_df.iterrows():
            item_id = row["item_id"]
            sim_scores = cosine_sim[idx]
            content_scores[item_id] = float(np.mean(sim_scores))

        # Normalize content scores
        c_values = np.array(list(content_scores.values()))
        c_min, c_max = c_values.min(), c_values.max()
        norm_content = {
            iid: (s - c_min) / (c_max - c_min + 1e-8)
            for iid, s in content_scores.items()
        }

        # Compute hybrid score
        hybrid_scores = {
            iid: self.alpha * norm_cf.get(iid, 0) + (1 - self.alpha) * norm_content.get(iid, 0)
            for iid in all_item_ids
        }

        top_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        top_ids = [iid for iid, _ in top_items]
        scores = [s for _, s in top_items]

        result = movies_df[movies_df["item_id"].isin(top_ids)][["item_id", "title"]].copy()
        score_map = dict(zip(top_ids, scores))
        result["hybrid_score"] = result["item_id"].map(score_map)
        result = result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        return result
