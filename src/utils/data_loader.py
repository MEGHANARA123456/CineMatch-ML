"""
data_loader.py
Loads and splits the MovieLens 100K dataset.
"""

import pandas as pd
from surprise import Dataset, Reader
from sklearn.model_selection import train_test_split


def load_movielens(ratings_path: str, movies_path: str):
    """
    Load MovieLens ratings and movies metadata.

    Args:
        ratings_path: Path to u.data (tab-separated: user_id, item_id, rating, timestamp)
        movies_path:  Path to u.item (pipe-separated movie metadata)

    Returns:
        ratings_df, movies_df as DataFrames
    """
    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    movies_df = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        usecols=[0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        names=[
            "item_id", "title", "release_date", "unknown", "Action", "Adventure",
            "Animation", "Childrens", "Comedy", "Crime", "Documentary", "Drama",
            "Fantasy", "Film_Noir", "Horror", "Musical", "Mystery", "Romance",
            "Sci_Fi", "Thriller", "War", "Western",
        ],
    )

    return ratings_df, movies_df


def train_test_split_ratings(ratings_df, test_size=0.2, random_state=42):
    """Split ratings into train and test sets."""
    return train_test_split(ratings_df, test_size=test_size, random_state=random_state)


def to_surprise_dataset(ratings_df):
    """Convert a ratings DataFrame to Surprise Dataset format."""
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(ratings_df[["user_id", "item_id", "rating"]], reader)
