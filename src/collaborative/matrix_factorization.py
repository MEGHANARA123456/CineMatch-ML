"""
matrix_factorization.py
SVD and NMF models using the Surprise library.
"""

import joblib
from surprise import SVD, NMF
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy


def train_svd(data, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
    """
    Train SVD model.

    Args:
        data: Surprise Dataset object
        n_factors: number of latent factors
        n_epochs: training epochs
        lr_all: learning rate
        reg_all: regularization term

    Returns:
        trained SVD model, predictions
    """
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return algo, predictions


def train_nmf(data, n_factors=15, n_epochs=50):
    """Train NMF model."""
    algo = NMF(n_factors=n_factors, n_epochs=n_epochs)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)
    return algo, predictions


def cross_validate_model(algo, data, measures=["RMSE", "MAE"], cv=5):
    """Run k-fold cross-validation and return results."""
    results = cross_validate(algo, data, measures=measures, cv=cv, verbose=True)
    return results


def save_model(model, path: str):
    """Save a trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    """Load a saved model from disk."""
    return joblib.load(path)


def get_top_n_recommendations(predictions, n=10):
    """
    Get top-N recommendations for each user.

    Returns:
        dict: {user_id: [(item_id, estimated_rating), ...]}
    """
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n
