"""
evaluator.py
Evaluation metrics: RMSE, MAE, Precision@K, Recall@K.
"""

import numpy as np
from collections import defaultdict


def rmse(predictions):
    """Compute RMSE from Surprise predictions list."""
    errors = [(true - est) ** 2 for (_, _, true, est, _) in predictions]
    return np.sqrt(np.mean(errors))


def mae(predictions):
    """Compute MAE from Surprise predictions list."""
    errors = [abs(true - est) for (_, _, true, est, _) in predictions]
    return np.mean(errors)


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Compute Precision@K and Recall@K for all users.

    Args:
        predictions: list of Surprise prediction objects
        k: number of top recommendations to consider
        threshold: minimum rating to consider as relevant

    Returns:
        avg_precision, avg_recall
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum(true_r >= threshold for (_, true_r) in user_ratings)
        n_rec_k = sum(est >= threshold for (est, _) in top_k)
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for (est, true_r) in top_k)

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = np.mean(list(precisions.values()))
    avg_recall = np.mean(list(recalls.values()))
    return avg_precision, avg_recall


def print_metrics(predictions, k=10):
    """Print all evaluation metrics in a formatted table."""
    r = rmse(predictions)
    m = mae(predictions)
    p, rec = precision_recall_at_k(predictions, k=k)
    print(f"\n{'Metric':<20} {'Score':>10}")
    print("-" * 32)
    print(f"{'RMSE':<20} {r:>10.4f}")
    print(f"{'MAE':<20} {m:>10.4f}")
    print(f"{'Precision@{k}':<20} {p:>10.4f}")
    print(f"{'Recall@{k}':<20} {rec:>10.4f}")
