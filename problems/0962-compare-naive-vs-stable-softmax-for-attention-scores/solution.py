import numpy as np

def naive_softmax(X: np.ndarray):
    exp_X = np.exp(X)
    denominator = np.sum(exp_X)

    if np.isinf(denominator):
        return np.full_like(exp_X, np.nan)

    return exp_X / denominator

def stable_softmax(X: np.ndarray):
    X_norm = X - np.max(X)
    exp_X = np.exp(X_norm)
    return exp_X / np.sum(exp_X)


def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""
    scores_np = np.array(scores)
    softmax_scores = {}
    naive_softmax_scores = np.round(naive_softmax(scores), 6)
    stable_softmax_scores = np.round(stable_softmax(scores), 6)
    max_abs_diff = np.abs(np.max(naive_softmax_scores - stable_softmax_scores))

    softmax_scores["naive"] = naive_softmax_scores.tolist()
    softmax_scores["stable"] = stable_softmax_scores.tolist()
    softmax_scores["max_abs_diff"] = float(max_abs_diff)

    return softmax_scores
