"""Evaluation metrics: weighted/class-wise F1, McNemar test, CI coverage."""
import numpy as np
from sklearn.metrics import f1_score


def scores_to_preds(scores: np.ndarray, eps: float = 0.5) -> np.ndarray:
    return (scores >= eps).astype(int)


def report_f1(labels: np.ndarray, scores: np.ndarray, eps: float = 0.5) -> dict:
    preds = scores_to_preds(scores, eps)
    return {
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "class0_f1": f1_score(labels, preds, pos_label=0),
        "class1_f1": f1_score(labels, preds, pos_label=1),
    }


def mcnemar_test(labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """Exact McNemar test (binomial) on paired predictions."""
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        from math import comb
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(comb(n, i) * 0.5 ** n for i in range(0, k + 1)))
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def ci_coverage(labels: np.ndarray, theta: np.ndarray, half_width: np.ndarray) -> float:
    """Fraction of instances where the true label falls within [theta-hw, theta+hw]."""
    lo, hi = theta - half_width, theta + half_width
    return float(np.mean((labels >= lo) & (labels <= hi)))


def out_of_range_rate(theta: np.ndarray) -> float:
    return float(np.mean((theta < 0) | (theta > 1)))
