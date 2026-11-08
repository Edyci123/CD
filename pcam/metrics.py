import numpy as np
from sklearn.metrics import roc_auc_score


def binary_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """AUC (threshold-free) plus threshold-based metrics. Positive class = tumour."""
    y_true = y_true.astype(int)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
