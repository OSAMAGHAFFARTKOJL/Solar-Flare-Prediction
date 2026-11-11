"""Forecast skill scores. TSS and HSS are the standard pair in flare forecasting."""

import numpy as np
from sklearn.metrics import roc_auc_score


def flare_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = (2 * (tp * tn - fp * fn) / denom) if denom else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    return {
        "TSS": tpr - fpr, "HSS": hss, "AUC": auc,
        "precision": precision, "recall": recall, "F1": f1,
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def best_threshold(y_true, y_prob, grid=None):
    """Threshold maximising TSS. Always fit this on validation, never on test."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    scores = [flare_metrics(y_true, y_prob, t)["TSS"] for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])
