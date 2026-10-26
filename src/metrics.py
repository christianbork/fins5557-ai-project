"""Credit-scoring metrics shared by model comparison (train.py) and the final test report (threshold.py)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the largest gap between the score distributions of defaulters and
    non-defaulters. Same measure as Code Bank W05-05's ks_statistic(), computed from the ROC
    curve (KS = max(TPR - FPR)) instead of a Python loop.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score, threshold: float = 0.5) -> dict:
    """Metrics used in credit scoring.

    AUC / Gini / KS measure ranking (does the model put defaulters above non-defaulters?).
    Brier score measures probability accuracy (lower is better). Balanced accuracy needs a
    cut-off: 0.5 when comparing models, the cost-based threshold in the final test report.
    """
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, (np.asarray(y_score) >= threshold).astype(int)),
    }
