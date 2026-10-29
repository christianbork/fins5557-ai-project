"""Credit-scoring metrics shared by model comparison (train.py) and the final test report (threshold.py)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the largest gap between the score distributions of defaulters and
    non-defaulters. Same measure as Code Bank W05-05's ks_statistic(), computed from the ROC
    curve (KS = max(TPR - FPR)) instead of a Python loop.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def balanced_accuracy_threshold(y_true, y_score) -> float:
    """The cut-off that maximises balanced accuracy, (TPR + TNR) / 2 (the Big Data exam's method).

    Balanced accuracy = (TPR + 1 - FPR) / 2, so it peaks where TPR - FPR peaks: the same point as
    the KS statistic. Scores at or above the cut-off are predicted to default.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(min(thresholds[np.argmax(tpr - fpr)], 1.0))  # roc_curve's first threshold is inf


def evaluate(y_true, y_score, threshold: float = 0.5) -> dict:
    """Metrics used in credit scoring.

    Threshold-free: AUC / Gini / KS measure ranking (does the model put defaulters above
    non-defaulters?); PR-AUC (average precision) does too, but focuses on the rare defaulters, so it
    is more sensitive than AUC when only 8% default; Brier score measures probability accuracy
    (lower is better).

    At a cut-off (score >= threshold = predicted default = declined):
    TPR (recall) = share of defaulters caught; TNR (specificity) = share of good customers approved;
    precision = share of declined applicants who would really have defaulted; F1 combines precision
    and TPR; balanced accuracy = (TPR + TNR) / 2.
    """
    y_true = np.asarray(y_true)
    predicted = (np.asarray(y_score) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "tpr": recall_score(y_true, predicted, zero_division=0),
        "tnr": recall_score(y_true, predicted, pos_label=0, zero_division=0),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
    }
