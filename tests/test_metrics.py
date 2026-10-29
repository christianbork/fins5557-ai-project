"""Tests for src/metrics.py."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from src.metrics import balanced_accuracy_threshold, evaluate, ks_statistic


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ks_statistic(y, [0.5, 0.5, 0.5, 0.5]) == 0.0


def test_evaluate_gini_is_2auc_minus_1():
    m = evaluate([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.6])
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert set(m) == {
        "auc",
        "gini",
        "ks",
        "pr_auc",
        "brier",
        "balanced_accuracy",
        "tpr",
        "tnr",
        "precision",
        "f1",
    }


def test_threshold_metrics_by_hand():
    # 2 defaulters, 4 good customers. At 0.5: declined = scores 0.9, 0.6, 0.55 -> 1 defaulter caught
    # (TPR 1/2), 2 good customers wrongly declined (TNR 2/4), precision 1/3.
    y = [1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.3, 0.6, 0.55, 0.2, 0.1]
    m = evaluate(y, scores, threshold=0.5)
    assert m["tpr"] == pytest.approx(0.5)
    assert m["tnr"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(2 * (1 / 3) * 0.5 / (1 / 3 + 0.5))
    assert m["balanced_accuracy"] == pytest.approx(0.5)


def test_nothing_declined_gives_zero_tpr_not_an_error():
    m = evaluate([0, 1, 0, 1], [0.1, 0.2, 0.1, 0.3], threshold=0.9)
    assert m["tpr"] == 0 and m["precision"] == 0 and m["tnr"] == 1


def test_balanced_accuracy_threshold_is_the_best_cut_off():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.08
    scores = np.clip(rng.normal(0.1 + 0.15 * y, 0.08), 0, 1)
    best = balanced_accuracy_threshold(y, scores)
    best_ba = balanced_accuracy_score(y, scores >= best)
    for t in np.linspace(0.01, 0.6, 60):
        assert balanced_accuracy_score(y, scores >= t) <= best_ba + 1e-12
    assert best_ba - 0.5 == pytest.approx(ks_statistic(y, scores) / 2)  # balanced accuracy = (1 + KS) / 2
