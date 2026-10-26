"""Tests for src/metrics.py."""
import numpy as np
import pytest

from src.metrics import evaluate, ks_statistic


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ks_statistic(y, [0.5, 0.5, 0.5, 0.5]) == 0.0


def test_evaluate_gini_is_2auc_minus_1():
    m = evaluate([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.6])
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert set(m) == {"auc", "gini", "ks", "brier", "balanced_accuracy"}
