"""Tests for src/train.py. They train on the small fake dataset with tiny grids, so they run in
seconds and need neither the real data nor a saved model."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import clean, features_target, split
from src.train import (
    build_metadata,
    choose_model,
    evaluate,
    ks_statistic,
    save_artifacts,
    train_all,
)
from tests.test_data import make_fake

TINY_GRIDS = {
    "logistic_regression": {"model__C": [1.0]},
    "random_forest": {"model__max_depth": [4]},
    "xgboost": {"model__max_depth": [2]},
}


@pytest.fixture(scope="module")
def trained():
    train, val, _ = split(clean(make_fake(600)))
    result = train_all(train, val, param_grids=TINY_GRIDS, cv_folds=2, verbose=False)
    return result, train, val


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ks_statistic(y, [0.5, 0.5, 0.5, 0.5]) == 0.0


def test_evaluate_gini_is_2auc_minus_1():
    m = evaluate([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.6])
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert set(m) == {"auc", "gini", "ks", "brier", "balanced_accuracy"}


def test_choose_model_rule():
    table = pd.DataFrame({"auc": [0.770, 0.775, 0.790]},
                         index=["logistic_regression", "random_forest", "xgboost"])
    assert choose_model(table, min_gain=0.01)[0] == "xgboost"  # +0.020 clears the margin
    assert choose_model(table, min_gain=0.05)[0] == "logistic_regression"  # +0.020 doesn't
    table.loc["logistic_regression", "auc"] = 0.80
    assert choose_model(table)[0] == "logistic_regression"


def test_train_all_compares_three_models(trained):
    result, _, _ = trained
    assert set(result["comparison"].index) == {"logistic_regression", "random_forest", "xgboost"}
    assert result["chosen"] in result["pipelines"]
    assert result["reason"]


def test_saved_model_loads_and_returns_probabilities(trained, tmp_path):
    result, train, val = trained
    pipe = result["pipelines"][result["chosen"]]
    save_artifacts(pipe, build_metadata(result, len(train), len(val)),
                   tmp_path / "model.joblib", tmp_path / "metadata.json")

    loaded = joblib.load(tmp_path / "model.joblib")
    X_val, _ = features_target(val)
    proba = loaded.predict_proba(X_val)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["chosen_model"] == result["chosen"]
    assert meta["rows"] == {"train": len(train), "validation": len(val)}
    assert "scikit-learn" in meta["versions"]


def test_model_ignores_protected_attributes(trained):
    """Changing SEX, MARRIAGE or AGE must not change anyone's score."""
    result, _, val = trained
    pipe = result["pipelines"][result["chosen"]]
    X = val.drop(columns="default")
    changed = X.assign(SEX=3 - X["SEX"], MARRIAGE=1, AGE=99)
    np.testing.assert_array_equal(pipe.predict_proba(X), pipe.predict_proba(changed))
