"""Tests for src/train.py, using the small fake model from conftest.py."""
import json

import joblib
import numpy as np
import pandas as pd

from src.data import features_target
from src.train import choose_model


def test_choose_model_rule():
    table = pd.DataFrame({"auc": [0.770, 0.775, 0.790]},
                         index=["logistic_regression", "random_forest", "xgboost"])
    assert choose_model(table, min_gain=0.01)[0] == "xgboost"  # +0.020 clears the margin
    assert choose_model(table, min_gain=0.05)[0] == "logistic_regression"  # +0.020 doesn't
    table.loc["logistic_regression", "auc"] = 0.80
    assert choose_model(table)[0] == "logistic_regression"


def test_train_all_compares_three_models(trained):
    result = trained[0]
    assert set(result["comparison"].index) == {"logistic_regression", "random_forest", "xgboost"}
    assert result["chosen"] in result["pipelines"]
    assert result["reason"]


def test_saved_model_loads_and_returns_probabilities(trained, saved_model_dir):
    result, _, train, val, _ = trained
    loaded = joblib.load(saved_model_dir / "model.joblib")
    X_val, _ = features_target(val)
    proba = loaded.predict_proba(X_val)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    meta = json.loads((saved_model_dir / "metadata.json").read_text())
    assert meta["chosen_model"] == result["chosen"]
    assert meta["rows"]["train"] == len(train)
    assert 0 < meta["threshold"] < 1
    assert meta["input_columns"] == list(X_val.columns)
    assert "scikit-learn" in meta["versions"]


def test_model_ignores_protected_attributes(trained):
    """Changing SEX, MARRIAGE or AGE must not change anyone's PD."""
    model, val = trained[1]["model"], trained[3]
    X = val.drop(columns="default")
    changed = X.assign(SEX=3 - X["SEX"], MARRIAGE=1, AGE=99)
    np.testing.assert_array_equal(model.predict_proba(X), model.predict_proba(changed))
