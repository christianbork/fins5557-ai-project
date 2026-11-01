"""Tests for src/train.py, using the small fake model from conftest.py."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.data import features_target
from src.train import choose_model, make_search, search_spaces


def test_choose_model_rule():
    table = pd.DataFrame(
        {"auc": [0.770, 0.775, 0.790]}, index=["logistic_regression", "random_forest", "xgboost"]
    )
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
    """Changing gender, age or family status must not change anyone's PD."""
    model, val = trained[1]["model"], trained[3]
    X = val.drop(columns="default")
    changed = X.assign(
        CODE_GENDER=X["CODE_GENDER"].map({"F": "M", "M": "F"}), DAYS_BIRTH=-25_000, NAME_FAMILY_STATUS="Widow"
    )
    np.testing.assert_array_equal(model.predict_proba(X), model.predict_proba(changed))


def test_search_spaces_tune_the_imbalance_correction():
    y = pd.Series([0] * 90 + [1] * 10)
    spaces = search_spaces(y)
    assert spaces["xgboost"]["model__scale_pos_weight"] == [1.0, 3.0, 9.0]  # none, sqrt(9), neg/pos = 9
    assert None in spaces["random_forest"]["model__class_weight"]
    assert spaces["random_forest"]["model__max_features"] == ["sqrt", 0.2, 0.33, 0.5]  # the exam's mtry
    assert set(spaces["logistic_regression"]["model__class_weight"]) == {None, "balanced"}


def test_make_search_uses_a_grid_only_when_small():
    spaces = search_spaces(pd.Series([0] * 90 + [1] * 10))
    cv = 3
    assert isinstance(
        make_search("logistic_regression", None, spaces["logistic_regression"], cv), GridSearchCV
    )
    xgb = make_search("xgboost", None, spaces["xgboost"], cv, n_iter=30)
    assert isinstance(xgb, RandomizedSearchCV) and xgb.n_iter == 30
    assert isinstance(make_search("random_forest", None, {"model__max_depth": [4, 8]}, cv), GridSearchCV)


def test_every_candidate_is_recorded(trained):
    tuning = trained[0]["tuning"]
    assert set(tuning["model"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert {"rank", "cv_auc", "cv_pr_auc", "fit_seconds"} <= set(tuning.columns)
    assert tuning["cv_auc"].between(0, 1).all() and tuning["cv_pr_auc"].between(0, 1).all()


def test_comparison_has_imbalance_metrics(trained):
    comparison = trained[0]["comparison"]
    assert {"cv_pr_auc", "pr_auc", "tpr", "tnr", "precision", "f1", "ba_threshold"} <= set(comparison.columns)


def test_metadata_records_tuning_and_both_cut_offs(saved_model_dir):
    meta = json.loads((saved_model_dir / "metadata.json").read_text())
    assert set(meta["tuning"]["best_params_all_models"]) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert set(meta["threshold_comparison"]) == {"profit", "balanced_accuracy"}
    assert meta["threshold_comparison"]["profit"]["cutoff"] == meta["threshold"]
