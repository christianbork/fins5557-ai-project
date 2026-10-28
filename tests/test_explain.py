"""Tests for src/explain.py, using the fake models trained in conftest.py."""

import numpy as np
import pandas as pd
import pytest

from src.config import PROTECTED, TARGET
from src.data import features_target
from src.explain import REASON_PHRASES, build_explainer, explain, top_reasons, unwrap
from src.features import feature_columns
from src.predict import load_artifacts, predict
from src.threshold import calibrate
from tests.test_data import make_fake


@pytest.fixture(scope="module")
def xgb_model(trained):
    """The fake XGBoost pipeline, calibrated like the real one (whichever model won on fake data)."""
    result, _, _, val, _ = trained
    X_val, y_val = features_target(val)
    return calibrate(result["pipelines"]["xgboost"], X_val, y_val, "sigmoid")


@pytest.fixture(scope="module")
def applicants(trained):
    _, _, _, _, test = trained
    X_test, _ = features_target(test)
    return X_test


def test_every_model_input_has_a_phrase():
    numeric, categorical = feature_columns()
    assert set(numeric + categorical) <= set(REASON_PHRASES)


def test_protected_attributes_are_never_reasons():
    assert not set(PROTECTED) & set(REASON_PHRASES)


def test_top_reasons_only_lists_factors_that_raise_risk():
    contrib = pd.Series(
        {"EXT_SOURCE_3": 0.8, "credit_term": 0.3, "AMT_INCOME_TOTAL": -0.5, "FLAG_OWN_CAR": 0.0}
    )
    assert top_reasons(contrib) == ["External credit scores", "Length of the loan"]


def test_top_reasons_empty_when_nothing_raises_risk():
    assert top_reasons(pd.Series({"EXT_SOURCE_3": -0.4, "credit_term": -0.1})) == []


def test_top_reasons_adds_up_inputs_that_share_a_phrase():
    # Two small external-score effects together outweigh one larger loan-term effect
    contrib = pd.Series({"EXT_SOURCE_1": 0.2, "ext_source_mean": 0.2, "credit_term": 0.3})
    assert top_reasons(contrib, k=1) == ["External credit scores"]


def test_contributions_add_up_to_the_model_score(xgb_model, applicants):
    explainer = build_explainer(xgb_model)
    contrib = explain(applicants, explainer)
    pipe = unwrap(xgb_model)
    margin = pipe[-1].predict(pipe[:-1].transform(applicants), output_margin=True)
    np.testing.assert_allclose(explainer["tree"].expected_value + contrib.sum(axis=1), margin, atol=1e-4)
    assert set(contrib.columns) <= set(REASON_PHRASES)  # one column per model input, not per one-hot column


def test_no_explainer_for_logistic_regression(trained):
    result = trained[0]
    assert build_explainer(result["pipelines"]["logistic_regression"]) is None


def test_predict_returns_reasons_and_shap(saved_model_dir, xgb_model):
    artifacts = load_artifacts(saved_model_dir / "model.joblib", saved_model_dir / "metadata.json",
                               saved_model_dir / "reference_stats.json")
    artifacts.update(model=xgb_model, explainer=build_explainer(xgb_model))
    applicant = make_fake(1, seed=3).drop(columns=TARGET).iloc[0].to_dict()
    out = predict(applicant, artifacts)
    assert 0 < len(out["reasons"]) <= 3
    assert set(out["reasons"]) <= set(REASON_PHRASES.values())
    numeric, categorical = feature_columns()
    assert set(out["shap"]) == set(numeric + categorical)  # raw and engineered inputs
    top = sum(v for f, v in out["shap"].items() if REASON_PHRASES[f] == out["reasons"][0])
    assert top > 0
