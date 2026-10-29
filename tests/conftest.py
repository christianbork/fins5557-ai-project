"""Shared test fixtures: one small model trained on fake data, reused by several test files.
Takes a few seconds and needs neither the real data nor a saved model."""
import pytest

from src.data import clean, features_target, split
from src.threshold import calibrate_and_set_threshold
from src.train import build_metadata, save_artifacts, train_all
from src.validation import build_reference_stats, save_reference_stats
from tests.test_data import make_fake

TINY_GRIDS = {
    "logistic_regression": {"model__C": [1.0]},
    "random_forest": {"model__max_depth": [4]},
    "xgboost": {"model__max_depth": [2]},
}


@pytest.fixture(scope="session")
def trained():
    """(train_all result, calibration/threshold result, train, val, test) on 1,000 fake customers."""
    train, val, test = split(clean(make_fake(1000)))
    result = train_all(train, val, param_grids=TINY_GRIDS, cv_folds=2, verbose=False)
    X_val, y_val = features_target(val)
    X_test, y_test = features_target(test)
    decision = calibrate_and_set_threshold(result["pipelines"][result["chosen"]], X_val, y_val, X_test, y_test)
    return result, decision, train, val, test


@pytest.fixture(scope="session")
def saved_model_dir(trained, tmp_path_factory):
    """The fake model saved like `python -m src.train` saves the real one."""
    result, decision, train, val, test = trained
    out = tmp_path_factory.mktemp("models")
    X_val, _ = features_target(val)
    meta = build_metadata(result, decision, list(X_val.columns),
                          {"train": len(train), "validation": len(val), "test": len(test)})
    save_artifacts(decision["model"], meta, out / "model.joblib", out / "metadata.json")
    save_reference_stats(build_reference_stats(features_target(train)[0]), out / "reference_stats.json")
    return out


@pytest.fixture(autouse=True)
def no_decision_log(monkeypatch, tmp_path):
    """Tests never write to logs/decisions.jsonl: the decision log is off unless a test turns it on,
    and then it goes to a temp file."""
    monkeypatch.setattr("src.audit.DECISION_LOG", False)
    monkeypatch.setattr("src.audit.DECISION_LOG_PATH", tmp_path / "decisions.jsonl")
