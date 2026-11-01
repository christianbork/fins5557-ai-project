"""Tests for src/predict.py, using the fake model saved by conftest.py."""

import pytest

from src.config import TARGET
from src.predict import load_artifacts, predict, predict_batch
from tests.test_data import make_fake


@pytest.fixture(scope="module")
def artifacts(saved_model_dir):
    return load_artifacts(
        saved_model_dir / "model.joblib",
        saved_model_dir / "metadata.json",
        saved_model_dir / "reference_stats.json",
    )


@pytest.fixture
def applicant():
    return make_fake(1).drop(columns=TARGET).iloc[0].to_dict()


def test_predict_valid_applicant(artifacts, applicant):
    out = predict(applicant, artifacts)
    assert set(out) == {"pd", "decision", "reasons", "shap", "warnings"}
    assert 0 <= out["pd"] <= 1
    assert out["decision"] in {"approve", "refer", "decline"}


def test_predict_missing_field_gives_clear_error(artifacts, applicant):
    del applicant["AMT_CREDIT"]
    with pytest.raises(ValueError, match="Missing applicant fields: AMT_CREDIT"):
        predict(applicant, artifacts)


def test_predict_does_not_need_protected_attributes(artifacts, applicant):
    for field in ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "SK_ID_CURR"):
        applicant.pop(field)
    assert 0 <= predict(applicant, artifacts)["pd"] <= 1


def test_predict_batch_of_100(artifacts):
    df = make_fake(100, seed=1).drop(columns=TARGET)
    out = predict_batch(df, artifacts)
    assert len(out) == 100
    assert out["pd"].between(0, 1).all()
    assert set(out["decision"]) <= {"approve", "refer", "decline"}
    assert list(out.columns[:-3]) == list(df.columns)  # input columns kept, unchanged


def test_predict_with_missing_optional_values(artifacts, applicant):
    """Thin-file applicants: no external scores and no car age still get a PD."""
    applicant.update(EXT_SOURCE_1=None, EXT_SOURCE_2=None, EXT_SOURCE_3=None, OWN_CAR_AGE=None)
    assert 0 <= predict(applicant, artifacts)["pd"] <= 1


def test_batch_and_single_agree(artifacts):
    df = make_fake(5, seed=2).drop(columns=TARGET)
    batch = predict_batch(df, artifacts)
    for i in range(5):
        assert predict(df.iloc[i].to_dict(), artifacts)["pd"] == pytest.approx(batch["pd"].iloc[i])


def test_missing_model_files_explain_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        load_artifacts(tmp_path / "model.joblib", tmp_path / "metadata.json")


def test_predict_batch_adds_reasons_only_where_needed(artifacts):
    df = make_fake(200, seed=4).drop(columns=TARGET)
    out = predict_batch(df, artifacts, with_reasons=True)
    assert (out.loc[out["decision"] == "approve", "reasons"] == "").all()
    if artifacts.get("explainer") is not None:  # the fake model may be logistic regression (no explainer)
        assert (out.loc[out["decision"] != "approve", "reasons"] != "").all()
