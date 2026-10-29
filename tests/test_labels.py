"""Tests for src/labels.py."""

import pandas as pd
import pytest

from src.config import RAW_INPUTS
from src.labels import (
    COLUMNS,
    column_label,
    decision_label,
    flag_text,
    headers,
    input_label,
    model_choice_reason,
    model_label,
    status_label,
)


def test_decisions_are_capitalised():
    assert [decision_label(d) for d in ("approve", "refer", "decline")] == ["Approve", "Refer", "Decline"]


def test_model_and_status_names():
    assert model_label("xgboost") == "XGBoost"
    assert model_label("logistic_regression") == "Logistic regression"
    assert status_label("alert") == "Alert"


@pytest.mark.parametrize("field", RAW_INPUTS)
def test_every_model_input_has_a_readable_name(field):
    name = input_label(field)
    assert name != field and "_" not in name and name[0].isupper()


def test_day_fields_are_named_in_days():
    assert input_label("DAYS_EMPLOYED") == "Days in current job"
    assert input_label("EXT_SOURCE_2") == "External credit score 2"


def test_every_column_header_is_readable():
    for header in COLUMNS.values():
        assert "_" not in header and header[0].isupper()


def test_headers_renames_columns_and_index():
    df = pd.DataFrame({"pd": [0.1], "AMT_CREDIT": [1.0], "my_ref": ["a"]}).rename_axis("group")
    out = headers(df)
    assert list(out.columns) == ["PD", "Loan amount", "my_ref"]  # unknown columns (the user's own) are kept
    assert out.index.name == "Group"
    assert column_label("rule") == "Rule"


def test_flag_text():
    assert flag_text("parity gap > 0.05; equalised odds gap > 0.05") == (
        "Parity gap > 0.05; Equalised odds gap > 0.05"
    )
    assert flag_text("") == ""


def comparison(xgb_auc):
    return pd.DataFrame(
        {"auc": [0.740, 0.745, xgb_auc]}, index=["logistic_regression", "random_forest", "xgboost"]
    )


def test_model_choice_reason_uses_display_names():
    reason = model_choice_reason(comparison(0.757), "logistic_regression", 0.01)
    assert reason.startswith(
        "XGBoost has the highest validation AUC (0.757), 0.017 above logistic regression"
    )
    assert "xgboost" not in reason and "_" not in reason


def test_model_choice_reason_when_the_gain_is_too_small():
    reason = model_choice_reason(comparison(0.745), "logistic_regression", 0.01)
    assert reason.startswith("Random forest has the highest validation AUC, but beats logistic regression")
    assert "baseline is kept" in reason
