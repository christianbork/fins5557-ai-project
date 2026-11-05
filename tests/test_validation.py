"""Tests for src/validation.py: input errors, the out-of-range guard and the reference statistics."""

import pandas as pd
import pytest

from src.config import CATEGORICAL_INPUTS, MAX_BATCH_ROWS, MAX_UPLOAD_MB, NUMERIC_INPUTS, RAW_INPUTS, TARGET
from src.data import features_target
from src.predict import load_artifacts, predict, predict_batch
from src.validation import (
    NUMERIC_RULES,
    build_reference_stats,
    out_of_range,
    read_csv_upload,
    scale_notes,
    typical_range,
    validate_applicant,
    validate_batch,
)
from tests.test_data import make_fake


@pytest.fixture(scope="module")
def reference(trained):
    train = trained[2]
    return build_reference_stats(features_target(train)[0])


@pytest.fixture
def applicant():
    return make_fake(1, seed=5)[RAW_INPUTS].iloc[0].to_dict()


@pytest.fixture
def batch():
    return make_fake(20, seed=6)[RAW_INPUTS]


def test_every_numeric_input_has_a_rule():
    assert set(NUMERIC_RULES) == set(NUMERIC_INPUTS)


def test_reference_stats_have_no_rows(reference):
    assert set(reference) == {
        "rows",
        "guard_quantiles",
        "ratio_guard_quantiles",
        "numeric",
        "ratios",
        "no_score_share",
        "categorical",
    }
    assert set(reference["numeric"]) == set(NUMERIC_INPUTS)
    assert set(reference["categorical"]) == set(CATEGORICAL_INPUTS)
    assert all(len(s["deciles"]) == 9 for s in reference["numeric"].values())


def test_valid_applicant_has_no_errors(applicant, reference):
    assert validate_applicant(applicant, reference) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("AMT_CREDIT", -5000, "AMT_CREDIT: must be at least 1"),
        ("AMT_INCOME_TOTAL", None, "AMT_INCOME_TOTAL: is required"),
        ("AMT_INCOME_TOTAL", "lots", "AMT_INCOME_TOTAL: must be a number"),
        ("EXT_SOURCE_2", 1.3, "EXT_SOURCE_2: must be between 0 and 1"),
        ("REGION_RATING_CLIENT_W_CITY", 4, "REGION_RATING_CLIENT_W_CITY: must be between 1 and 3"),
        ("CNT_CHILDREN", 1.5, "CNT_CHILDREN: must be a whole number"),
        ("DAYS_EMPLOYED", 500, "DAYS_EMPLOYED: must be between -25000 and 0 (or 365243 for not employed)"),
        (
            "NAME_CONTRACT_TYPE",
            "Mortgage",
            "NAME_CONTRACT_TYPE: must be one of 'Cash loans', 'Revolving loans'",
        ),
        ("FLAG_OWN_CAR", "", "FLAG_OWN_CAR: is required"),
    ],
)
def test_each_rule_gives_a_clear_error(applicant, reference, field, value, message):
    applicant[field] = value
    errors = validate_applicant(applicant, reference)
    assert len(errors) == 1
    assert errors[0].startswith(message)


def test_error_shows_the_value_entered(applicant, reference):
    applicant["EXT_SOURCE_2"] = 1.3
    assert "(got 1.3)" in validate_applicant(applicant, reference)[0]


def test_repayment_larger_than_loan_is_an_error(applicant, reference):
    applicant["AMT_ANNUITY"] = applicant["AMT_CREDIT"] + 1
    assert validate_applicant(applicant, reference)[0].startswith(
        "AMT_ANNUITY: repayment can't be larger than the loan"
    )


def test_not_employed_code_is_allowed(applicant, reference):
    applicant["DAYS_EMPLOYED"] = 365243
    assert validate_applicant(applicant, reference) == []


def test_optional_fields_may_be_blank(applicant, reference):
    for field in [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "OWN_CAR_AGE",
        "OCCUPATION_TYPE",
        "AMT_GOODS_PRICE",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]:
        applicant[field] = None
    assert validate_applicant(applicant, reference) == []


def test_missing_field_is_named(applicant, reference):
    del applicant["AMT_CREDIT"]
    assert validate_applicant(applicant, reference) == ["AMT_CREDIT: field is missing (use None if unknown)"]


def test_batch_keeps_valid_rows_and_reports_bad_ones(batch, reference):
    batch = batch.copy()
    batch.loc[batch.index[3], "EXT_SOURCE_3"] = 2.0
    batch.loc[batch.index[7], "NAME_HOUSING_TYPE"] = "Castle"
    valid, errors = validate_batch(batch, reference)
    assert len(valid) == 18
    assert list(errors["row"]) == [batch.index[3], batch.index[7]]
    assert list(errors["field"]) == ["EXT_SOURCE_3", "NAME_HOUSING_TYPE"]


def test_batch_with_missing_column_is_rejected(batch, reference):
    with pytest.raises(ValueError, match="Missing columns: AMT_CREDIT"):
        validate_batch(batch.drop(columns="AMT_CREDIT"), reference)


def test_batch_row_limit(reference):
    big = make_fake(MAX_BATCH_ROWS + 1)[RAW_INPUTS]
    with pytest.raises(ValueError, match="limit is 10,000"):
        validate_batch(big, reference)


def test_empty_batch_is_rejected(batch, reference):
    with pytest.raises(ValueError, match="no rows"):
        validate_batch(batch.iloc[:0], reference)


def test_csv_upload_round_trip_with_blank_cells(batch, reference):
    batch = batch.copy()
    batch["EXT_SOURCE_1"] = None  # thin file: written as empty cells
    valid, errors = validate_batch(read_csv_upload(batch.to_csv(index=False).encode()), reference)
    assert errors.empty and len(valid) == len(batch)
    assert valid["EXT_SOURCE_1"].isna().all()


def test_csv_upload_size_limit():
    with pytest.raises(ValueError, match=f"limit is {MAX_UPLOAD_MB} MB"):
        read_csv_upload(b"x" * (MAX_UPLOAD_MB * 1024**2 + 1))


def test_not_a_csv_gives_a_clear_error():
    with pytest.raises(ValueError, match="Couldn't read the file as a CSV"):
        read_csv_upload(b"")


def test_guard_flags_values_outside_training_range(batch, reference):
    batch = batch.copy()
    batch.loc[batch.index[0], "AMT_INCOME_TOTAL"] = reference["numeric"]["AMT_INCOME_TOTAL"]["high"] * 10
    warnings = out_of_range(batch, reference)
    assert warnings.iloc[0][0].startswith("Income ($")
    assert "outside the range of past approved applicants" in warnings.iloc[0][0]


def test_guard_ignores_not_employed_code_and_blanks(batch, reference):
    batch = batch.copy()
    batch["DAYS_EMPLOYED"] = 365243
    batch["EXT_SOURCE_1"] = None
    assert not any(
        "DAYS_EMPLOYED" in w or "EXT_SOURCE_1" in w for ws in out_of_range(batch, reference) for w in ws
    )


def test_guard_turns_approve_into_refer(saved_model_dir):
    artifacts = load_artifacts(
        saved_model_dir / "model.joblib",
        saved_model_dir / "metadata.json",
        saved_model_dir / "reference_stats.json",
    )
    applicants = make_fake(200, seed=7).drop(columns=TARGET)
    normal = predict_batch(applicants, artifacts)
    approved = applicants[normal["decision"].eq("approve") & normal["warnings"].eq("")].head(1)
    assert len(approved) == 1, "need one approved applicant to test with"
    applicant = approved.iloc[0].to_dict()
    applicant["AMT_INCOME_TOTAL"] = artifacts["reference"]["numeric"]["AMT_INCOME_TOTAL"]["high"] * 10
    out = predict(applicant, artifacts)
    assert out["decision"] == "refer"
    assert out["warnings"] and out["warnings"][0].startswith("Income")


def test_guard_never_softens_a_decline(saved_model_dir):
    artifacts = load_artifacts(
        saved_model_dir / "model.joblib",
        saved_model_dir / "metadata.json",
        saved_model_dir / "reference_stats.json",
    )
    applicants = make_fake(300, seed=8).drop(columns=TARGET)
    applicants["AMT_INCOME_TOTAL"] = 1e9  # everyone out of range
    scored = predict_batch(applicants, artifacts)
    threshold, band = artifacts["metadata"]["threshold"], artifacts["metadata"]["refer_band"]
    assert (scored.loc[scored["pd"] >= threshold + band, "decision"] == "decline").all()
    assert not scored["decision"].eq("approve").any()
    assert isinstance(scored["warnings"].iloc[0], str) and scored["warnings"].iloc[0]
    pd.testing.assert_index_equal(scored.index, applicants.index)


# ---- Ratio and no-score checks (5 Nov, after user testing) -----------------------------------
def _ordinary(reference):
    """A row whose fields and ratios all sit in the middle of the training data."""
    num = reference["numeric"]
    return {
        "AMT_INCOME_TOTAL": num["AMT_INCOME_TOTAL"]["deciles"][4],
        "AMT_CREDIT": num["AMT_CREDIT"]["deciles"][4],
        "AMT_ANNUITY": num["AMT_CREDIT"]["deciles"][4] / 20,
        "AMT_GOODS_PRICE": num["AMT_CREDIT"]["deciles"][4] / 1.1,
        "EXT_SOURCE_1": None,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": None,
    }


def _ratio_warnings(row, reference):
    df = pd.DataFrame([row])
    for col in NUMERIC_INPUTS:  # the other fields at their median, so only the tested change shows
        if col not in df:
            df[col] = reference["numeric"][col]["deciles"][4]
    return out_of_range(df, reference).iloc[0]


def test_reference_stats_hold_ratio_ranges(reference):
    assert set(reference["ratios"]) == {
        "instalments",
        "loan_to_goods",
        "repayment_to_income",
        "loan_to_income",
    }
    for stats in reference["ratios"].values():
        assert stats["low"] <= stats["high"]
    assert 0 <= reference["no_score_share"] <= 1


def test_ordinary_ratios_give_no_warning(reference):
    assert _ratio_warnings(_ordinary(reference), reference) == []


def test_unusual_combination_is_flagged_though_each_field_is_normal(reference):
    """The tester's case: normal fields, but a loan twice the goods price."""
    row = _ordinary(reference)
    row["AMT_GOODS_PRICE"] = row["AMT_CREDIT"] / 2
    assert any(
        w.startswith("Loan relative to the price of the goods") for w in _ratio_warnings(row, reference)
    )


def test_too_many_instalments_is_flagged(reference):
    row = _ordinary(reference)
    row["AMT_ANNUITY"] = row["AMT_CREDIT"] / (reference["ratios"]["instalments"]["high"] * 1.2)
    assert any(w.startswith("Number of instalments") for w in _ratio_warnings(row, reference))


def test_blank_goods_price_is_not_a_ratio_warning(reference):
    row = _ordinary(reference)
    row["AMT_GOODS_PRICE"] = None
    assert not any("goods" in w for w in _ratio_warnings(row, reference))


def test_no_external_score_at_all_is_flagged(reference):
    row = _ordinary(reference)
    row["EXT_SOURCE_2"] = None
    assert any("no external credit score" in w for w in _ratio_warnings(row, reference))


def test_old_reference_files_without_ratios_still_work(reference):
    old = {k: v for k, v in reference.items() if k not in ("ratios", "no_score_share")}
    assert _ratio_warnings(_ordinary(reference), old) == []


def test_typical_range_is_the_middle_80_percent(reference):
    low, high = typical_range("AMT_INCOME_TOTAL", reference)
    deciles = reference["numeric"]["AMT_INCOME_TOTAL"]["deciles"]
    assert (low, high) == (deciles[0], deciles[-1])
    assert typical_range("NOT_A_FIELD", reference) is None


def test_scale_note_for_an_amount_outside_the_middle_98_percent(reference):
    row = _ordinary(reference)
    assert scale_notes(row, reference) == []
    row["AMT_INCOME_TOTAL"] = reference["numeric"]["AMT_INCOME_TOTAL"]["typical_low"] / 2
    notes = scale_notes(row, reference)
    assert len(notes) == 1 and "lower than 99% of past applicants" in notes[0]
