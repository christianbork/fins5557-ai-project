"""Tests for src/applicant.py: the form (years, a 'not employed' tick) <-> model fields (days, codes)."""

import pandas as pd
import pytest

from src.applicant import (
    FIELDS,
    NOT_EMPLOYED,
    YEAR_FIELDS,
    friendly_error,
    label,
    to_form_values,
    to_model_inputs,
)
from src.config import DATA_DIR, DAYS_EMPLOYED_SENTINEL, RAW_INPUTS, TARGET
from tests.test_data import make_fake


def test_every_model_input_has_a_form_field():
    form_fields = {YEAR_FIELDS.get(f, f) for f in RAW_INPUTS}
    assert form_fields <= set(FIELDS)
    assert set(FIELDS) == form_fields | {NOT_EMPLOYED}


def test_years_become_negative_days():
    model = to_model_inputs({"YEARS_EMPLOYED": 6, "YEARS_SINCE_REGISTRATION": 12, "YEARS_SINCE_ID": 0.5})
    assert model["DAYS_EMPLOYED"] == -2192
    assert model["DAYS_REGISTRATION"] == -4383
    assert model["DAYS_ID_PUBLISH"] == -183


def test_not_employed_tick_gives_the_home_credit_code():
    model = to_model_inputs({NOT_EMPLOYED: True, "YEARS_EMPLOYED": 4})
    assert model["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL


def test_blanks_stay_blank():
    model = to_model_inputs({"EXT_SOURCE_1": None, "OCCUPATION_TYPE": "", "YEARS_EMPLOYED": None})
    assert (
        model["EXT_SOURCE_1"] is None and model["OCCUPATION_TYPE"] is None and model["DAYS_EMPLOYED"] is None
    )
    assert set(model) == set(RAW_INPUTS)


def test_round_trip_keeps_the_applicant_within_a_day():
    original = make_fake(20, seed=9).drop(columns=TARGET)[RAW_INPUTS]
    for _, row in original.iterrows():
        back = to_model_inputs(to_form_values(row.to_dict()))
        for field in YEAR_FIELDS:
            if row[field] == DAYS_EMPLOYED_SENTINEL:
                assert back[field] == DAYS_EMPLOYED_SENTINEL
            else:
                assert back[field] == pytest.approx(row[field], abs=19)  # years are rounded to 0.1 (~18 days)


def test_example_applicants_are_complete():
    examples = pd.read_csv(DATA_DIR / "sample" / "sample_applicants.csv")
    assert list(examples.columns) == ["example", *RAW_INPUTS]
    assert examples["example"].is_unique


def test_errors_use_form_labels():
    assert friendly_error("EXT_SOURCE_2: must be between 0 and 1 (got 1.3)") == (
        "External credit score 2: must be between 0 and 1 (got 1.3)"
    )
    assert label("DAYS_EMPLOYED") == "Years in current job"
    assert friendly_error("Something else went wrong") == "Something else went wrong"
