"""Tests for src/monitoring.py: PSI maths, bins, statuses and the synthetic demo batches."""

import math

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_INPUTS
from src.data import features_target
from src.monitoring import (
    RARE,
    UNSEEN,
    bin_detail,
    categorical_shares,
    compute_psi,
    numeric_bin_shares,
    psi_status,
    psi_table,
    synthetic_batch,
)
from src.validation import build_reference_stats, validate_batch


@pytest.fixture(scope="module")
def train_inputs(trained):
    return features_target(trained[2])[0]


@pytest.fixture(scope="module")
def reference(train_inputs):
    return build_reference_stats(train_inputs)


def test_psi_of_identical_distributions_is_zero(train_inputs, reference):
    assert compute_psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0
    assert psi_table(train_inputs, reference)["psi"].abs().max() < 1e-9  # the training data against itself


def test_psi_by_hand():
    # (0.25 - 0.5) ln(0.25 / 0.5) + (0.75 - 0.5) ln(0.75 / 0.5)
    expected = -0.25 * math.log(0.5) + 0.25 * math.log(1.5)
    assert compute_psi([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected)
    assert compute_psi([0.5, 0.5], [0.0, 1.0]) > 5  # an empty bin is floored, not an error


@pytest.mark.parametrize(
    ("psi", "status"), [(0.0, "stable"), (0.099, "stable"), (0.1, "watch"), (0.25, "watch"), (0.26, "alert")]
)
def test_status_bands(psi, status):
    assert psi_status(psi) == status


def test_numeric_bins_with_sentinel_and_missing():
    values = pd.Series([1, 2, 3, 4, 365243, np.nan, np.nan, 5])
    shares = numeric_bin_shares(values, edges=[2, 4], sentinel=365243)
    # bins: <= 2 | 2-4 | > 4 | not employed | missing
    np.testing.assert_allclose(shares, [2 / 8, 2 / 8, 1 / 8, 1 / 8, 2 / 8])


def test_more_thin_files_trigger_an_alert(train_inputs, reference):
    batch = train_inputs.copy()
    batch.loc[batch.sample(frac=0.6, random_state=0).index, "EXT_SOURCE_3"] = np.nan
    row = psi_table(batch, reference).set_index("input").loc["EXT_SOURCE_3"]
    assert row["status"] == "alert" and row["missing_batch"] > row["missing_train"]


def test_rare_and_unseen_categories_are_pooled():
    expected = {"Common": 0.9, "Rare A": 0.005, "Rare B": 0.005, "Other": 0.09}
    labels, exp, act = categorical_shares(pd.Series(["Common", "Rare A", "Rare B", "New", None]), expected)
    assert labels == ["Common", "Other", RARE, UNSEEN]
    np.testing.assert_allclose(exp, [0.9, 0.09, 0.01, 0.0])
    np.testing.assert_allclose(act, [0.2, 0.0, 0.4, 0.4])  # None -> "Unknown", which training never had


def test_missing_columns_are_named(train_inputs, reference):
    with pytest.raises(ValueError, match="Missing columns: AMT_CREDIT"):
        psi_table(train_inputs.drop(columns="AMT_CREDIT"), reference)


def test_bin_detail_shares_sum_to_one(train_inputs, reference):
    for field in ("AMT_CREDIT", "DAYS_EMPLOYED", "NAME_HOUSING_TYPE"):
        detail = bin_detail(train_inputs, reference, field)
        assert detail["training"].sum() == pytest.approx(1) and detail["batch"].sum() == pytest.approx(1)


def test_reference_has_psi_bins(reference):
    for stats in reference["numeric"].values():
        assert sum(stats["psi_bins"]["expected"]) == pytest.approx(1, abs=1e-4)
        assert stats["psi_bins"]["edges"] == sorted(set(stats["psi_bins"]["edges"]))


def test_normal_synthetic_batch_is_stable_and_shifted_one_alerts(reference):
    normal = psi_table(synthetic_batch(reference, n=2000), reference)
    shifted = psi_table(synthetic_batch(reference, n=2000, shifted=True), reference)
    assert (normal["status"] == "stable").all()
    assert (shifted["status"] == "alert").sum() >= 1
    assert {"EXT_SOURCE_2", "DAYS_EMPLOYED"} <= set(shifted.loc[shifted["status"] == "alert", "input"])


def test_synthetic_batches_are_valid_and_invented(reference, train_inputs):
    for shifted in (False, True):
        batch = synthetic_batch(reference, n=500, shifted=shifted)
        assert list(batch.columns) == RAW_INPUTS
        valid, errors = validate_batch(batch, reference)
        assert errors.empty and len(valid) == 500
        copied = batch.merge(train_inputs.drop_duplicates(), how="inner", on=RAW_INPUTS)
        assert copied.empty  # no training row reproduced


def test_synthetic_batch_is_reproducible(reference):
    pd.testing.assert_frame_equal(synthetic_batch(reference, n=50), synthetic_batch(reference, n=50))
