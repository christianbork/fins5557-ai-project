"""Tests for src/features.py, using the small fake dataset from test_data.py."""
import numpy as np
import pandas as pd
import pytest

from src.config import ENGINEERED_FEATURES, PROTECTED, TARGET
from src.data import clean, split
from src.features import add_features, build_preprocessor, feature_columns, safe_divide
from tests.test_data import make_fake


def test_safe_divide_zero_and_negative_denominators():
    out = safe_divide([10, 10, 10], [2, 0, -5], fill=1.0)
    np.testing.assert_array_equal(out, [5.0, 1.0, 1.0])


def test_zero_and_negative_bills_give_finite_features():
    df = make_fake(6)
    df["BILL_AMT2"] = [0, -500, 0, -1, 100, 1]  # payment_ratio uses the previous bill
    df["BILL_AMT1"] = [0, -500, 0, -1, 100, 1]
    df["PAY_AMT1"] = [0, 200, 500, 0, 50, 10_000]
    out = add_features(df)
    for col in ENGINEERED_FEATURES:
        assert np.isfinite(out[col]).all(), col
    # Nothing owed counts as fully paid; overpaying a tiny bill is capped at 1.
    assert out["payment_ratio"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.5, 1.0]


def test_months_late_counts():
    df = make_fake(2)
    df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] = [[2, 1, 0, -1, -2, 3], [-1] * 6]
    out = add_features(df)
    assert out["months_late_6m"].tolist() == [3, 0]
    assert out["max_months_late"].tolist() == [3, 0]


def test_add_features_does_not_change_the_input():
    df = make_fake()
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_pipeline_same_columns_on_train_and_validation():
    train, val, _ = split(add_features(clean(make_fake(2000))))
    pre = build_preprocessor()
    X_train = pre.fit_transform(train)
    X_val = pre.transform(val)
    assert X_train.shape[1] == X_val.shape[1]
    assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
    names = list(pre.get_feature_names_out())
    assert not set(PROTECTED + [TARGET]) & set(names)
    assert {"EDUCATION_1", "EDUCATION_4"} <= set(names)


def test_unseen_category_does_not_crash():
    df = add_features(clean(make_fake(200)))
    pre = build_preprocessor().fit(df)
    new = df.head(1).copy()
    new["EDUCATION"] = 9  # not seen in training
    out = pre.transform(new)
    assert out.shape[1] == len(pre.get_feature_names_out())


def test_feature_sets_exclude_protected_and_unknown_set_errors():
    numeric, categorical = feature_columns()
    assert not set(PROTECTED) & set(numeric + categorical)
    with pytest.raises(ValueError, match="Unknown feature set"):
        feature_columns("nope")
