"""Tests for src/features.py, using the small fake dataset from test_data.py."""
import numpy as np
import pandas as pd
import pytest

from src.config import DAYS_EMPLOYED_SENTINEL, ENGINEERED_FEATURES, ID_COLUMN, PROTECTED, TARGET
from src.data import clean, split
from src.features import add_features, build_preprocessor, feature_columns, safe_divide
from tests.test_data import make_fake


def test_safe_divide_bad_denominators_give_missing():
    out = safe_divide([10, 10, 10, 10], [2, 0, -5, np.nan])
    assert out[0] == 5.0
    assert np.isnan(out[1:]).all()


def test_engineered_features_values():
    df = make_fake(2)
    df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "CNT_FAM_MEMBERS"]] = [
        [100_000, 300_000, 15_000, 270_000, 2], [50_000, 100_000, 0, 100_000, 1]]
    df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = [[np.nan, 0.4, 0.6], [np.nan, np.nan, np.nan]]
    out = add_features(df)
    assert out["credit_income_ratio"].tolist() == [3.0, 2.0]
    assert out["annuity_income_ratio"].iloc[0] == pytest.approx(0.15)
    assert out["credit_term"].iloc[0] == 20.0 and np.isnan(out["credit_term"].iloc[1])  # no repayment amount
    assert out["goods_credit_ratio"].iloc[0] == pytest.approx(0.9)
    assert out["ext_source_mean"].iloc[0] == pytest.approx(0.5) and np.isnan(out["ext_source_mean"].iloc[1])
    assert out["ext_source_missing"].tolist() == [1, 3]
    assert out["income_per_person"].tolist() == [50_000, 50_000]


def test_not_employed_code_is_not_read_as_days():
    df = make_fake(3)
    df["DAYS_EMPLOYED"] = [DAYS_EMPLOYED_SENTINEL, -730.5, np.nan]
    out = add_features(df)
    assert out["not_employed"].tolist() == [1, 0, 1]
    assert np.isnan(out["years_employed"].iloc[0]) and out["years_employed"].iloc[1] == pytest.approx(2.0)
    assert np.isnan(out["DAYS_EMPLOYED"].iloc[0])  # the 365243 code is removed


def test_text_and_none_inputs_from_the_app_are_handled():
    row = make_fake(1).iloc[0].to_dict()
    row["AMT_INCOME_TOTAL"] = "120000"
    row["OWN_CAR_AGE"] = None
    out = add_features(pd.DataFrame([row]))
    assert out["AMT_INCOME_TOTAL"].iloc[0] == 120_000
    assert np.isnan(out["OWN_CAR_AGE"].iloc[0])


def test_add_features_does_not_change_the_input():
    df = make_fake()
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_pipeline_same_columns_no_missing_no_protected():
    train, val, _ = split(add_features(clean(make_fake(2000))))
    pre = build_preprocessor()
    X_train = pre.fit_transform(train)
    X_val = pre.transform(val)
    assert X_train.shape[1] == X_val.shape[1]
    assert np.isfinite(X_train).all() and np.isfinite(X_val).all()
    names = list(pre.get_feature_names_out())
    assert not {*PROTECTED, TARGET, ID_COLUMN} & set(names)
    assert "missingindicator_EXT_SOURCE_1" in names  # missingness kept as information
    assert any(n.startswith("NAME_EDUCATION_TYPE_") for n in names)


def test_unseen_category_does_not_crash():
    df = add_features(clean(make_fake(300)))
    pre = build_preprocessor().fit(df)
    new = df.head(1).copy()
    new["ORGANIZATION_TYPE"] = "Space agency"  # not seen in training
    out = pre.transform(new)
    assert out.shape[1] == len(pre.get_feature_names_out())


def test_all_engineered_features_are_model_inputs():
    numeric, _ = feature_columns()
    assert set(ENGINEERED_FEATURES) <= set(numeric)


def test_feature_sets_exclude_protected_and_unknown_set_errors():
    numeric, categorical = feature_columns()
    assert not set(PROTECTED) & set(numeric + categorical)
    with pytest.raises(ValueError, match="Unknown feature set"):
        feature_columns("nope")
