"""Tests for src/data.py. They use a small fake dataset shaped like Home Credit's
application_train.csv, so no download is needed (CI never sees the real data)."""
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_INPUTS,
    DAYS_EMPLOYED_SENTINEL,
    ID_COLUMN,
    PROTECTED,
    RAW_INPUTS,
    RAW_TARGET,
    TARGET,
)
from src.data import age_years, clean, download_raw, features_target, load_raw, split, validate_schema

CATEGORY_VALUES = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "FLAG_OWN_CAR": ["Y", "N"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_INCOME_TYPE": ["Working", "Pensioner", "Commercial associate", "State servant"],
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education", "Incomplete higher",
                            "Lower secondary"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Rented apartment"],
    "OCCUPATION_TYPE": ["Laborers", "Sales staff", "Core staff", "Managers", None],
    "ORGANIZATION_TYPE": ["Business Entity Type 3", "Self-employed", "School", "XNA", "Bank"],
}


def make_fake(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Fake applications with the columns, codes and missing values of the real file (after loading)."""
    rng = np.random.default_rng(seed)
    income = rng.integers(5, 60, n) * 5_000.0
    credit = income * rng.uniform(0.5, 6, n)
    df = pd.DataFrame({
        ID_COLUMN: np.arange(100_000, 100_000 + n),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit.round(),
        "AMT_ANNUITY": (credit / rng.integers(10, 60, n)).round(),
        "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, n)).round(),
        "CNT_CHILDREN": rng.integers(0, 4, n),
        "DAYS_EMPLOYED": np.where(rng.random(n) < 0.18, DAYS_EMPLOYED_SENTINEL, -rng.integers(30, 12_000, n)),
        "DAYS_REGISTRATION": -rng.integers(0, 20_000, n).astype(float),
        "DAYS_ID_PUBLISH": -rng.integers(0, 7_000, n),
        "OWN_CAR_AGE": np.where(rng.random(n) < 0.65, np.nan, rng.integers(0, 30, n)),
        "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, n),
        "EXT_SOURCE_1": np.where(rng.random(n) < 0.56, np.nan, rng.random(n)),
        "EXT_SOURCE_2": np.where(rng.random(n) < 0.01, np.nan, rng.random(n)),
        "EXT_SOURCE_3": np.where(rng.random(n) < 0.20, np.nan, rng.random(n)),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(n) < 0.13, np.nan, rng.integers(0, 6, n)),
        "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.66, 0.34]),
        "DAYS_BIRTH": -rng.integers(21 * 365, 69 * 365, n),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage", "Widow"], n),
    })
    df["CNT_FAM_MEMBERS"] = (df["CNT_CHILDREN"] + rng.integers(1, 3, n)).astype(float)
    for col, values in CATEGORY_VALUES.items():
        df[col] = rng.choice(np.array(values, dtype=object), n)
    # Default is more likely with low external scores, so the fake models have something to learn.
    score = df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1).fillna(0.5)
    df[TARGET] = (rng.random(n) < 0.16 * (1.2 - score)).astype(int)
    return df[[ID_COLUMN, TARGET, *RAW_INPUTS, *PROTECTED]]


def test_load_raw_reads_used_columns_and_renames_target(tmp_path):
    raw = make_fake(20).rename(columns={TARGET: RAW_TARGET}).assign(APARTMENTS_AVG=0.1)
    path = tmp_path / "application_train.csv"
    raw.to_csv(path, index=False)
    df = load_raw(path)
    assert TARGET in df.columns and RAW_TARGET not in df.columns
    assert "APARTMENTS_AVG" not in df.columns  # columns we don't use aren't loaded


def test_load_raw_missing_file_explains_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle.com"):
        load_raw(tmp_path / "nope.csv")


def test_clean_tidies_placeholder_codes():
    df = make_fake(10)
    df.loc[0, "CODE_GENDER"] = "XNA"
    df.loc[1, "NAME_FAMILY_STATUS"] = "Unknown"
    out = clean(df)
    assert pd.isna(out.loc[0, "CODE_GENDER"])
    assert pd.isna(out.loc[1, "NAME_FAMILY_STATUS"])


def test_clean_does_not_change_the_input():
    df = make_fake()
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)


def test_split_sizes_no_overlap_and_stratified():
    df = make_fake(3000)
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.6) < 0.01
    assert not set(train[ID_COLUMN]) & set(val[ID_COLUMN])
    assert not set(train[ID_COLUMN]) & set(test[ID_COLUMN])
    rate = df[TARGET].mean()
    for part in (train, val, test):
        assert abs(part[TARGET].mean() - rate) < 0.01


def test_features_are_only_the_listed_inputs():
    X, y = features_target(make_fake())
    assert list(X.columns) == RAW_INPUTS
    assert not {TARGET, ID_COLUMN, *PROTECTED} & set(X.columns)
    assert len(X) == len(y)


def test_validate_schema_catches_problems():
    df = make_fake()
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df.drop(columns=["AMT_CREDIT"]))
    bad = df.copy()
    bad.loc[0, "AMT_CREDIT"] = 0
    with pytest.raises(ValueError, match="AMT_CREDIT"):
        validate_schema(bad)
    bad = df.copy()
    bad.loc[1, ID_COLUMN] = bad.loc[0, ID_COLUMN]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_schema(bad)
    bad = df.copy()
    bad[TARGET] = bad[TARGET].astype(float)
    bad.loc[0, TARGET] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_schema(bad)


def test_missing_inputs_are_allowed():
    """Home Credit inputs often have gaps (e.g. EXT_SOURCE_1); the pipeline imputes them."""
    df = make_fake(50)
    df.loc[:10, "OCCUPATION_TYPE"] = None
    clean(df)  # no error


def test_age_years():
    df = pd.DataFrame({"DAYS_BIRTH": [-365.25 * 30, -365.25 * 45.5]})
    assert age_years(df).tolist() == [30.0, 45.5]


def test_categorical_inputs_are_covered_by_fake_data():
    assert set(CATEGORY_VALUES) == set(CATEGORICAL_INPUTS)


def test_download_fetches_only_the_application_file(tmp_path, monkeypatch):
    """kagglehub is faked: no network or Kaggle login is needed."""
    import sys
    import types

    source = tmp_path / "cache" / "application_train.csv"
    source.parent.mkdir()
    make_fake(5).rename(columns={TARGET: RAW_TARGET}).to_csv(source, index=False)
    calls = []

    def fake_download(handle, path=None, force_download=False):
        calls.append((handle, path))
        return str(source)

    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(competition_download=fake_download))
    out = download_raw(tmp_path / "raw" / "application_train.csv")
    assert calls == [("home-credit-default-risk", "application_train.csv")]
    assert len(load_raw(out)) == 5
    download_raw(out)  # file already there: no second download
    assert len(calls) == 1
