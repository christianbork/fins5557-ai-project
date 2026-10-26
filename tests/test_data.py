"""Tests for src/data.py. They use a small fake dataset, so no download is needed."""
import numpy as np
import pandas as pd
import pytest

from src.config import PROTECTED, TARGET
from src.data import FEATURE_NAMES, PAY_STATUS, clean, features_target, split, validate_schema


def make_fake(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "LIMIT_BAL": rng.integers(1, 50, n) * 10_000,
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(0, 7, n),  # includes undocumented 0, 5, 6
        "MARRIAGE": rng.integers(0, 4, n),  # includes undocumented 0
        "AGE": rng.integers(21, 75, n),
    })
    for col in PAY_STATUS:
        df[col] = rng.integers(-2, 9, n)
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.integers(-5_000, 200_000, n)
        df[f"PAY_AMT{i}"] = rng.integers(0, 20_000, n)
    df[TARGET] = (rng.random(n) < 0.22).astype(int)
    return df[[*FEATURE_NAMES, TARGET]]


def test_clean_groups_undocumented_codes():
    out = clean(make_fake())
    assert set(out["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(out["MARRIAGE"].unique()) <= {1, 2, 3}


def test_clean_does_not_change_the_input():
    df = make_fake()
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)


def test_split_sizes_no_overlap_and_stratified():
    df = make_fake(2000)
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert abs(len(train) / len(df) - 0.6) < 0.01
    assert abs(len(val) / len(df) - 0.2) < 0.01
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    rate = df[TARGET].mean()
    for part in (train, val, test):
        assert abs(part[TARGET].mean() - rate) < 0.01


def test_features_exclude_target_and_protected():
    X, y = features_target(make_fake())
    assert TARGET not in X.columns
    assert not set(PROTECTED) & set(X.columns)
    assert len(X) == len(y)


def test_validate_schema_catches_problems():
    df = make_fake()
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df.drop(columns=["AGE"]))
    bad = df.copy()
    bad.loc[0, "LIMIT_BAL"] = 0
    with pytest.raises(ValueError, match="LIMIT_BAL"):
        validate_schema(bad)
    bad = df.copy()
    bad[TARGET] = bad[TARGET].astype(float)
    bad.loc[0, TARGET] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_schema(bad)


def test_download_renames_uci_columns(tmp_path, monkeypatch):
    """The UCI copy names the features X1..X23 and the target Y; we rename and save them."""
    import sys
    import types

    from src.data import download_raw

    fake = make_fake(50)
    features = fake[FEATURE_NAMES].copy()
    features.columns = [f"X{i}" for i in range(1, 24)]
    targets = fake[[TARGET]].rename(columns={TARGET: "Y"})
    dataset = types.SimpleNamespace(data=types.SimpleNamespace(features=features, targets=targets))
    monkeypatch.setitem(sys.modules, "ucimlrepo", types.SimpleNamespace(fetch_ucirepo=lambda id: dataset))

    out_path = tmp_path / "raw.csv"
    df = download_raw(out_path)
    assert list(df.columns) == [*FEATURE_NAMES, TARGET]
    assert out_path.exists()
    pd.testing.assert_frame_equal(pd.read_csv(out_path), df)
