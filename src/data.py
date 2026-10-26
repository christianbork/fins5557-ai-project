"""Station 1 (ETL): download, check, clean and split the UCI credit card default data.

Typical use (notebooks and src/train.py):

    from src.data import load_clean, split, features_target

    df = load_clean()                # downloads once, then reads data/raw/
    train, val, test = split(df)     # stratified 60/20/20
    X_train, y_train = features_target(train)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROTECTED,
    RANDOM_STATE,
    RAW_PATH,
    TARGET,
    TEST_SIZE,
    UCI_DATASET_ID,
    VAL_SIZE,
)

# The 23 features in UCI's order (the ucimlrepo copy calls them X1..X23).
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # Sep back to Apr 2005
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURE_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_STATUS, *BILL_AMT, *PAY_AMT]

# Documented codes; anything else is grouped into "other".
EDUCATION_CODES = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
MARRIAGE_CODES = {1: "married", 2: "single", 3: "other"}


def download_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Download the dataset from UCI, give the columns readable names and save a CSV copy."""
    from ucimlrepo import fetch_ucirepo  # dev-only package: the deployed app never downloads data

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X = dataset.data.features.copy()
    y = dataset.data.targets

    if list(X.columns) != FEATURE_NAMES:
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} feature columns from UCI, got {X.shape[1]}.")
        X.columns = FEATURE_NAMES  # X1..X23 are in the documented order

    df = X.assign(**{TARGET: y.iloc[:, 0].to_numpy()})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_raw(path: Path = RAW_PATH, download_if_missing: bool = True) -> pd.DataFrame:
    """Read the local CSV, downloading it first if it isn't there yet."""
    if path.exists():
        return pd.read_csv(path)
    if not download_if_missing:
        raise FileNotFoundError(f"{path} not found. Run load_raw() once with internet access.")
    return download_raw(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Fail early, with a clear message, if the data isn't what the pipeline expects."""
    missing = [c for c in [*FEATURE_NAMES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[[*FEATURE_NAMES, TARGET]].isna().any().any():
        nulls = df[[*FEATURE_NAMES, TARGET]].isna().sum()
        raise ValueError(f"Unexpected missing values:\n{nulls[nulls > 0]}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"'{TARGET}' must be 0/1, found {sorted(df[TARGET].unique())}")
    if (df["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL must be positive")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Check the data and group undocumented category codes.

    - EDUCATION 0, 5, 6 are not in the UCI documentation -> 4 ("other").
    - MARRIAGE 0 is not documented -> 3 ("other").
    - PAY_* stay as they are: -1 = paid on time, 1..9 = months late. -2 and 0 are not
      documented; they are usually read as "no spending" and "paid the minimum (revolving)".
    - Negative BILL_AMT values (credit balances) are real and kept; ratio features must
      handle them safely.
    """
    validate_schema(df)
    out = df.copy()
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].isin(list(EDUCATION_CODES)), 4)
    out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"].isin(list(MARRIAGE_CODES)), 3)
    return out


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """Raw data, checked and cleaned: the starting point for every notebook and model."""
    return clean(load_raw(path))


def split(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """Stratified 60/20/20 train/validation/test split.

    Done before any fitting (scaling, encoding, imputation) so no information from the
    validation or test rows leaks into training. Stratifying keeps the ~22% default rate
    the same in all three parts.
    """
    train, temp = train_test_split(
        df, test_size=TEST_SIZE + VAL_SIZE, stratify=df[TARGET], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), stratify=temp[TARGET], random_state=seed
    )
    return train, val, test


def features_target(df: pd.DataFrame):
    """Model inputs (X) and target (y). Protected attributes are removed from X."""
    X = df.drop(columns=[TARGET, *PROTECTED])
    y = df[TARGET]
    return X, y
