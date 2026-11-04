"""Station 1 (ETL). Load, check, clean and split the Home Credit application data.

Typical use (notebooks and src/train.py):

    from src.data import load_clean, split, features_target

    df = load_clean()                # reads data/raw/application_train.csv
    train, val, test = split(df)     # stratified 60/20/20
    X_train, y_train = features_target(train)

Getting the data (each teammate does this with their own Kaggle account; the file is never shared):

    1. On kaggle.com: log in, verify your phone and accept the Home Credit competition rules.
    2. Once per laptop: save a Kaggle API token to ~/.kaggle/access_token (command in the README;
       kagglehub.login() in a terminal only lasts for that one Python process)
    3. python -m src.data      downloads application_train.csv into data/raw/

The competition rules don't allow redistribution, so the file is gitignored and never committed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORICAL_INPUTS,
    ID_COLUMN,
    KAGGLE_URL,
    NUMERIC_INPUTS,
    PROTECTED,
    RANDOM_STATE,
    RAW_INPUTS,
    RAW_PATH,
    RAW_TARGET,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)

# Only these columns are read from the 122 in the file: faster and uses far less memory.
COLUMNS = [ID_COLUMN, RAW_TARGET, *RAW_INPUTS, *PROTECTED]


KAGGLE_COMPETITION = "home-credit-default-risk"
RAW_FILE = "application_train.csv"


def download_raw(path: Path = RAW_PATH, force: bool = False) -> Path:
    """Download application_train.csv from Kaggle with kagglehub and copy it to data/raw/.

    Only this one file is fetched, not the whole 2.7 GB competition. Needs a Kaggle login
    (~/.kaggle/access_token, or KAGGLE_USERNAME / KAGGLE_KEY in .env) and the competition rules
    accepted on kaggle.com; kagglehub's own error says which one is missing.
    """
    path = Path(path)
    if path.exists() and not force:
        return path
    import kagglehub  # dev-only package: the deployed app never downloads data

    downloaded = kagglehub.competition_download(KAGGLE_COMPETITION, path=RAW_FILE, force_download=force)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, path)
    return path


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the columns we use from application_train.csv and rename TARGET to `default`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Accept the competition rules at {KAGGLE_URL}, then run "
            "`python -m src.data` to download it (see the instructions at the top of src/data.py)."
        )
    df = pd.read_csv(path, usecols=lambda c: c in COLUMNS)
    return df.rename(columns={RAW_TARGET: TARGET})


def validate_schema(df: pd.DataFrame) -> None:
    """Fail early, with a clear message, if the data isn't what the pipeline expects.

    Missing values in the inputs are allowed (Home Credit has many, e.g. EXT_SOURCE_1); the
    model pipeline imputes them. The ID, target and loan amounts must be complete.
    """
    required = [ID_COLUMN, TARGET, *RAW_INPUTS, *PROTECTED]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col in (ID_COLUMN, TARGET, "AMT_CREDIT", "AMT_INCOME_TOTAL"):
        if df[col].isna().any():
            raise ValueError(f"Unexpected missing values in {col} ({int(df[col].isna().sum())} rows)")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"'{TARGET}' must be 0/1, found {sorted(df[TARGET].unique())}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate {ID_COLUMN} values: each application should appear once")
    for col in ("AMT_CREDIT", "AMT_INCOME_TOTAL"):
        if (df[col] <= 0).any():
            raise ValueError(f"{col} must be positive")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Check the data and tidy a few codes. Everything else is handled inside the model pipeline.

    - CODE_GENDER "XNA" (4 rows) and NAME_FAMILY_STATUS "Unknown" (2 rows) become missing:
      they are placeholders, not categories. These columns are only used for the fairness audit.
    - Text categories get surrounding spaces stripped.
    - DAYS_EMPLOYED = 365243 ("not employed", about 18% of rows) is left as it is here and
      turned into a flag plus a missing value in src/features.py, so the app and the training
      data go through exactly the same step.
    """
    validate_schema(df)
    out = df.copy()
    for col in [*CATEGORICAL_INPUTS, "CODE_GENDER", "NAME_FAMILY_STATUS"]:
        out[col] = text_or_nan(out[col])
    out["CODE_GENDER"] = out["CODE_GENDER"].replace("XNA", np.nan)
    out["NAME_FAMILY_STATUS"] = out["NAME_FAMILY_STATUS"].replace("Unknown", np.nan)
    for col in NUMERIC_INPUTS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def text_or_nan(values: pd.Series) -> pd.Series:
    """Text with surrounding spaces stripped; None, pd.NA and blanks become NaN (what sklearn expects)."""
    text = values.astype(object).where(values.notna(), np.nan)
    stripped = text.map(lambda v: v.strip() if isinstance(v, str) else v)
    return stripped.replace("", np.nan).astype(object)


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    """Raw data, checked and cleaned: the starting point for every notebook and model."""
    return clean(load_raw(path))


def split(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """Stratified 60/20/20 train/validation/test split.

    Done before any fitting (imputation, scaling, encoding) so no information from the
    validation or test rows leaks into training. Stratifying keeps the ~8% default rate the
    same in all three parts.
    """
    train, temp = train_test_split(df, test_size=TEST_SIZE + VAL_SIZE, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(
        temp, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE), stratify=temp[TARGET], random_state=seed
    )
    return train, val, test


def features_target(df: pd.DataFrame):
    """Model inputs (X) and target (y).

    X holds only the raw inputs listed in config (RAW_INPUTS), so the ID, the target and the
    protected attributes can never reach the model, and the app only has to collect these fields.
    """
    return df[RAW_INPUTS].copy(), df[TARGET]


def age_years(df: pd.DataFrame) -> pd.Series:
    """Applicant age in years from DAYS_BIRTH (for the fairness audit and EDA only)."""
    return (-df["DAYS_BIRTH"] / 365.25).round(1)


if __name__ == "__main__":
    out = download_raw()
    df = load_clean(out)
    print(f"{out}: {len(df):,} applications, default rate {df[TARGET].mean():.1%}")
