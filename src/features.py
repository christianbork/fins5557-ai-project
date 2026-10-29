"""Station 2 (features): engineered credit-risk features and the preprocessing pipeline.

Typical use (notebooks and src/train.py):

    from src.features import add_features, build_preprocessor

    train = add_features(train)
    pre = build_preprocessor()       # fitted later, on training data only, inside the model pipeline
    X_train_ready = pre.fit_transform(train)

The pipeline pattern (preprocessing + model in one sklearn object) follows Code Bank W06-04,
extended with a ColumnTransformer so numeric and categorical columns are treated differently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_INPUTS,
    DAYS_EMPLOYED_SENTINEL,
    FEATURE_SET,
    FEATURE_SETS,
    NUMERIC_INPUTS,
    RARE_CATEGORY_SHARE,
)
from src.data import text_or_nan

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def safe_divide(numerator, denominator) -> np.ndarray:
    """numerator / denominator, missing (NaN) wherever the denominator is zero, negative or missing.

    Missing is the honest answer here (e.g. no repayment amount means no loan term), and the
    pipeline's imputer then fills it and adds a "was missing" flag the model can use.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=np.nan_to_num(den, nan=0.0) > 0)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered features added (see ENGINEERED_FEATURES in config).

    Each one captures a standard credit-risk idea:
    - credit_income_ratio: how big the loan is relative to income (leverage).
    - annuity_income_ratio: share of income that goes on repayments, the classic affordability
      (debt-service) ratio responsible-lending rules look at.
    - credit_term: loan / repayment, roughly how many repayments; longer loans carry more risk.
    - goods_credit_ratio: goods price / loan. Below 1 means the applicant borrows extra cash on
      top of the goods, which tends to be riskier.
    - years_employed / not_employed: job stability. Home Credit codes "not employed" (mostly
      pensioners) as DAYS_EMPLOYED = 365243, which must not be read as 1,000 years in a job.
    - ext_source_mean / ext_source_missing: the external credit scores, and how many are missing.
      Missing scores mean little credit history: the thin-file problem.
    - income_per_person: income spread over the household.
    """
    out = df.copy()
    for col in NUMERIC_INPUTS:  # values typed into the app may arrive as text or None
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_INPUTS:  # None / blank -> NaN, which the pipeline's imputer fills
        out[col] = text_or_nan(out[col])

    out["credit_income_ratio"] = safe_divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["annuity_income_ratio"] = safe_divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["credit_term"] = safe_divide(out["AMT_CREDIT"], out["AMT_ANNUITY"])
    out["goods_credit_ratio"] = safe_divide(out["AMT_GOODS_PRICE"], out["AMT_CREDIT"])

    not_employed = out["DAYS_EMPLOYED"].isna() | (out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL)
    out["not_employed"] = not_employed.astype(int)
    out["years_employed"] = (-out["DAYS_EMPLOYED"] / 365.25).where(~not_employed)
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].where(~not_employed)  # the code isn't a real number of days

    out["ext_source_mean"] = out[EXT_SOURCES].mean(axis=1)  # mean of the available scores
    out["ext_source_missing"] = out[EXT_SOURCES].isna().sum(axis=1)
    out["income_per_person"] = safe_divide(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])
    return out


def feature_columns(feature_set: str = FEATURE_SET) -> tuple[list[str], list[str]]:
    """(numeric, categorical) model input columns for the chosen feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Options: {list(FEATURE_SETS)}")
    cols = FEATURE_SETS[feature_set]
    return list(cols["numeric"]), list(cols["categorical"])


def build_preprocessor(feature_set: str = FEATURE_SET) -> ColumnTransformer:
    """Unfitted ColumnTransformer that turns the raw + engineered columns into model inputs.

    Numeric: fill missing values with the training median and add a 0/1 "was missing" column
    (missingness carries information here, e.g. no external score), then scale (needed for
    logistic regression, harmless for trees).
    Categorical: missing becomes "Unknown", categories rarer than RARE_CATEGORY_SHARE are grouped,
    then one-hot encoded. An unseen category at prediction time becomes all zeros instead of
    crashing the app.
    Columns not in the feature set (the ID, the target, the protected attributes) are dropped.
    It must be fitted on training data only, which happens automatically inside the model Pipeline.
    """
    numeric, categorical = feature_columns(feature_set)
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=RARE_CATEGORY_SHARE,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_steps, numeric), ("cat", categorical_steps, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
