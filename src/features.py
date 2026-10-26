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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FEATURE_SET, FEATURE_SETS
from src.data import BILL_AMT, PAY_STATUS


def safe_divide(numerator, denominator, fill: float = 0.0) -> np.ndarray:
    """numerator / denominator, with `fill` wherever the denominator is zero or negative."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(num.shape, fill, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered features added (see ENGINEERED_FEATURES in config).

    Each one captures a standard credit-risk idea:
    - utilisation: how much of the limit is used. High utilisation signals financial stress.
    - payment_ratio: share of the previous bill paid (PAY_AMT1 / BILL_AMT2). Paying only a
      small part means revolving debt.
      A bill of zero or less (nothing owed, or a credit balance) counts as fully paid (1.0).
      Capped at 1, so overpaying doesn't produce huge values from tiny bills.
    - months_late_6m / max_months_late: how often and how badly the customer was in arrears.
    - avg_utilisation_6m: utilisation smoothed over 6 months, less noisy than one month.
    - bill_trend: whether the balance is growing (positive) or being paid down (negative).
    """
    out = df.copy()
    limit = out["LIMIT_BAL"]

    out["utilisation"] = safe_divide(out["BILL_AMT1"], limit)
    # PAY_AMT1 is paid in September against the August statement (BILL_AMT2), not September's.
    out["payment_ratio"] = np.clip(safe_divide(out["PAY_AMT1"], out["BILL_AMT2"], fill=1.0), 0, 1)

    late = out[PAY_STATUS] > 0  # PAY_* of 1..9 = months late; -2, -1, 0 = not late
    out["months_late_6m"] = late.sum(axis=1)
    out["max_months_late"] = out[PAY_STATUS].max(axis=1).clip(lower=0)

    monthly_util = np.column_stack([safe_divide(out[col], limit) for col in BILL_AMT])
    out["avg_utilisation_6m"] = monthly_util.mean(axis=1)
    out["bill_trend"] = out["BILL_AMT1"] - out["BILL_AMT6"]
    return out


def feature_columns(feature_set: str = FEATURE_SET) -> tuple[list[str], list[str]]:
    """(numeric, categorical) model input columns for the chosen feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Options: {list(FEATURE_SETS)}")
    cols = FEATURE_SETS[feature_set]
    return list(cols["numeric"]), list(cols["categorical"])


def build_preprocessor(feature_set: str = FEATURE_SET) -> ColumnTransformer:
    """Unfitted ColumnTransformer: scale numeric columns, one-hot encode categorical ones.

    Scaling matters for logistic regression (amounts range from -165k to ~1m NT$); tree
    models ignore it, so one preprocessor works for all three models. Columns not in the
    feature set (including the protected attributes and the target) are dropped.
    It must be fitted on training data only, which happens automatically when it is the
    first step of a model Pipeline.
    """
    numeric, categorical = feature_columns(feature_set)
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            # handle_unknown="ignore": an unseen code at prediction time becomes all zeros
            # instead of crashing the app.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
