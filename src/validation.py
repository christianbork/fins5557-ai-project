"""Step 7 (input validation): catch bad input before it reaches the model, with clear messages.

Two levels:
- **Errors** (validate_applicant, validate_batch): values that are impossible or that the model
  can't use, e.g. a negative loan, an external score above 1, an unknown category. The applicant
  is not scored, and the message names the field and what is allowed.
- **Warnings** (out_of_range): values that are possible but outside the range of the training
  data (GUARD_QUANTILES). The model only saw loans Home Credit had already approved (selection bias), so
  it hasn't learnt how such applicants behave; predict() refers them to an analyst instead of
  approving them automatically.

Training ranges and category lists come from models/reference_stats.json: summary statistics
of the training split only, no applicant rows. Write it without retraining with:

    python -m src.validation

(src/train.py also writes it). Step 11 (PSI monitoring) reuses its deciles and category shares.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_INPUTS,
    DAYS_EMPLOYED_SENTINEL,
    GUARD_QUANTILES,
    MAX_BATCH_ROWS,
    MAX_UPLOAD_MB,
    NUMERIC_INPUTS,
    RAW_INPUTS,
    REFERENCE_STATS_PATH,
)
from src.formatting import money
from src.labels import MONEY_FIELDS, input_label
from src.monitoring import psi_bins

# field: (lowest allowed, highest allowed or None, required, whole number)
# Hard limits only: values outside them are impossible or meaningless. Unusual but possible
# values (a very high income) are the guard's job, not an error.
NUMERIC_RULES = {
    "AMT_INCOME_TOTAL": (1, None, True, False),
    "AMT_CREDIT": (1, None, True, False),
    "AMT_ANNUITY": (1, None, True, False),
    "AMT_GOODS_PRICE": (1, None, False, False),  # blank: not a loan for goods
    "CNT_CHILDREN": (0, 20, True, True),
    "CNT_FAM_MEMBERS": (1, 25, True, True),
    "DAYS_EMPLOYED": (-25_000, 0, True, True),  # or 365243, Home Credit's "not employed" code
    "DAYS_REGISTRATION": (-30_000, 0, True, True),
    "DAYS_ID_PUBLISH": (-30_000, 0, True, True),
    "OWN_CAR_AGE": (0, 100, False, True),  # blank: no car
    "REGION_RATING_CLIENT_W_CITY": (1, 3, True, True),
    "EXT_SOURCE_1": (0, 1, False, False),  # blank: no score from this source (thin file)
    "EXT_SOURCE_2": (0, 1, False, False),
    "EXT_SOURCE_3": (0, 1, False, False),
    "AMT_REQ_CREDIT_BUREAU_YEAR": (0, 100, False, True),
}
OPTIONAL_CATEGORICAL = {"OCCUPATION_TYPE"}  # blank becomes "Unknown", as in training


# ---- Reference statistics (from the training split) -------------------------------------
def build_reference_stats(X: pd.DataFrame) -> dict:
    """Guard ranges, deciles, PSI bins (src/monitoring.py) and category shares of the training inputs. No rows."""
    numeric = {}
    for col in NUMERIC_INPUTS:
        values = X[col].dropna()
        if col == "DAYS_EMPLOYED":  # the "not employed" code would distort the range
            values = values[values != DAYS_EMPLOYED_SENTINEL]
        low, high = values.quantile(list(GUARD_QUANTILES))
        numeric[col] = {
            "low": float(low),
            "high": float(high),
            "deciles": [float(v) for v in values.quantile(np.arange(1, 10) / 10)],
            "missing_share": round(float(X[col].isna().mean()), 6),
            "psi_bins": psi_bins(X[col], DAYS_EMPLOYED_SENTINEL if col == "DAYS_EMPLOYED" else None),
        }
    categorical = {
        col: X[col].fillna("Unknown").value_counts(normalize=True).round(6).to_dict()
        for col in CATEGORICAL_INPUTS
    }
    return {
        "rows": len(X),
        "guard_quantiles": list(GUARD_QUANTILES),
        "numeric": numeric,
        "categorical": categorical,
    }


def save_reference_stats(stats: dict, path: Path = REFERENCE_STATS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2))


def load_reference_stats(path: Path = REFERENCE_STATS_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Create it with: python -m src.validation")
    return json.loads(Path(path).read_text())


# ---- Errors --------------------------------------------------------------------------
def _is_blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype(str).str.strip().eq("")


def _allowed_text(values: list[str]) -> str:
    shown = ", ".join(f"'{v}'" for v in values[:6])
    return f"one of {shown}" + (f" ... ({len(values)} options)" if len(values) > 6 else "")


def _row_errors(df: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """One row per problem: (row, field, problem). Checks run per column, so large batches stay fast."""
    problems = []

    def add(mask: pd.Series, field: str, problem: str, show_value: bool = True) -> None:
        for row in df.index[mask.to_numpy()]:  # loops over failing rows only
            value = df.at[row, field]
            value = value.item() if isinstance(value, np.generic) else value  # 1.3, not np.float64(1.3)
            got = f" (got {value!r})" if show_value else ""
            problems.append((row, field, problem + got))

    numbers = {}
    for field, (low, high, required, whole) in NUMERIC_RULES.items():
        blank = _is_blank(df[field])
        values = pd.to_numeric(df[field], errors="coerce")
        numbers[field] = values
        if required:
            add(blank, field, "is required", show_value=False)
        add(~blank & values.isna(), field, "must be a number")
        allowed = values.between(low, np.inf if high is None else high)
        rule = f"must be at least {low}" if high is None else f"must be between {low} and {high}"
        if field == "DAYS_EMPLOYED":
            allowed |= values.eq(DAYS_EMPLOYED_SENTINEL)
            rule += f" (or {DAYS_EMPLOYED_SENTINEL} for not employed)"
        add(values.notna() & ~allowed, field, rule)
        if whole:
            add(allowed & (values % 1 != 0), field, "must be a whole number")

    # Only compared when the loan itself is valid, so a bad loan amount gives one error, not two
    add(
        (numbers["AMT_ANNUITY"] > numbers["AMT_CREDIT"]) & (numbers["AMT_CREDIT"] >= 1),
        "AMT_ANNUITY",
        "repayment can't be larger than the loan (AMT_CREDIT)",
    )

    for field in CATEGORICAL_INPUTS:
        blank = _is_blank(df[field])
        allowed = sorted(set(reference["categorical"][field]) - {"Unknown"})
        if field not in OPTIONAL_CATEGORICAL:
            add(blank, field, "is required", show_value=False)
        add(~blank & ~df[field].isin(allowed), field, f"must be {_allowed_text(allowed)}")

    return pd.DataFrame(problems, columns=["row", "field", "problem"]).sort_values("row", kind="stable")


def validate_applicant(applicant: dict, reference: dict) -> list[str]:
    """Problems with one applicant as "FIELD: problem" messages. An empty list means it can be scored.

    Every input must be present; optional ones (external scores, car age, occupation, goods price,
    bureau enquiries) may be None, which the model treats as "not available".
    """
    missing = [f for f in RAW_INPUTS if f not in applicant]
    if missing:
        return [f"{f}: field is missing (use None if unknown)" for f in missing]
    errors = _row_errors(pd.DataFrame([applicant]), reference)
    return [f"{field}: {problem}" for field, problem in zip(errors["field"], errors["problem"], strict=True)]


def read_csv_upload(data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV, refusing files over MAX_UPLOAD_MB or that aren't valid CSV."""
    size_mb = len(data) / 1024**2
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(
            f"The file is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB. Split it into smaller files."
        )
    try:
        return pd.read_csv(io.BytesIO(data))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ValueError(f"Couldn't read the file as a CSV: {e}") from e


def validate_batch(df: pd.DataFrame, reference: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(valid rows ready to score, errors with columns row / field / problem) for an uploaded batch.

    Problems with the whole file (no rows, too many rows, missing columns) raise a ValueError.
    Rows with errors are left out of the valid rows, so the rest of the batch can still be scored.
    """
    if df.empty:
        raise ValueError("The file has no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(
            f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}. Split it into smaller files."
        )
    missing = [c for c in RAW_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(missing)}. Optional columns can be left blank but must be there."
        )
    errors = _row_errors(df, reference)
    valid = df.drop(index=errors["row"].unique())
    valid = valid.assign(**{c: pd.to_numeric(valid[c], errors="coerce") for c in NUMERIC_INPUTS})
    valid[CATEGORICAL_INPUTS] = valid[CATEGORICAL_INPUTS].where(~valid[CATEGORICAL_INPUTS].apply(_is_blank))
    return valid, errors


# ---- Warnings: the out-of-range guard --------------------------------------------------
def _number(value: float) -> str:
    """50,000,000 and 0.062 rather than 5e+07 and 0.0620000."""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


def out_of_range(df: pd.DataFrame, reference: dict) -> pd.Series:
    """For each row, a list of warnings for inputs outside the training range (GUARD_QUANTILES)."""
    warnings = [[] for _ in range(len(df))]
    for field, stats in reference["numeric"].items():
        values = pd.to_numeric(df[field], errors="coerce")
        if field == "DAYS_EMPLOYED":
            values = values.where(values != DAYS_EMPLOYED_SENTINEL)
        outside = (values < stats["low"]) | (values > stats["high"])
        show = money if field in MONEY_FIELDS else _number
        for pos in np.flatnonzero(outside.to_numpy()):
            warnings[pos].append(
                f"{input_label(field)} ({show(values.iloc[pos])}) is outside the range of past approved "
                f"applicants ({show(stats['low'])} to {show(stats['high'])})"
            )
    return pd.Series(warnings, index=df.index, dtype=object)


if __name__ == "__main__":
    from src.data import features_target, load_clean, split

    train, _, _ = split(load_clean())
    X_train, _ = features_target(train)
    save_reference_stats(build_reference_stats(X_train))
    print(f"Saved {REFERENCE_STATS_PATH} from {len(X_train):,} training rows")
