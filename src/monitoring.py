"""Step 11 (monitoring): has the population applying today drifted away from the one the model
learnt from?

Course reference: Code Bank W02-10, W05-07 (compute_psi with traffic-light thresholds).

    table = psi_table(new_batch, reference)       # one row per input: PSI and status

The **population stability index** compares how a batch's values spread over bins with how the
training data spread over the same bins:

    PSI = sum over bins of (actual share - expected share) x ln(actual share / expected share)

Below 0.1 the input is stable, 0.1-0.25 is worth watching, above 0.25 is an alert (the usual
credit-scoring thresholds; W02-10 uses 0.20 as the trigger for a model review under SR 11-7).
Numeric bins are the training deciles; missing values get their own bin (a rise in thin files,
i.e. missing external scores, is exactly the kind of shift to catch), and so does Home Credit's
"not employed" code. Categorical inputs use one bin per category, plus one for categories never
seen in training. The bins and training shares are stored in models/reference_stats.json, so
monitoring needs no training data.

Why it matters here: the model only saw applicants Home Credit had already approved (selection
bias). If the applicants change, for example a campaign brings in younger thin-file customers,
the PDs can be wrong even though the model itself hasn't changed.

Demo batches: `python -m src.monitoring` writes two **synthetic** batches (drawn from the training
summary statistics, never from Kaggle rows) to data/sample/: one like the training population
and one shifted towards weaker, thinner-file applicants.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_INPUTS,
    DATA_DIR,
    DAYS_EMPLOYED_SENTINEL,
    NUMERIC_INPUTS,
    RANDOM_STATE,
    RARE_CATEGORY_SHARE,
    RAW_INPUTS,
)

PSI_STABLE, PSI_ALERT = 0.10, 0.25
FLOOR = 1e-6  # stands in for an empty bin, so ln() stays finite (as in W02-10)
UNSEEN = "Not seen in training"
RARE = f"Rare categories (< {RARE_CATEGORY_SHARE:.0%} of training)"
# Inputs that only take whole numbers, so synthetic values are drawn as integers
WHOLE = {
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "REGION_RATING_CLIENT_W_CITY",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
}
SAMPLE_PATHS = {
    "normal": DATA_DIR / "sample" / "monitoring_normal.csv",
    "shifted": DATA_DIR / "sample" / "monitoring_shifted.csv",
}


# ---- Bins ----------------------------------------------------------------------------------
def numeric_bin_labels(edges: list[float], sentinel: float | None = None) -> list[str]:
    labels = [f"<= {edges[0]:,.4g}"] if edges else ["All values"]
    labels += [f"{lo:,.4g} to {hi:,.4g}" for lo, hi in pairwise(edges)]
    labels += [f"> {edges[-1]:,.4g}"] if edges else []
    return labels + (["Not employed"] if sentinel is not None else []) + ["Missing"]


def numeric_bin_shares(values: pd.Series, edges: list[float], sentinel: float | None = None) -> np.ndarray:
    """Share of rows in each bin: one bin per gap between edges, then the sentinel (if any), then missing."""
    values = pd.to_numeric(values, errors="coerce")
    missing = values.isna()
    special = values.eq(sentinel) if sentinel is not None else pd.Series(False, index=values.index)
    regular = values[~missing & ~special].to_numpy()
    counts = np.bincount(np.searchsorted(edges, regular, side="left"), minlength=len(edges) + 1)
    counts = np.append(counts, [special.sum()] if sentinel is not None else [])
    counts = np.append(counts, missing.sum())
    return counts / max(len(values), 1)


def psi_bins(values: pd.Series, sentinel: float | None = None) -> dict:
    """Training bins for one numeric input: decile edges (duplicates removed) and the share in each bin."""
    regular = pd.to_numeric(values, errors="coerce").dropna()
    if sentinel is not None:
        regular = regular[regular != sentinel]
    edges = sorted({float(v) for v in regular.quantile(np.arange(1, 10) / 10)})
    return {
        "edges": edges,
        "expected": [round(float(s), 6) for s in numeric_bin_shares(values, edges, sentinel)],
    }


def _sentinel(field: str) -> float | None:
    return DAYS_EMPLOYED_SENTINEL if field == "DAYS_EMPLOYED" else None


# ---- PSI -----------------------------------------------------------------------------------
def compute_psi(expected, actual) -> float:
    """PSI between two sets of bin shares (same bins, same order)."""
    expected = np.maximum(np.asarray(expected, dtype=float), FLOOR)
    actual = np.maximum(np.asarray(actual, dtype=float), FLOOR)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_status(psi: float) -> str:
    return "stable" if psi < PSI_STABLE else ("watch" if psi <= PSI_ALERT else "alert")


def categorical_shares(values: pd.Series, expected: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Bins, expected shares and actual shares for one categorical input.

    Categories rarer than RARE_CATEGORY_SHARE in training share one bin: the model's encoder groups
    them too, so shifts among them can't change a score, and with 58 employer types they would
    otherwise make PSI noisy for small batches. Categories never seen in training get their own bin.
    """
    common = {c: s for c, s in expected.items() if s >= RARE_CATEGORY_SHARE}
    labels = [*common, RARE, UNSEEN]
    values = values.fillna("Unknown")
    pooled = values.where(values.isin(common), values.where(~values.isin(expected), RARE))
    pooled = pooled.where(pooled.isin(labels), UNSEEN)
    expected_shares = np.array(
        [*common.values(), sum(s for c, s in expected.items() if c not in common), 0.0]
    )
    return (
        labels,
        expected_shares,
        pooled.value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy(),
    )


def psi_table(batch: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """PSI and status for every input, largest drift first. Raises ValueError for missing columns."""
    missing = [c for c in RAW_INPUTS if c not in batch.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    rows = []
    for field in NUMERIC_INPUTS:
        bins = reference["numeric"][field]["psi_bins"]
        actual = numeric_bin_shares(batch[field], bins["edges"], _sentinel(field))
        rows.append(
            {
                "input": field,
                "type": "numeric",
                "psi": compute_psi(bins["expected"], actual),
                "missing_train": bins["expected"][-1],
                "missing_batch": actual[-1],
            }
        )
    for field in CATEGORICAL_INPUTS:
        _, expected, actual = categorical_shares(batch[field], reference["categorical"][field])
        rows.append(
            {
                "input": field,
                "type": "categorical",
                "psi": compute_psi(expected, actual),
                "missing_train": np.nan,
                "missing_batch": np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["status"] = table["psi"].map(psi_status)
    return table.sort_values("psi", ascending=False, ignore_index=True)


def bin_detail(batch: pd.DataFrame, reference: dict, field: str) -> pd.DataFrame:
    """Training vs batch share in each bin of one input, to see where a shift comes from."""
    if field in reference["numeric"]:
        bins = reference["numeric"][field]["psi_bins"]
        labels = numeric_bin_labels(bins["edges"], _sentinel(field))
        expected, actual = (
            np.array(bins["expected"]),
            numeric_bin_shares(batch[field], bins["edges"], _sentinel(field)),
        )
    else:
        labels, expected, actual = categorical_shares(batch[field], reference["categorical"][field])
    return pd.DataFrame({"bin": labels, "training": expected, "batch": actual})


# ---- Synthetic demo batches -----------------------------------------------------------------
def synthetic_batch(
    reference: dict, n: int = 1000, shifted: bool = False, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Invented applicants drawn from the training summary statistics (no real rows).

    Each numeric input picks one of its PSI bins with the training share (including the missing
    and "not employed" bins), then a value inside it; the two open-ended outer bins are kept within
    one bin-width of the outer deciles so the tails stay plausible. Each category is drawn with its
    training share. Inputs are drawn independently, so individual rows are not realistic people,
    but each input's distribution over the PSI bins matches training, so a "normal" batch differs
    from training only by sampling noise.

    shifted=True imitates a campaign that brings in younger, thinner-file applicants: lower and
    more often missing external scores, shorter employment and fewer pensioners, more recent ID
    and address changes, larger loans, and more applicants renting or living with parents.
    """
    rng = np.random.default_rng(seed + (1 if shifted else 0))
    batch = {field: _draw_numeric(field, reference["numeric"][field], n, rng) for field in NUMERIC_INPUTS}
    for field in CATEGORICAL_INPUTS:
        shares = reference["categorical"][field]
        p = np.array(list(shares.values()))
        batch[field] = rng.choice(np.array(list(shares), dtype=object), n, p=p / p.sum())
    df = pd.DataFrame(batch)
    df[CATEGORICAL_INPUTS] = df[CATEGORICAL_INPUTS].replace("Unknown", np.nan)  # "Unknown" means left blank
    df["FLAG_OWN_CAR"] = np.where(df["OWN_CAR_AGE"].notna(), "Y", "N")  # a car age means a car

    if shifted:
        for field in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
            df[field] = df[field] * 0.75
            df.loc[rng.random(n) < 0.25, field] = np.nan  # more thin files
        employed = df["DAYS_EMPLOYED"].notna() & ~df["DAYS_EMPLOYED"].eq(DAYS_EMPLOYED_SENTINEL)
        df.loc[employed, "DAYS_EMPLOYED"] *= 0.4  # shorter time in the current job
        fewer_pensioners = df["DAYS_EMPLOYED"].eq(DAYS_EMPLOYED_SENTINEL) & (rng.random(n) < 0.7)
        df.loc[fewer_pensioners, "DAYS_EMPLOYED"] = -rng.integers(30, 1500, int(fewer_pensioners.sum()))
        df["DAYS_ID_PUBLISH"] *= 0.6
        df["DAYS_REGISTRATION"] *= 0.6
        df["AMT_CREDIT"] *= 1.2
        df["AMT_GOODS_PRICE"] *= 1.2
        renters = rng.random(n) < 0.25
        df.loc[renters, "NAME_HOUSING_TYPE"] = rng.choice(
            np.array(["Rented apartment", "With parents"], dtype=object), int(renters.sum())
        )

    # Keep every row valid for the app: whole numbers where needed, repayment no larger than the loan
    for field in (
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "REGION_RATING_CLIENT_W_CITY",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ):
        df[field] = df[field].round()
    df["AMT_ANNUITY"] = np.minimum(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    amounts = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    df[amounts] = df[amounts].round()
    df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = df[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].round(4)
    return df[RAW_INPUTS]


def _draw_numeric(field: str, stats: dict, n: int, rng) -> np.ndarray:
    """n values of one numeric input, spread over its PSI bins with the training shares."""
    edges, shares = stats["psi_bins"]["edges"], np.array(stats["psi_bins"]["expected"])
    lower, upper = np.array([stats["low"], *edges]), np.array([*edges, stats["high"]])
    if len(edges) >= 2:
        lower[0] = max(stats["low"], edges[0] - (edges[1] - edges[0]))
        upper[-1] = min(stats["high"], edges[-1] + (edges[-1] - edges[-2]))
    k = len(edges) + 1  # regular bins; then the sentinel bin (DAYS_EMPLOYED only), then missing
    chosen = rng.choice(len(shares), n, p=shares / shares.sum())
    values = np.full(n, np.nan)
    regular = chosen < k
    lo, hi = lower[chosen[regular]], upper[chosen[regular]]
    if field in WHOLE:
        # whole numbers in (lo, hi]; the first bin also includes the training minimum itself
        start = np.where(chosen[regular] == 0, np.ceil(lo), np.floor(lo) + 1)
        count = np.maximum(np.floor(hi) - start + 1, 1)
        values[regular] = start + np.floor(rng.random(regular.sum()) * count)
    else:
        values[regular] = hi - rng.random(regular.sum()) * (hi - lo)
    if _sentinel(field) is not None:
        values[chosen == k] = DAYS_EMPLOYED_SENTINEL
    return values


if __name__ == "__main__":
    import json

    from src.config import REFERENCE_STATS_PATH

    reference = json.loads(REFERENCE_STATS_PATH.read_text())
    for name, path in SAMPLE_PATHS.items():
        synthetic_batch(reference, shifted=name == "shifted").to_csv(path, index=False)
        print(f"Wrote {path} (synthetic, 1,000 applicants)")
