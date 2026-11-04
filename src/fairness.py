"""Station 3 (Model Design). Does the model's decision treat groups of applicants differently?

Course reference: Code Bank W04-06 (demographic parity and equalised odds).

    table = fairness_table(test_df, test_pd, threshold)   # one row per group

Sex, age and family status are never model inputs, but the model can still treat groups
differently through inputs that are correlated with them (proxies: "not employed" is mostly
pensioners, so it carries age). So the audit checks the decisions, group by group, on the test
split, using the automatic rule (approve when PD < cut-off):

- Approval rate and the **disparate-impact ratio**: a group's approval rate divided by the
  most-approved group's. Below DI_THRESHOLD (0.8, the "four-fifths rule") is flagged.
- **Demographic parity gap**: the approval-rate difference to that group; flagged above
  DP_THRESHOLD (0.05, the CFPB safe harbour used in W04-06).
- **Equalised odds**: are good customers approved equally often in every group (equal
  opportunity), and defaulters too? The gap is the larger of the two differences to the reference
  group; flagged above EO_THRESHOLD.
- **Calibration within the group**: mean PD vs the actual default rate. If the PDs are right in
  every group, a lower approval rate reflects higher measured risk rather than a biased score.

Different approval rates are not automatically unfair: a group with a higher actual default rate
gets a lower approval rate under any accurate model. The equalised-odds and calibration columns
show whether applicants of similar risk are treated alike. Groups with fewer than MIN_GROUP_SIZE
applicants are reported but not judged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DI_THRESHOLD, DP_THRESHOLD, EO_THRESHOLD, MIN_GROUP_SIZE, TARGET
from src.data import age_years

AGE_BANDS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ["Under 30", "30-39", "40-49", "50-59", "60 and over"]
SEX_LABELS = {"F": "Female", "M": "Male"}


def audit_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Attribute -> each applicant's group label. Missing values become "Unknown"."""
    groups = {
        "Sex": df["CODE_GENDER"].map(SEX_LABELS),
        "Age band": pd.cut(age_years(df), AGE_BANDS, labels=AGE_LABELS, right=False).astype(object),
        "Family status": df["NAME_FAMILY_STATUS"],
        "Education": df["NAME_EDUCATION_TYPE"],
    }
    return {attribute: labels.fillna("Unknown") for attribute, labels in groups.items()}


def _rates(approved: np.ndarray, y: np.ndarray, pd_: np.ndarray) -> dict:
    good, bad = y == 0, y == 1
    return {
        "n": len(y),
        "approval_rate": float(approved.mean()),
        "default_rate": float(y.mean()),
        "mean_pd": float(pd_.mean()),
        "good_customers_approved": float(approved[good].mean()) if good.any() else np.nan,
        "defaulters_approved": float(approved[bad].mean()) if bad.any() else np.nan,
    }


def fairness_table(df: pd.DataFrame, pd_, threshold: float, min_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """One row per (attribute, group) with rates, gaps to the most-approved group and flags."""
    pd_ = np.asarray(pd_, dtype=float)
    y = df[TARGET].to_numpy()
    approved = pd_ < threshold
    rows = []
    for attribute, labels in audit_groups(df).items():
        labels = labels.to_numpy()
        group_rows = [
            {
                "attribute": attribute,
                "group": g,
                **_rates(approved[labels == g], y[labels == g], pd_[labels == g]),
            }
            for g in pd.unique(labels)
        ]
        judged = [r for r in group_rows if r["n"] >= min_size]
        reference = max(judged, key=lambda r: r["approval_rate"]) if judged else None
        for r in group_rows:
            r["judged"] = r["n"] >= min_size
            r["reference"] = r is reference
            if r["judged"] and reference is not None:
                r["di_ratio"] = r["approval_rate"] / reference["approval_rate"]
                r["dp_gap"] = reference["approval_rate"] - r["approval_rate"]
                r["eo_gap"] = max(
                    abs(r["good_customers_approved"] - reference["good_customers_approved"]),
                    abs(r["defaulters_approved"] - reference["defaulters_approved"]),
                )
                flags = []
                if r["di_ratio"] < DI_THRESHOLD:
                    flags.append(f"disparate impact < {DI_THRESHOLD}")
                if r["dp_gap"] > DP_THRESHOLD:
                    flags.append(f"parity gap > {DP_THRESHOLD}")
                if r["eo_gap"] > EO_THRESHOLD:
                    flags.append(f"equalised odds gap > {EO_THRESHOLD}")
                r["flags"] = "; ".join(flags)
            else:
                r.update(
                    di_ratio=np.nan,
                    dp_gap=np.nan,
                    eo_gap=np.nan,
                    flags="" if r["judged"] else f"too few to judge (< {min_size})",
                )
        rows.extend(sorted(group_rows, key=lambda r: -r["n"]))
    table = pd.DataFrame(rows)
    table["pd_gap"] = table["mean_pd"] - table["default_rate"]  # > 0: PDs too high for this group
    return table
