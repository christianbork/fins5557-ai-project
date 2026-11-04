"""Station 4 (Implementation). Display text in one place: decisions, model names, input names, table
headers and statuses.

The code keeps short lower-case codes ("approve", "xgboost", "approval_rate"); anything a user
sees goes through this module, so the same thing is always called the same:

    decision_label("refer")         -> "Refer"
    model_label("random_forest")    -> "Random forest"
    input_label("DAYS_EMPLOYED")    -> "Days in current job"
    headers(df)                     -> df with every column renamed for display
"""

from __future__ import annotations

import pandas as pd

from src.applicant import FIELDS, label
from src.config import MIN_AUC_GAIN
from src.formatting import score

DECISIONS = {"approve": "Approve", "refer": "Refer", "decline": "Decline"}
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
MONEY_FIELDS = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"}  # shown with $
STATUSES = {"stable": "Stable", "watch": "Watch", "alert": "Alert"}
# The model's day fields as Home Credit stores them (negative = before the application). The form
# asks for years (src/applicant.py); monitoring compares the stored values, so it names them in days.
DAY_FIELDS = {
    "DAYS_EMPLOYED": "Days in current job",
    "DAYS_REGISTRATION": "Days since registration details changed",
    "DAYS_ID_PUBLISH": "Days since ID document was renewed",
}

# Column code -> header, for every table the app shows. Model inputs use input_label() instead.
COLUMNS = {
    # Batch scoring
    "example": "Example applicant",
    "pd": "PD",
    "decision": "Decision",
    "reasons": "Main reasons (refer / decline)",
    "warnings": "Warnings",
    "policy": "Policy rule broken",
    "expected_loss": "Expected loss",
    "csv_line": "Line in file",
    "field": "Column",
    "field_label": "Field",
    "problem": "Problem",
    # Model comparison and cut-offs
    "model": "Model",
    "cv_auc": "CV AUC",
    "auc": "Validation AUC",
    "pr_auc": "PR-AUC",
    "ks": "KS",
    "tpr": "Defaulters caught",
    "tnr": "Good customers approved",
    "precision": "Precision",
    "rule": "Rule",
    "cutoff": "Cut-off",
    "approval_rate": "Approval rate",
    "balanced_accuracy": "Balanced accuracy",
    "realised_profit": "Profit",
    # Fairness
    "attribute": "Attribute",
    "group": "Group",
    "n": "Applicants",
    "di_ratio": "Disparate-impact ratio",
    "dp_gap": "Parity gap",
    "good_customers_approved": "Good customers approved",
    "defaulters_approved": "Defaulters approved",
    "eo_gap": "Equalised-odds gap",
    "mean_pd": "Mean PD",
    "default_rate": "Default rate",
    "flags": "Flags",
    # Monitoring
    "input": "Input",
    "psi": "PSI",
    "status": "Status",
    "missing_train": "Missing (training)",
    "missing_batch": "Missing (batch)",
}
RULES = {"profit": "Profit (app)", "balanced_accuracy": "Balanced accuracy"}


def decision_label(code: str) -> str:
    return DECISIONS[code]


def model_label(code: str) -> str:
    return MODEL_NAMES.get(code, code)


def status_label(code: str) -> str:
    return STATUSES[code]


def input_label(field: str) -> str:
    """Display name of a model input: the form label without its unit hint ("($)", "(0 to 1)": the
    values show the unit), or the days wording for the stored day fields."""
    if field in DAY_FIELDS:
        return DAY_FIELDS[field]
    if field not in FIELDS:
        return field
    name = label(field)
    for unit in (" ($)", " ($ per instalment)", " (0 to 1)"):
        name = name.removesuffix(unit)
    return name


def column_label(code: str) -> str:
    """Header for a table column: a known column, else a model input, else the code unchanged."""
    if code in COLUMNS:
        return COLUMNS[code]
    return input_label(code)


def headers(df: pd.DataFrame) -> pd.DataFrame:
    """df with its columns (and index name) renamed for display."""
    out = df.rename(columns=column_label)
    if out.index.name is not None:
        out.index.name = column_label(out.index.name)
    return out


def flag_text(flags: str) -> str:
    """'parity gap > 0.05; equalised odds gap > 0.05' -> 'Parity gap > 0.05; Equalised odds gap > 0.05'."""
    return "; ".join(f[:1].upper() + f[1:] for f in flags.split("; ")) if flags else ""


def model_choice_reason(
    comparison: pd.DataFrame, baseline: str = "logistic_regression", min_gain: float = MIN_AUC_GAIN
) -> str:
    """The sentence explaining the model choice (src.train.choose_model), from the validation AUCs
    in comparison (rows = model codes), using display names."""
    best = comparison["auc"].idxmax()
    base_auc = comparison.loc[baseline, "auc"]
    gain = comparison.loc[best, "auc"] - base_auc
    if best == baseline:
        return f"{model_label(baseline)} has the highest validation AUC ({score(base_auc)})."
    if gain < min_gain:
        return (
            f"{model_label(best)} has the highest validation AUC, but beats {model_label(baseline).lower()} by "
            f"only {score(gain)} (< {min_gain}), so the simpler, more explainable baseline is kept."
        )
    return (
        f"{model_label(best)} has the highest validation AUC ({score(comparison.loc[best, 'auc'])}), "
        f"{score(gain)} above {model_label(baseline).lower()} ({score(base_auc)}), which clears the "
        f"{min_gain} margin needed to justify a more complex model."
    )
