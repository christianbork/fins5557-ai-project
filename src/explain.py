"""Station 4 (Implementation). Which inputs pushed an applicant's PD up or down, and by how much.

Course reference: Code Bank W05-06 (TreeExplainer + adverse-action reasons) and W02-07 (waterfall).

    explainer = build_explainer(model)            # once; predict.load_artifacts does this
    contrib = explain(applicants_df, explainer)   # one row per applicant, one column per input
    reasons = top_reasons(contrib.iloc[0])        # e.g. ["External credit scores", ...]

SHAP splits the model's score for one applicant into a contribution per feature, so that
base value + contributions = the applicant's score. Contributions are in log-odds of the
uncalibrated model. Calibration (isotonic) only rescales the score and never reverses an
order, so a feature that pushes the score up also pushes the calibrated PD up.

Two steps make the output readable for an analyst and a declined customer:
1. Model columns are added back up to the input they came from: the one-hot columns of
   OCCUPATION_TYPE become one OCCUPATION_TYPE value, and EXT_SOURCE_1 plus its
   "was missing" column become one EXT_SOURCE_1 value.
2. Inputs are turned into phrases (REASON_PHRASES); related inputs share a phrase, so the
   three external scores and their average give one reason, not four.

Phrases name the factor, not a direction ("Time in current job", not "Short time in job"):
several effects are U-shaped or reversed (e.g. pensioners, coded "not employed", default less),
so a fixed direction would sometimes be false, and a decline reason must never be false.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.config import FEATURE_SET
from src.features import feature_columns

# Every model input (raw and engineered) -> the plain-English factor shown to analysts and in
# decline letters. Protected attributes are never model inputs, so they can never be a reason.
REASON_PHRASES = {
    # Amounts and affordability
    "AMT_INCOME_TOTAL": "Level of income",
    "income_per_person": "Level of income",
    "AMT_CREDIT": "Size of the loan requested",
    "AMT_GOODS_PRICE": "Size of the loan requested",
    "credit_income_ratio": "Loan size relative to income",
    "AMT_ANNUITY": "Repayments relative to income",
    "annuity_income_ratio": "Repayments relative to income",
    "credit_term": "Length of the loan",
    "goods_credit_ratio": "Amount borrowed beyond the price of the goods",
    "NAME_CONTRACT_TYPE": "Type of loan (cash or revolving)",
    # Credit history
    "EXT_SOURCE_1": "External credit scores",
    "EXT_SOURCE_2": "External credit scores",
    "EXT_SOURCE_3": "External credit scores",
    "ext_source_mean": "External credit scores",
    "ext_source_missing": "Length of credit history (missing external scores)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Recent credit enquiries",
    # Employment and income source
    "DAYS_EMPLOYED": "Time in current job",
    "years_employed": "Time in current job",
    "not_employed": "Employment status",
    "NAME_INCOME_TYPE": "Source of income",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Type of employer",
    "NAME_EDUCATION_TYPE": "Education level",
    # Household and stability
    "CNT_CHILDREN": "Household size",
    "CNT_FAM_MEMBERS": "Household size",
    "NAME_HOUSING_TYPE": "Housing situation",
    "FLAG_OWN_REALTY": "Property ownership",
    "FLAG_OWN_CAR": "Car ownership",
    "OWN_CAR_AGE": "Car ownership",
    "DAYS_REGISTRATION": "Time since registration details changed",
    "DAYS_ID_PUBLISH": "Time since ID document was renewed",
    "REGION_RATING_CLIENT_W_CITY": "Region of residence",
}


def unwrap(model) -> Pipeline:
    """The fitted feature + model pipeline inside the calibration wrapper (or the model itself)."""
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator
    return getattr(model, "estimator", model)  # FrozenEstimator keeps the pipeline in .estimator


def _source_column(name: str, numeric: list[str], categorical: list[str]) -> str:
    """Model column (after preprocessing) -> the input it came from."""
    if name.startswith("missingindicator_"):
        return name.removeprefix("missingindicator_")
    if name in numeric:
        return name
    matches = [c for c in categorical if name.startswith(f"{c}_")]
    if not matches:
        raise ValueError(f"Can't trace model column '{name}' back to an input")
    return max(matches, key=len)


def build_explainer(model, feature_set: str = FEATURE_SET) -> dict | None:
    """SHAP TreeExplainer for the model, built once (slow-ish) and cached with the model.

    Returns None for a model that isn't tree-based: the chosen model is XGBoost, and if a retrain
    ever picks logistic regression, its coefficients would need a LinearExplainer instead.
    """
    pipe = unwrap(model)
    estimator = pipe.steps[-1][1]
    if not hasattr(estimator, "feature_importances_"):
        return None
    numeric, categorical = feature_columns(feature_set)
    columns = pipe.named_steps["preprocess"].get_feature_names_out()
    return {
        "tree": shap.TreeExplainer(estimator),
        "transform": pipe[:-1],  # feature engineering + preprocessing, already fitted
        "sources": [_source_column(c, numeric, categorical) for c in columns],
    }


def explain(df: pd.DataFrame, explainer: dict) -> pd.DataFrame:
    """SHAP contribution of every input for every row of df (log-odds; positive = more risk)."""
    values = explainer["tree"].shap_values(explainer["transform"].transform(df))
    values = np.asarray(values)
    if values.ndim == 3:  # random forest: one set of values per class, keep "default"
        values = values[:, :, 1]
    per_column = pd.DataFrame(values, index=df.index, columns=explainer["sources"])
    return per_column.T.groupby(level=0, sort=False).sum().T


def contributions_by_reason(contributions: pd.Series | dict) -> pd.Series:
    """Per-input contributions added up by reason phrase, largest effect first (for the app's chart)."""
    contributions = pd.Series(contributions, dtype=float)
    by_phrase = contributions.groupby([REASON_PHRASES[f] for f in contributions.index]).sum()
    return by_phrase.reindex(by_phrase.abs().sort_values(ascending=False).index)


def top_reasons(contributions: pd.Series, k: int = 3) -> list[str]:
    """The k factors that raised this applicant's risk the most, as plain-English phrases.

    Only factors with a positive total contribution are listed, so a reason given for a decline
    always increased the PD. Fewer than k are returned if fewer factors raised the risk.
    """
    by_phrase = contributions.groupby([REASON_PHRASES[f] for f in contributions.index]).sum()
    return list(by_phrase[by_phrase > 0].nlargest(k).index)
