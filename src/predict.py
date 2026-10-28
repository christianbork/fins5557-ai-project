"""Station 5 (inference): score applicants with the saved, calibrated model.

This is the only module the app needs for scoring:

    from src.predict import load_artifacts, predict, predict_batch

    artifacts = load_artifacts()          # the app wraps this in st.cache_resource
    result = predict(applicant_dict, artifacts)
    scored = predict_batch(applicants_df, artifacts)

The cut-off and refer band come from models/metadata.json, so the app always uses the
threshold that was chosen when the model was trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH, REFERENCE_STATS_PATH
from src.explain import build_explainer, explain, top_reasons
from src.threshold import decide
from src.validation import load_reference_stats, out_of_range


def load_artifacts(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH,
                   reference_path: Path = REFERENCE_STATS_PATH) -> dict:
    """Load the model, its metadata, the training reference ranges and the SHAP explainer once
    (cached by the app)."""
    missing = [str(p) for p in (model_path, metadata_path, reference_path) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Model files not found: {missing}. Train the model first: python -m src.train")
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get("threshold") is None:
        raise ValueError("Model metadata has no threshold. Retrain with: python -m src.train")
    model = joblib.load(model_path)
    return {"model": model, "metadata": metadata, "reference": load_reference_stats(reference_path),
            "explainer": build_explainer(model)}


def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing applicant fields: {', '.join(missing)}")


def _score(df: pd.DataFrame, artifacts: dict) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """PDs, decisions and out-of-range warnings for every row of df, in one vectorised call.

    An applicant with an input outside the training range is never approved automatically:
    the model hasn't seen anyone like them (selection bias), so an approve becomes a refer.
    """
    meta = artifacts["metadata"]
    required = meta["input_columns"]
    _check_columns(df, required)
    pds = artifacts["model"].predict_proba(df[required])[:, 1]
    decisions = decide(pds, meta["threshold"], meta["refer_band"])
    warnings = out_of_range(df, artifacts["reference"])
    decisions = np.where((decisions == "approve") & (warnings.str.len() > 0).to_numpy(), "refer", decisions)
    return pds, decisions, warnings


def predict(applicant: dict, artifacts: dict) -> dict:
    """Score one applicant.

    Returns {"pd": float, "decision": "approve" | "refer" | "decline", "reasons": [...], "shap": {...},
    "warnings": [...]}. "warnings" lists inputs outside the training range (an approve becomes refer).
    "reasons" are the top 3 factors that raised the risk (src/explain.py); "shap" is each input's
    contribution in log-odds, for the app's waterfall chart. Both stay empty if the model has no
    explainer (a non-tree model).
    Run src.validation.validate_applicant first for clear input errors; here a missing field
    raises a ValueError naming it.
    """
    if not isinstance(applicant, dict):
        raise TypeError("applicant must be a dict of field name -> value")
    df = pd.DataFrame([applicant])
    pds, decisions, warnings = _score(df, artifacts)
    reasons, contributions = [], {}
    if artifacts.get("explainer") is not None:
        row = explain(df[artifacts["metadata"]["input_columns"]], artifacts["explainer"]).iloc[0]
        reasons, contributions = top_reasons(row), row.round(4).to_dict()
    return {"pd": float(pds[0]), "decision": str(decisions[0]), "reasons": reasons, "shap": contributions,
            "warnings": warnings.iloc[0]}


def predict_batch(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Score many applicants at once. Returns a copy of df with "pd", "decision" and "warnings"
    (out-of-range inputs, joined with "; ") columns added."""
    pds, decisions, warnings = _score(df, artifacts)
    return df.assign(pd=pds, decision=decisions, warnings=warnings.str.join("; "))
