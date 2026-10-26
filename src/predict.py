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

from src.config import METADATA_PATH, MODEL_PATH
from src.threshold import decide


def load_artifacts(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH) -> dict:
    """Load the model and its metadata once. Slow-ish (a 300-tree forest), hence cached by the app."""
    missing = [str(p) for p in (model_path, metadata_path) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Model files not found: {missing}. Train the model first: python -m src.train")
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get("threshold") is None:
        raise ValueError("Model metadata has no threshold. Retrain with: python -m src.train")
    return {"model": joblib.load(model_path), "metadata": metadata}


def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing applicant fields: {', '.join(missing)}")


def _score(df: pd.DataFrame, artifacts: dict) -> tuple[np.ndarray, np.ndarray]:
    """PDs and decisions for every row of df, in one vectorised call."""
    meta = artifacts["metadata"]
    required = meta["input_columns"]
    _check_columns(df, required)
    pds = artifacts["model"].predict_proba(df[required])[:, 1]
    return pds, decide(pds, meta["threshold"], meta["refer_band"])


def predict(applicant: dict, artifacts: dict) -> dict:
    """Score one applicant.

    Returns {"pd": float, "decision": "approve" | "refer" | "decline", "reasons": [...], "shap": {...}}.
    "reasons" and "shap" are filled in by the explainability step (src/explain.py, Step 6).
    Input range checks (age, valid codes...) are added in src/validation.py (Step 7); here a
    missing field raises a ValueError naming it.
    """
    if not isinstance(applicant, dict):
        raise TypeError("applicant must be a dict of field name -> value")
    pds, decisions = _score(pd.DataFrame([applicant]), artifacts)
    return {"pd": float(pds[0]), "decision": str(decisions[0]), "reasons": [], "shap": {}}


def predict_batch(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Score many applicants at once. Returns a copy of df with "pd" and "decision" columns added."""
    pds, decisions = _score(df, artifacts)
    return df.assign(pd=pds, decision=decisions)
