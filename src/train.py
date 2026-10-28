"""Station 3 (models): train, tune and compare three credit-default models, then calibrate and save
the chosen one.

Run from the project root (about 10 minutes on the 184k training rows):

    python -m src.train

1. Trains logistic regression (the traditional scorecard baseline), a random forest and
   XGBoost, tuning each with stratified cross-validation on the training split only.
2. Compares them on the validation split and picks one with the rule in config (MIN_AUC_GAIN).
3. Calibrates the chosen model and sets the cost-based threshold (src/threshold.py).
4. Reports the test split once, at that threshold.

Saves:
    models/model.joblib                 calibrated pipeline: raw applicant columns in, PD out
    models/metadata.json                what was trained, when, on what, threshold and results
    reports/model_comparison.csv        validation metrics for all three models
    reports/profit_curve.csv            profit and approval rate at each candidate cut-off
    reports/test_metrics.json           final test-split results
    reports/validation_predictions.csv  validation scores per model, for plots (gitignored)
"""
from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    EAD_COLUMN,
    FEATURE_SET,
    LGD,
    MARGIN,
    METADATA_PATH,
    MIN_AUC_GAIN,
    MODEL_PATH,
    RANDOM_STATE,
    REFER_BAND,
    REPORTS_DIR,
)
from src.data import features_target, load_clean, split
from src.features import add_features, build_preprocessor, feature_columns
from src.metrics import evaluate
from src.threshold import break_even_pd, calibrate_and_set_threshold
from src.validation import build_reference_stats, save_reference_stats

BASELINE = "logistic_regression"

# Small grids on purpose: the whole search takes a few minutes, not hours.
PARAM_GRIDS = {
    # C = inverse regularisation strength: smaller C shrinks the coefficients more.
    "logistic_regression": {"model__C": [0.01, 0.1, 1.0]},
    "random_forest": {"model__max_depth": [8, 12], "model__min_samples_leaf": [50, 200]},
    "xgboost": {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]},
}


def make_models(y_train: pd.Series) -> dict:
    """The three candidate models, each set up for the ~8% default rate.

    Class weighting makes a missed defaulter cost more during training than a false alarm,
    so the models don't just predict "no default" for everyone. It also pushes the predicted
    probabilities upwards, which is why the chosen model is recalibrated before its scores
    are used as PDs.
    """
    negatives, positives = np.bincount(y_train)
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_samples=0.3,  # each tree sees a 30% bootstrap sample: much faster on 184k rows
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            scale_pos_weight=negatives / positives,  # XGBoost's version of class weighting
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def build_pipeline(model, feature_set: str = FEATURE_SET) -> Pipeline:
    """Raw applicant columns in, probability of default out.

    Putting feature engineering and preprocessing inside the pipeline means:
    - they are fitted on training folds only during cross-validation (no leakage), and
    - the saved model takes the same raw columns the app collects, so the app can't apply
      the steps differently from training.
    """
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", build_preprocessor(feature_set)),
        ("model", model),
    ])


def choose_model(comparison: pd.DataFrame, min_gain: float = MIN_AUC_GAIN) -> tuple[str, str]:
    """Pick the final model from the validation comparison (rows = models). Returns (name, reason)."""
    best = comparison["auc"].idxmax()
    base_auc = comparison.loc[BASELINE, "auc"]
    gain = comparison.loc[best, "auc"] - base_auc
    if best == BASELINE:
        return BASELINE, f"Logistic regression has the highest validation AUC ({base_auc:.3f})."
    if gain < min_gain:
        return BASELINE, (
            f"{best} has the highest validation AUC, but beats logistic regression by only "
            f"{gain:.3f} (< {min_gain}), so the simpler, more explainable baseline is kept."
        )
    return best, (
        f"{best} has the highest validation AUC ({comparison.loc[best, 'auc']:.3f}), "
        f"{gain:.3f} above logistic regression ({base_auc:.3f}), which clears the "
        f"{min_gain} margin needed to justify a more complex model."
    )


def train_all(train: pd.DataFrame, val: pd.DataFrame, param_grids: dict = PARAM_GRIDS,
              cv_folds: int = CV_FOLDS, verbose: bool = True) -> dict:
    """Tune each model with CV on train, score it on validation, choose one.

    Returns a dict with the fitted pipelines, the comparison table, validation predictions,
    best parameters, and the chosen model with its reason.
    """
    X_train, y_train = features_target(train)
    X_val, y_val = features_target(val)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    fitted, rows, preds, params = {}, {}, {}, {}
    for name, model in make_models(y_train).items():
        start = time.perf_counter()
        search = GridSearchCV(build_pipeline(model), param_grids[name], scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)  # refits the best settings on the full training split
        fitted[name] = search.best_estimator_
        params[name] = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        preds[name] = search.predict_proba(X_val)[:, 1]
        rows[name] = {"cv_auc": search.best_score_, **evaluate(y_val, preds[name])}
        if verbose:
            print(f"{name:20s} CV AUC {search.best_score_:.3f}  val AUC {rows[name]['auc']:.3f}  "
                  f"{params[name]}  ({time.perf_counter() - start:.0f}s)")

    comparison = pd.DataFrame(rows).T
    comparison.index.name = "model"
    chosen, reason = choose_model(comparison)
    return {
        "pipelines": fitted,
        "comparison": comparison,
        "val_predictions": pd.DataFrame({"default": y_val.to_numpy(), **preds}, index=y_val.index),
        "best_params": params,
        "chosen": chosen,
        "reason": reason,
    }


def build_metadata(result: dict, decision: dict, input_columns: list[str], n_rows: dict) -> dict:
    """Everything needed to know later what the saved model is (model risk documentation)."""
    chosen = result["chosen"]
    numeric, categorical = feature_columns()
    return {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chosen_model": chosen,
        "reason": result["reason"],
        "best_params": result["best_params"][chosen],
        "validation_metrics": {k: round(float(v), 4) for k, v in result["comparison"].loc[chosen].items()},
        "all_models": result["comparison"].round(4).to_dict(orient="index"),
        "calibration": {"method": decision["method"],
                        "validation_brier": {k: round(float(v), 4) for k, v in decision["brier"].items()}},
        "threshold": decision["threshold"],
        "refer_band": REFER_BAND,
        "assumptions": {"lgd": LGD, "margin": MARGIN, "ead": EAD_COLUMN,
                        "break_even_pd": round(break_even_pd(), 4)},
        "test_metrics": decision["test"],
        "input_columns": input_columns,
        "rows": n_rows,
        "feature_set": FEATURE_SET,
        "features": {"numeric": numeric, "categorical": categorical},
        "random_state": RANDOM_STATE,
        "versions": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]},
        },
    }


def save_artifacts(model, metadata: dict, model_path: Path = MODEL_PATH,
                   metadata_path: Path = METADATA_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    start = time.perf_counter()
    train, val, test = split(load_clean())
    print(f"Training on {len(train):,} rows, validating on {len(val):,} ({CV_FOLDS}-fold CV for tuning)")
    result = train_all(train, val)
    chosen = result["chosen"]
    print(f"\nChosen: {chosen}. {result['reason']}")

    X_val, y_val = features_target(val)
    X_test, y_test = features_target(test)
    decision = calibrate_and_set_threshold(result["pipelines"][chosen], X_val, y_val, X_test, y_test,
                                           raw_val_scores=result["val_predictions"][chosen])

    REPORTS_DIR.mkdir(exist_ok=True)
    result["comparison"].round(4).to_csv(REPORTS_DIR / "model_comparison.csv")
    decision["curve"].round(4).to_csv(REPORTS_DIR / "profit_curve.csv", index=False)
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(decision["test"], indent=2))
    result["val_predictions"].assign(calibrated_pd=decision["val_pd"]).to_csv(
        REPORTS_DIR / "validation_predictions.csv", index=False)

    n_rows = {"train": len(train), "validation": len(val), "test": len(test)}
    save_artifacts(decision["model"], build_metadata(result, decision, list(X_val.columns), n_rows))
    save_reference_stats(build_reference_stats(features_target(train)[0]))  # input checks + monitoring

    t = decision["test"]
    print(f"Calibration: {decision['method']} (validation Brier "
          + ", ".join(f"{k} {v:.4f}" for k, v in decision["brier"].items()) + ")")
    print(f"Threshold: decline at PD >= {decision['threshold']:.3f} (break-even PD {break_even_pd():.3f}), "
          f"refer band +/- {REFER_BAND}")
    print(f"Test: AUC {t['auc']:.3f}, Brier {t['brier']:.4f}, approval rate {t['approval_rate']:.1%}, "
          f"default rate among approved {t['default_rate_approved']:.1%} (vs {t['default_rate_all']:.1%} overall)")
    print(f"Saved {MODEL_PATH.relative_to(MODEL_PATH.parents[1])} and metadata "
          f"in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
