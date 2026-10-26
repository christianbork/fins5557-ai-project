"""Station 3 (models): train, tune and compare three credit-default models, then save the chosen one.

Run from the project root:

    python -m src.train

It trains logistic regression (the traditional scorecard baseline), a random forest and
XGBoost, tunes each with stratified cross-validation on the training split only, compares
them on the validation split, picks one with the rule in config (MIN_AUC_GAIN) and saves:

    models/model.joblib               the chosen pipeline (features + preprocessing + model)
    models/metadata.json              what was trained, when, on what, and how well it did
    reports/model_comparison.csv      validation metrics for all three models
    reports/validation_predictions.csv  validation PDs per model, for ROC plots (gitignored)

The test split is not touched here; it is used once, for final reporting, in Step 4.
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
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    FEATURE_SET,
    METADATA_PATH,
    MIN_AUC_GAIN,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
)
from src.data import features_target, load_clean, split
from src.features import add_features, build_preprocessor, feature_columns

BASELINE = "logistic_regression"

# Small grids on purpose: the whole search takes a few minutes, not hours.
PARAM_GRIDS = {
    # C = inverse regularisation strength: smaller C shrinks the coefficients more.
    "logistic_regression": {"model__C": [0.01, 0.1, 1.0]},
    "random_forest": {"model__max_depth": [8, 12], "model__min_samples_leaf": [5, 20]},
    "xgboost": {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]},
}


def make_models(y_train: pd.Series) -> dict:
    """The three candidate models, each set up for the ~22% default rate.

    Class weighting makes a missed defaulter cost more during training than a false alarm,
    so the models don't just predict "no default" for everyone. It also pushes the predicted
    probabilities upwards, which is why Step 4 recalibrates them before they are used as PDs.
    """
    negatives, positives = np.bincount(y_train)
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
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


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the largest gap between the score distributions of defaulters and
    non-defaulters. Same measure as Code Bank W05-05's ks_statistic(), computed from the ROC
    curve (KS = max(TPR - FPR)) instead of a Python loop.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score, threshold: float = 0.5) -> dict:
    """Validation metrics used in credit scoring.

    AUC / Gini / KS measure ranking (does the model put defaulters above non-defaulters?).
    Brier score measures probability accuracy (lower is better). Balanced accuracy uses a
    0.5 cut-off for now; the real, cost-based threshold is chosen in Step 4.
    """
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, (np.asarray(y_score) >= threshold).astype(int)),
    }


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


def build_metadata(result: dict, n_train: int, n_val: int) -> dict:
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
        "rows": {"train": n_train, "validation": n_val},
        "feature_set": FEATURE_SET,
        "features": {"numeric": numeric, "categorical": categorical},
        "random_state": RANDOM_STATE,
        "threshold": None,  # set in Step 4 (expected-loss cut-off)
        "versions": {
            "python": platform.python_version(),
            **{pkg: version(pkg) for pkg in ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]},
        },
    }


def save_artifacts(pipeline, metadata: dict, model_path: Path = MODEL_PATH,
                   metadata_path: Path = METADATA_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    start = time.perf_counter()
    train, val, _test = split(load_clean())
    print(f"Training on {len(train):,} rows, validating on {len(val):,} ({CV_FOLDS}-fold CV for tuning)")
    result = train_all(train, val)

    REPORTS_DIR.mkdir(exist_ok=True)
    result["comparison"].round(4).to_csv(REPORTS_DIR / "model_comparison.csv")
    result["val_predictions"].to_csv(REPORTS_DIR / "validation_predictions.csv", index=False)

    chosen = result["chosen"]
    save_artifacts(result["pipelines"][chosen], build_metadata(result, len(train), len(val)))

    print()
    print(result["comparison"].round(3).to_string())
    print(f"\nChosen: {chosen}. {result['reason']}")
    print(f"Saved {MODEL_PATH.relative_to(MODEL_PATH.parents[1])} and metadata "
          f"in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
