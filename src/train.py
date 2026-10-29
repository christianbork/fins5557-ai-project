"""Station 3 (models): train, tune and compare three credit-default models, then calibrate and save
the chosen one.

Run from the project root (about 30 minutes on the 184k training rows, see Step 5c):

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
    reports/tuning_results.csv          every tuning candidate's CV scores
    reports/threshold_comparison.csv    profit vs balanced-accuracy cut-off on the test split
    reports/fairness.csv, roc_points.csv, reliability.csv   aggregates for the app (src/reporting.py)
"""

from __future__ import annotations

import json
import math
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
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
    N_ITER,
    RANDOM_STATE,
    REFER_BAND,
    REPORTS_DIR,
    TUNING_SCORING,
)
from src.data import features_target, load_clean, split
from src.features import add_features, build_preprocessor, feature_columns
from src.metrics import balanced_accuracy_threshold, evaluate
from src.reporting import write_app_reports
from src.threshold import break_even_pd, calibrate_and_set_threshold
from src.validation import build_reference_stats, save_reference_stats

BASELINE = "logistic_regression"


def search_spaces(y_train: pd.Series) -> dict:
    """What each model's tuning may try (model v2, Step 5c).

    The imbalance correction is tuned rather than fixed, as the Big Data exam tuned its random
    forest's sampsize: no weighting, partial (square root of the ratio) or full (neg/pos). Without
    weighting the raw scores are too low rather than too high, but calibration fixes either, so
    the search can pick whichever ranks defaulters best. Random forest max_features is the exam's
    mtry (how many inputs each split may choose from).
    """
    negatives, positives = np.bincount(y_train)
    ratio = negatives / positives
    return {
        "logistic_regression": {  # full grid, 5 x 2 = 10 combinations
            "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],  # inverse regularisation: smaller C shrinks more
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__max_features": ["sqrt", 0.2, 0.33, 0.5],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_leaf": [20, 50, 100, 200],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "xgboost": {
            "model__max_depth": randint(2, 7),
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__n_estimators": randint(200, 801),
            "model__min_child_weight": randint(1, 21),
            "model__subsample": uniform(0.6, 0.4),  # 0.6-1.0: share of rows each tree sees
            "model__colsample_bytree": uniform(0.5, 0.5),  # 0.5-1.0: share of inputs each tree sees
            "model__scale_pos_weight": [1.0, float(np.sqrt(ratio)), float(ratio)],
        },
    }


def make_models(y_train: pd.Series) -> dict:
    """The three candidate models with their fixed settings; the search overrides the tuned ones.

    The defaults here are v1's: class weighting makes a missed defaulter cost more during training
    than a false alarm. It also pushes the predicted probabilities upwards, which is one reason the
    chosen model is recalibrated before its scores are used as PDs.
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
    return Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("preprocess", build_preprocessor(feature_set)),
            ("model", model),
        ]
    )


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


def _plain(value):
    """numpy number -> Python number (for JSON), rounded so the metadata stays readable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return round(float(value), 4)
    return value


def make_search(name: str, pipeline: Pipeline, space: dict, cv, n_iter: int = N_ITER):
    """Full grid for logistic regression or any small all-list space, otherwise a randomised search.

    Both score every candidate on AUC and PR-AUC (TUNING_SCORING) and refit the best by AUC.
    """
    lists = all(isinstance(v, list) for v in space.values())
    small = lists and math.prod(len(v) for v in space.values()) <= n_iter
    common = {"scoring": TUNING_SCORING, "refit": "roc_auc", "cv": cv, "n_jobs": -1}
    if name == BASELINE or small:
        return GridSearchCV(pipeline, space, **common)
    return RandomizedSearchCV(pipeline, space, n_iter=n_iter, random_state=RANDOM_STATE, **common)


def _tuning_table(name: str, search) -> pd.DataFrame:
    """One row per candidate tried: its settings, CV AUC and PR-AUC, rank and fit time."""
    cv = pd.DataFrame(search.cv_results_)
    settings = cv.filter(like="param_model__").rename(columns=lambda c: c.removeprefix("param_model__"))
    scores = pd.DataFrame(
        {
            "model": name,
            "rank": cv["rank_test_roc_auc"],
            "cv_auc": cv["mean_test_roc_auc"],
            "cv_auc_std": cv["std_test_roc_auc"],
            "cv_pr_auc": cv["mean_test_pr_auc"],
            "fit_seconds": cv["mean_fit_time"],
        }
    )
    return pd.concat([scores, settings.astype(object)], axis=1).sort_values("rank")


def train_all(
    train: pd.DataFrame,
    val: pd.DataFrame,
    param_grids: dict | None = None,
    cv_folds: int = CV_FOLDS,
    n_iter: int = N_ITER,
    verbose: bool = True,
) -> dict:
    """Tune each model with CV on train, score it on validation, choose one.

    param_grids: search space per model (default: search_spaces(); tests pass tiny grids).
    The comparison's threshold metrics (TPR, TNR, precision, F1, balanced accuracy) are taken at
    each model's own balanced-accuracy cut-off on validation: a fixed 0.5 would make an unweighted
    model, whose raw scores sit around 8%, look as if it caught no defaulters at all.

    Returns a dict with the fitted pipelines, the comparison table, validation predictions,
    best parameters, every candidate's CV scores, and the chosen model with its reason.
    """
    X_train, y_train = features_target(train)
    X_val, y_val = features_target(val)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    spaces = param_grids or search_spaces(y_train)

    fitted, rows, preds, params, tuning = {}, {}, {}, {}, []
    for name, model in make_models(y_train).items():
        start = time.perf_counter()
        search = make_search(name, build_pipeline(model), spaces[name], cv, n_iter)
        search.fit(X_train, y_train)  # refits the best settings on the full training split
        fitted[name] = search.best_estimator_
        params[name] = {k.removeprefix("model__"): _plain(v) for k, v in search.best_params_.items()}
        preds[name] = search.predict_proba(X_val)[:, 1]
        cut = balanced_accuracy_threshold(y_val, preds[name])
        rows[name] = {
            "cv_auc": search.best_score_,
            "cv_pr_auc": search.cv_results_["mean_test_pr_auc"][search.best_index_],
            **evaluate(y_val, preds[name], cut),
            "ba_threshold": cut,
        }
        tuning.append(_tuning_table(name, search))
        if verbose:
            print(
                f"{name:20s} CV AUC {search.best_score_:.3f}  val AUC {rows[name]['auc']:.3f}  "
                f"PR-AUC {rows[name]['pr_auc']:.3f}  {params[name]}  ({time.perf_counter() - start:.0f}s)"
            )

    comparison = pd.DataFrame(rows).T
    comparison.index.name = "model"
    chosen, reason = choose_model(comparison)
    return {
        "pipelines": fitted,
        "comparison": comparison,
        "val_predictions": pd.DataFrame({"default": y_val.to_numpy(), **preds}, index=y_val.index),
        "best_params": params,
        "tuning": pd.concat(tuning, ignore_index=True),
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
        "tuning": {
            "method": f"randomised search ({N_ITER} settings per tree model), full grid for logistic regression",
            "cv_folds": CV_FOLDS,
            "refit_metric": "roc_auc",
            "best_params_all_models": result["best_params"],
        },
        "validation_metrics": {k: round(float(v), 4) for k, v in result["comparison"].loc[chosen].items()},
        "all_models": result["comparison"].round(4).to_dict(orient="index"),
        "calibration": {
            "method": decision["method"],
            "validation_brier": {k: round(float(v), 4) for k, v in decision["brier"].items()},
        },
        "threshold": decision["threshold"],
        "refer_band": REFER_BAND,
        "assumptions": {
            "lgd": LGD,
            "margin": MARGIN,
            "ead": EAD_COLUMN,
            "break_even_pd": round(break_even_pd(), 4),
        },
        "test_metrics": decision["test"],
        "threshold_comparison": decision["threshold_comparison"].round(4).to_dict(orient="index"),
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


def save_artifacts(
    model, metadata: dict, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
) -> None:
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
    decision = calibrate_and_set_threshold(
        result["pipelines"][chosen],
        X_val,
        y_val,
        X_test,
        y_test,
        raw_val_scores=result["val_predictions"][chosen],
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    result["comparison"].round(4).to_csv(REPORTS_DIR / "model_comparison.csv")
    result["tuning"].round(4).to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    decision["threshold_comparison"].round(4).to_csv(REPORTS_DIR / "threshold_comparison.csv")
    decision["curve"].round(4).to_csv(REPORTS_DIR / "profit_curve.csv", index=False)
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(decision["test"], indent=2))
    result["val_predictions"].assign(calibrated_pd=decision["val_pd"]).to_csv(
        REPORTS_DIR / "validation_predictions.csv", index=False
    )

    n_rows = {"train": len(train), "validation": len(val), "test": len(test)}
    save_artifacts(decision["model"], build_metadata(result, decision, list(X_val.columns), n_rows))
    save_reference_stats(build_reference_stats(features_target(train)[0]))  # input checks + monitoring
    write_app_reports(  # fairness audit and chart data for the app's Model and fairness page
        test,
        decision["model"].predict_proba(X_test)[:, 1],
        decision["threshold"],
        result["val_predictions"].assign(calibrated_pd=decision["val_pd"]),
        chosen,
    )

    t = decision["test"]
    print(
        f"Calibration: {decision['method']} (validation Brier "
        + ", ".join(f"{k} {v:.4f}" for k, v in decision["brier"].items())
        + ")"
    )
    print(
        f"Threshold: decline at PD >= {decision['threshold']:.3f} (break-even PD {break_even_pd():.3f}), "
        f"refer band +/- {REFER_BAND}"
    )
    print(
        f"Test: AUC {t['auc']:.3f}, Brier {t['brier']:.4f}, approval rate {t['approval_rate']:.1%}, "
        f"default rate among approved {t['default_rate_approved']:.1%} (vs {t['default_rate_all']:.1%} overall)"
    )
    print("\nCut-off rules on the test split (the app uses 'profit'):")
    print(decision["threshold_comparison"].round(3).to_string())
    print(
        f"Saved {MODEL_PATH.relative_to(MODEL_PATH.parents[1])} and metadata "
        f"in {time.perf_counter() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
