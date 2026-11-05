"""Station 3 (Model Design). Turn model scores into calibrated PDs and approve / refer / decline decisions.

Called by `python -m src.train` after the model is chosen:

1. Calibrate: the class-weighted model ranks customers well but overstates probabilities.
   A calibrator fitted on the validation split maps scores to PDs that match observed
   default rates (sigmoid or isotonic, whichever has the lower out-of-fold Brier score).
2. Threshold: compute the cut-off under each of three decision rules (config.DECISION_RULES:
   profit, a 5 : 1 cost ratio and balanced accuracy) and use the one set by config.DECISION_RULE.
   The other two are reported alongside it, so the effect of the assumption is visible.
3. Decide: approve below the cut-off, decline above it, refer a band around it to an analyst.
4. Report the test split once, at the chosen cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    CALIBRATION_METHODS,
    COST_RATIO,
    DECISION_RULE,
    DECISION_RULES,
    EAD_COLUMN,
    LGD,
    MARGIN,
    RANDOM_STATE,
    REFER_BAND,
)
from src.metrics import balanced_accuracy_threshold, evaluate

APPROVE, REFER, DECLINE = "approve", "refer", "decline"
THRESHOLD_GRID = np.round(np.arange(0.01, 0.80, 0.005), 3)


# ---- 1. Calibration --------------------------------------------------------------


def calibrate(model, X_val, y_val, method: str) -> CalibratedClassifierCV:
    """Fit a calibrator on top of the already-trained model (the model itself is frozen, not refitted)."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_val, y_val)


def compare_calibration(model, X_val, y_val, methods=CALIBRATION_METHODS, folds: int = 5) -> dict:
    """Out-of-fold calibrated PDs and Brier score for each method.

    Each validation row is calibrated by a calibrator that never saw it, so the comparison
    isn't flattered by isotonic regression memorising the data. Returns
    {"brier": {method: score}, "oof": {method: array}, "best": method}.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    oof = {
        m: cross_val_predict(
            CalibratedClassifierCV(FrozenEstimator(model), method=m),
            X_val,
            y_val,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        for m in methods
    }
    brier = {m: brier_score_loss(y_val, p) for m, p in oof.items()}
    return {"brier": brier, "oof": oof, "best": min(brier, key=brier.get)}


# ---- 2. Profit and threshold -----------------------------------------------------


def break_even_pd(lgd: float = LGD, margin: float = MARGIN) -> float:
    """PD at which approving is expected to break even: (1 - PD) x margin = PD x LGD."""
    return margin / (margin + lgd)


def expected_profit(pd_, threshold: float, lgd: float = LGD, margin: float = MARGIN, ead=1.0) -> float:
    """Profit the model *expects* from approving everyone with PD below the threshold.

    Per approved customer: (1 - PD) x margin x EAD  -  PD x LGD x EAD (the expected loss).
    Only meaningful if the PDs are calibrated.
    """
    pd_ = np.asarray(pd_, dtype=float)
    ead = np.broadcast_to(np.asarray(ead, dtype=float), pd_.shape)
    approved = pd_ < threshold
    return float(np.sum(approved * ead * ((1 - pd_) * margin - pd_ * lgd)))


def realised_profit(pd_, y, threshold: float, lgd: float = LGD, margin: float = MARGIN, ead=1.0) -> float:
    """Profit actually earned from approving everyone below the threshold, using real outcomes:
    margin x EAD for each approved customer who repaid, minus LGD x EAD for each who defaulted."""
    pd_, y = np.asarray(pd_, dtype=float), np.asarray(y)
    ead = np.broadcast_to(np.asarray(ead, dtype=float), pd_.shape)
    approved = pd_ < threshold
    return float(np.sum(approved * ead * np.where(y == 1, -lgd, margin)))


def profit_curve(
    pd_, y, ead, thresholds=THRESHOLD_GRID, lgd: float = LGD, margin: float = MARGIN
) -> pd.DataFrame:
    """Realised and expected profit, and approval rate, at each candidate cut-off."""
    pd_ = np.asarray(pd_, dtype=float)
    rows = [
        {
            "threshold": t,
            "approval_rate": float(np.mean(pd_ < t)),
            "realised_profit": realised_profit(pd_, y, t, lgd, margin, ead),
            "expected_profit": expected_profit(pd_, t, lgd, margin, ead),
        }
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame) -> float:
    """The cut-off with the highest realised profit on the validation split."""
    return float(curve.loc[curve["realised_profit"].idxmax(), "threshold"])


# ---- 3. Decision -----------------------------------------------------------------


def decide(pd_, threshold: float, band: float = REFER_BAND):
    """approve / refer / decline for one PD (returns a str) or many (returns an array).

    approve: PD < threshold - band            (clearly profitable)
    decline: PD > threshold + band            (clearly loss-making)
    refer:   anything in between, edges included (too close to call, an analyst decides)
    """
    lower, upper = round(threshold - band, 6), round(threshold + band, 6)
    pd_arr = np.asarray(pd_, dtype=float)
    if np.isnan(pd_arr).any():
        raise ValueError("PD contains missing values; cannot make a decision.")
    out = np.select([pd_arr < lower, pd_arr > upper], [APPROVE, DECLINE], default=REFER)
    return str(out) if out.ndim == 0 else out


def cost_ratio_threshold(ratio: float = COST_RATIO) -> float:
    """Cut-off when approving a defaulter costs `ratio` times as much as declining a good customer.

    Approving someone with probability of default PD has expected cost PD x ratio; declining them
    has expected cost (1 - PD) x 1. Approving is cheaper while PD < 1 / (1 + ratio): 16.7% at 5 : 1
    (Hofmann, 1994). Needs calibrated PDs, which the model has.
    """
    return round(1 / (1 + ratio), 4)


def misclassification_cost(pd_, y, threshold: float, ratio: float = COST_RATIO) -> float:
    """Total cost of the automatic decisions in cost-ratio units: ratio per approved defaulter,
    1 per declined good customer."""
    pd_, y = np.asarray(pd_, dtype=float), np.asarray(y)
    approved = pd_ < threshold
    return float(ratio * np.sum(approved & (y == 1)) + np.sum(~approved & (y == 0)))


def choose_cutoffs(val_pd, y_val, val_ead) -> dict:
    """The cut-off under each decision rule, chosen on the validation split (the cost-ratio rule
    needs no data: it follows from the ratio)."""
    return {
        "profit": choose_threshold(profit_curve(val_pd, y_val, val_ead)),
        "cost_ratio": cost_ratio_threshold(),
        "balanced_accuracy": round(balanced_accuracy_threshold(y_val, val_pd), 4),
    }


# ---- 4. Final test report -----------------------------------------------------------


def test_report(pd_, y, ead, threshold: float, band: float = REFER_BAND) -> dict:
    """Test-split results at the chosen cut-off, reported once.

    The confusion matrix treats "decline" (PD >= threshold) as predicting default, ignoring
    the refer band, so it describes the automatic decision rule.
    """
    pd_, y = np.asarray(pd_, dtype=float), np.asarray(y)
    declined = (pd_ >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, declined, labels=[0, 1]).ravel()
    decisions = pd.Series(decide(pd_, threshold, band))
    approved = declined == 0
    return {
        **{k: float(v) for k, v in evaluate(y, pd_, threshold).items()},
        "threshold": threshold,
        "confusion_matrix": {
            "approved_repaid": int(tn),
            "declined_repaid": int(fp),
            "approved_defaulted": int(fn),
            "declined_defaulted": int(tp),
        },
        "approval_rate": float(approved.mean()),
        "default_rate_approved": float(y[approved].mean()) if approved.any() else None,
        "default_rate_all": float(y.mean()),
        "decision_shares": {d: float((decisions == d).mean()) for d in (APPROVE, REFER, DECLINE)},
        "realised_profit": realised_profit(pd_, y, threshold, ead=ead),
        "mean_pd": float(pd_.mean()),
    }


def compare_thresholds(
    cutoffs: dict, test_pd, y_test, test_ead, band: float = REFER_BAND, test_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """All decision rules side by side on the test split: one row per rule (config.DECISION_RULES).

    The same model gives different decisions depending on how the lender weighs an approved
    defaulter against a declined good customer: about 4 : 1 under the profit assumptions
    (LGD / margin), 5 : 1 under the cost-ratio rule and about 11 : 1 under balanced accuracy
    (good customers / defaulters). Columns describe the automatic rule at the cut-off, the decision
    shares with the refer band, profit under LGD and MARGIN, the cost at COST_RATIO and, if the test
    rows (with protected attributes) are given, the lowest disparate-impact ratio of any judged group.
    """
    from src.fairness import fairness_table  # imported here: fairness needs the data module

    test_pd, y_test = np.asarray(test_pd, dtype=float), np.asarray(y_test)
    rows = {}
    for rule in DECISION_RULES:
        cut = cutoffs[rule]
        m = evaluate(y_test, test_pd, cut)
        decisions = np.asarray(decide(test_pd, cut, band))
        approved = test_pd < cut
        row = {
            "cutoff": cut,
            "approval_rate": float(approved.mean()),
            "tpr": m["tpr"],
            "tnr": m["tnr"],
            "precision": m["precision"],
            "balanced_accuracy": m["balanced_accuracy"],
            "default_rate_approved": float(y_test[approved].mean()) if approved.any() else float("nan"),
            "approve_share": float(np.mean(decisions == APPROVE)),
            "refer_share": float(np.mean(decisions == REFER)),
            "decline_share": float(np.mean(decisions == DECLINE)),
            "defaulters_auto_approved": float(np.mean(decisions[y_test == 1] == APPROVE)),
            "good_auto_approved": float(np.mean(decisions[y_test == 0] == APPROVE)),
            "realised_profit": realised_profit(test_pd, y_test, cut, ead=test_ead),
            "cost_at_ratio": misclassification_cost(test_pd, y_test, cut),
        }
        if test_df is not None:
            judged = fairness_table(test_df, test_pd, cut).query("judged")
            row["lowest_di_ratio"] = float(judged["di_ratio"].min())
            row["groups_below_four_fifths"] = int((judged["di_ratio"] < 0.8).sum())
        rows[rule] = row
    return pd.DataFrame(rows).T.rename_axis("rule")


def calibrate_and_set_threshold(model, X_val, y_val, X_test, y_test, raw_val_scores=None) -> dict:
    """Steps 1-4 in order. Returns the calibrated model and everything needed for metadata and plots."""
    cal = compare_calibration(model, X_val, y_val)
    method = cal["best"]
    calibrated = calibrate(model, X_val, y_val, method)

    val_pd = cal["oof"][method]  # out-of-fold PDs: an honest basis for choosing the cut-off
    curve = profit_curve(val_pd, y_val, X_val[EAD_COLUMN])
    cutoffs = choose_cutoffs(val_pd, y_val, X_val[EAD_COLUMN])
    threshold = cutoffs[DECISION_RULE]

    test_pd = calibrated.predict_proba(X_test)[:, 1]
    return {
        "model": calibrated,
        "method": method,
        "brier": {
            **(
                {"uncalibrated": brier_score_loss(y_val, raw_val_scores)}
                if raw_val_scores is not None
                else {}
            ),
            **cal["brier"],
        },
        "val_pd": val_pd,
        "curve": curve,
        "threshold": threshold,
        "test": test_report(test_pd, y_test, X_test[EAD_COLUMN], threshold),
        "decision_rule": DECISION_RULE,
        "cutoffs": cutoffs,
        "threshold_comparison": compare_thresholds(cutoffs, test_pd, y_test, X_test[EAD_COLUMN]),
    }


# ---- Re-choose the cut-off without retraining ------------------------------------------------
def recut() -> dict:
    """Apply config.DECISION_RULE to the saved model: new cut-off, test report, rule comparison,
    metadata, fairness table and model card. The model itself is not retrained, so AUC, PR-AUC and
    Brier score are unchanged. Run with: python -m src.threshold (needs the Kaggle data in data/raw/)."""
    import json

    import joblib

    from src.config import METADATA_PATH, MODEL_PATH, REPORTS_DIR
    from src.data import features_target, load_clean, split
    from src.model_card import write_model_card
    from src.reporting import write_app_reports

    model = joblib.load(MODEL_PATH)
    meta = json.loads(METADATA_PATH.read_text())
    _, val, test = split(load_clean())
    X_val, y_val = features_target(val)
    X_test, y_test = features_target(test)
    val_predictions = pd.read_csv(REPORTS_DIR / "validation_predictions.csv")
    val_pd = val_predictions["calibrated_pd"].to_numpy()  # the out-of-fold PDs training chose the cut-off on
    cutoffs = choose_cutoffs(val_pd, y_val, X_val[EAD_COLUMN])
    threshold = cutoffs[DECISION_RULE]
    test_pd = model.predict_proba(X_test[meta["input_columns"]])[:, 1]
    report = test_report(test_pd, y_test, X_test[EAD_COLUMN], threshold)
    comparison = compare_thresholds(cutoffs, test_pd, y_test, X_test[EAD_COLUMN], test_df=test)

    meta.update(
        threshold=threshold,
        decision_rule=DECISION_RULE,
        cost_ratio=COST_RATIO,
        cutoffs=cutoffs,
        test_metrics=report,
        threshold_comparison=comparison.round(4).to_dict(orient="index"),
    )
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2))
    comparison.round(4).to_csv(REPORTS_DIR / "threshold_comparison.csv")
    write_app_reports(test, test_pd, threshold, val_predictions, meta["chosen_model"])
    write_model_card()
    return {"cutoffs": cutoffs, "threshold": threshold, "comparison": comparison}


if __name__ == "__main__":
    out = recut()
    print(f"Decision rule: {DECISION_RULE}, cut-off {out['threshold']:.4f}; all rules: {out['cutoffs']}")
    print(out["comparison"].round(3).to_string())
