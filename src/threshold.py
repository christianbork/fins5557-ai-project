"""Station 4 (decision): turn model scores into calibrated PDs and approve / refer / decline decisions.

Called by `python -m src.train` after the model is chosen:

1. Calibrate: the class-weighted model ranks customers well but overstates probabilities.
   A calibrator fitted on the validation split maps scores to PDs that match observed
   default rates (sigmoid or isotonic, whichever has the lower out-of-fold Brier score).
2. Threshold: choose the PD cut-off that maximises profit on the validation split, using
   the lender's economics (LGD, margin, EAD in config) instead of a default of 0.5.
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


def compare_thresholds(val_pd, y_val, test_pd, y_test, test_ead, profit_threshold: float) -> pd.DataFrame:
    """The profit cut-off (what the app uses) next to the balanced-accuracy cut-off (the Big Data
    exam's method), both chosen on the validation PDs and reported on the test split.

    Balanced accuracy weighs the two classes equally, so with 8% defaulters a missed defaulter
    implicitly costs (1 - 0.08) / 0.08, about 11 times a wrongly declined good customer. The
    lender's economics say LGD / margin = 0.60 / 0.15 = 4 times (a default loses 60% of the loan,
    a lost good customer forgoes a 15% margin). So the balanced-accuracy cut-off declines more people.
    """
    test_pd, y_test = np.asarray(test_pd, dtype=float), np.asarray(y_test)
    cutoffs = {"profit": profit_threshold, "balanced_accuracy": balanced_accuracy_threshold(y_val, val_pd)}
    rows = {}
    for rule, cut in cutoffs.items():
        m = evaluate(y_test, test_pd, cut)
        rows[rule] = {
            "cutoff": cut,
            "approval_rate": float(np.mean(test_pd < cut)),
            "tpr": m["tpr"],
            "tnr": m["tnr"],
            "precision": m["precision"],
            "balanced_accuracy": m["balanced_accuracy"],
            "realised_profit": realised_profit(test_pd, y_test, cut, ead=test_ead),
        }
    return pd.DataFrame(rows).T.rename_axis("rule")


def calibrate_and_set_threshold(model, X_val, y_val, X_test, y_test, raw_val_scores=None) -> dict:
    """Steps 1-4 in order. Returns the calibrated model and everything needed for metadata and plots."""
    cal = compare_calibration(model, X_val, y_val)
    method = cal["best"]
    calibrated = calibrate(model, X_val, y_val, method)

    val_pd = cal["oof"][method]  # out-of-fold PDs: an honest basis for choosing the cut-off
    curve = profit_curve(val_pd, y_val, X_val[EAD_COLUMN])
    threshold = choose_threshold(curve)

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
        "threshold_comparison": compare_thresholds(
            val_pd, y_val, test_pd, y_test, X_test[EAD_COLUMN], threshold
        ),
    }
