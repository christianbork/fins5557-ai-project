"""Step 10: the summary files behind the app's "Model and fairness" page.

The deployed app never has the Kaggle data, so everything the page shows is written here as
aggregates only (group rates, curve points), with no applicant rows:

    python -m src.reporting              # also run at the end of python -m src.train
    python -m src.reporting model-card   # docs/MODEL_CARD.md only (src/model_card.py)

Writes to reports/:
    fairness.csv     fairness audit on the test split (src/fairness.py); groups smaller than
                     MIN_GROUP_SIZE keep only their count, because a rate over one or a few
                     applicants would reveal their own data
    roc_points.csv   ROC curve of each model on the validation split, 101 points each
    reliability.csv  reliability curve (mean predicted PD vs observed default rate in 10 bins),
                     before and after calibration, validation split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config import REPORTS_DIR, TARGET
from src.fairness import fairness_table
from src.labels import model_label

FPR_GRID = np.linspace(0, 1, 101)


def roc_points(val_predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Each model's ROC curve on a common false-positive-rate grid (enough points for a chart)."""
    rows = []
    for model in models:
        fpr, tpr, _ = roc_curve(val_predictions[TARGET], val_predictions[model])
        rows.append(pd.DataFrame({"model": model, "fpr": FPR_GRID, "tpr": np.interp(FPR_GRID, fpr, tpr)}))
    return pd.concat(rows, ignore_index=True)


def reliability_points(y, scores: dict[str, np.ndarray], bins: int = 10) -> pd.DataFrame:
    """Mean predicted PD vs observed default rate in equal-sized bins, for each set of scores."""
    rows = []
    for series, score in scores.items():
        frame = pd.DataFrame({"score": np.asarray(score, dtype=float), "y": np.asarray(y)})
        frame["bin"] = pd.qcut(frame["score"].rank(method="first"), bins, labels=False)
        agg = frame.groupby("bin").agg(
            mean_predicted=("score", "mean"), observed=("y", "mean"), n=("y", "size")
        )
        rows.append(agg.assign(series=series).reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)[["series", "mean_predicted", "observed", "n"]]


def write_app_reports(
    test: pd.DataFrame,
    test_pd,
    threshold: float,
    val_predictions: pd.DataFrame,
    chosen: str,
    out_dir=REPORTS_DIR,
) -> None:
    """Write the three aggregate files. val_predictions: validation scores per model plus calibrated_pd."""
    out_dir.mkdir(exist_ok=True)
    fairness = fairness_table(test, test_pd, threshold)
    # A "rate" over one or a few people is their own data, so small groups keep only their count
    rates = [
        c for c in fairness.columns if c not in ("attribute", "group", "n", "judged", "reference", "flags")
    ]
    fairness.loc[~fairness["judged"], rates] = np.nan
    fairness.round(4).to_csv(out_dir / "fairness.csv", index=False)
    models = [c for c in val_predictions.columns if c not in (TARGET, "calibrated_pd")]
    roc_points(val_predictions, models).round(4).to_csv(out_dir / "roc_points.csv", index=False)
    scores = {
        f"{model_label(chosen)} (raw score)": val_predictions[chosen],
        "Calibrated PD": val_predictions["calibrated_pd"],
    }
    reliability_points(val_predictions[TARGET], scores).round(4).to_csv(
        out_dir / "reliability.csv", index=False
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["model-card"]:  # needs only models/ and reports/, not the Kaggle data
        from src.model_card import write_model_card

        print(f"Wrote {write_model_card()}")
        sys.exit()
    from src.data import features_target, load_clean, split
    from src.predict import load_artifacts

    artifacts = load_artifacts()
    meta = artifacts["metadata"]
    _, _, test = split(load_clean())
    X_test, _ = features_target(test)
    test_pd = artifacts["model"].predict_proba(X_test[meta["input_columns"]])[:, 1]
    val_predictions = pd.read_csv(REPORTS_DIR / "validation_predictions.csv")
    write_app_reports(test, test_pd, meta["threshold"], val_predictions, meta["chosen_model"])
    print(f"Wrote fairness.csv, roc_points.csv and reliability.csv to {REPORTS_DIR}")
