"""Model and fairness page: how the model was chosen, how good its PDs are, where the cut-off
comes from, and whether its decisions treat groups of applicants differently.

Reads only the summary files in models/ and reports/ (no applicant data), so it works on the
deployed app. Regenerate them with `python -m src.train` or `python -m src.reporting`.
"""

import json

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure  # not pyplot: Streamlit serves each session in its own thread

from src.config import (
    DI_THRESHOLD,
    DP_THRESHOLD,
    EO_THRESHOLD,
    METADATA_PATH,
    MIN_GROUP_SIZE,
    MODEL_NAMES,
    REPORTS_DIR,
    ROOT,
)

V1_METADATA = ROOT / "archive" / "v1-home-credit-baseline" / "metadata.json"

st.set_page_config(page_title="Model and fairness", page_icon="⚖️", layout="wide")


@st.cache_data
def load_reports() -> dict:
    """Metadata and the summary CSVs, read once."""
    return {
        "meta": json.loads(METADATA_PATH.read_text()),
        "v1": json.loads(V1_METADATA.read_text()) if V1_METADATA.exists() else None,
        "comparison": pd.read_csv(REPORTS_DIR / "model_comparison.csv", index_col="model"),
        "roc": pd.read_csv(REPORTS_DIR / "roc_points.csv"),
        "reliability": pd.read_csv(REPORTS_DIR / "reliability.csv"),
        "profit": pd.read_csv(REPORTS_DIR / "profit_curve.csv"),
        "cutoffs": pd.read_csv(REPORTS_DIR / "threshold_comparison.csv", index_col="rule"),
        "fairness": pd.read_csv(REPORTS_DIR / "fairness.csv", keep_default_na=False, na_values=[""]),
    }


try:
    r = load_reports()
except FileNotFoundError as e:
    st.error(f"A report file is missing, so this page can't be shown. Run `python -m src.reporting`. ({e})")
    st.stop()
meta, comparison = r["meta"], r["comparison"]
chosen, test = meta["chosen_model"], meta["test_metrics"]

st.title("Model and fairness")
st.caption(
    "How the model was chosen and checked. All numbers come from the validation and test splits of the "
    "Home Credit data, saved as summaries: no applicant data is shown or stored here."
)

# ---- Model choice -------------------------------------------------------------------------
st.header("1. Which model, and why")
cols = st.columns(4)
cols[0].metric("Chosen model", MODEL_NAMES[chosen])
cols[1].metric(
    "Test AUC",
    f"{test['auc']:.3f}",
    help="Chance the model ranks a random defaulter above a random good customer.",
)
cols[2].metric(
    "Test PR-AUC",
    f"{test['pr_auc']:.3f}",
    help="Ranking quality focused on the rare defaulters. Random = the default rate, about 0.08.",
)
cols[3].metric(
    "Test Brier score", f"{test['brier']:.3f}", help="Accuracy of the calibrated PDs (lower is better)."
)
st.write(meta["reason"])

table = comparison[["cv_auc", "auc", "pr_auc", "ks", "tpr", "tnr", "precision"]].rename(
    columns={
        "cv_auc": "CV AUC",
        "auc": "Validation AUC",
        "pr_auc": "PR-AUC",
        "ks": "KS",
        "tpr": "Defaulters caught",
        "tnr": "Good customers approved",
        "precision": "Precision",
    }
)
st.dataframe(
    table.style.format("{:.3f}").highlight_max(subset=["Validation AUC"], color="#d5f5e3"), width="stretch"
)
st.caption(
    "Validation split. Defaulters caught, good customers approved and precision are at each model's "
    "balanced-accuracy cut-off, so weighted and unweighted models compare fairly. A tree model must beat "
    "logistic regression by 0.01 AUC to be chosen over the more transparent scorecard."
)

if r["v1"] is not None:
    v1, v1t = r["v1"], r["v1"]["test_metrics"]
    with st.expander("Model v1 vs v2: did proper tuning help?"):
        st.dataframe(
            pd.DataFrame(
                {
                    "v1 (28 Oct)": [
                        v1["chosen_model"],
                        v1t["auc"],
                        v1["threshold"],
                        v1t["approval_rate"],
                        v1t["default_rate_approved"],
                    ],
                    "v2 (29 Oct)": [
                        chosen,
                        test["auc"],
                        meta["threshold"],
                        test["approval_rate"],
                        test["default_rate_approved"],
                    ],
                },
                index=["Model", "Test AUC", "Cut-off", "Approval rate", "Default rate among approved"],
            ).astype(str)
        )
        st.write(
            "v2 searched 30 settings per model, including how strongly to weight the rare defaulters. It "
            "changed almost nothing: v1 was already close to what application data allows, and class "
            "weighting made no difference once the scores were calibrated."
        )

roc_col, rel_col = st.columns(2)
with roc_col:
    fig = Figure(figsize=(5, 4.2))
    ax = fig.subplots()
    for model, points in r["roc"].groupby("model"):
        ax.plot(
            points["fpr"],
            points["tpr"],
            label=f"{model.replace('_', ' ')} (AUC {comparison.loc[model, 'auc']:.3f})",
            linewidth=2.2 if model == chosen else 1.2,
        )
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random (0.500)")
    ax.set(
        xlabel="Good customers declined (false positive rate)",
        ylabel="Defaulters caught (true positive rate)",
        title="ROC curves: validation split",
    )
    ax.legend(fontsize=8, loc="lower right")
    st.pyplot(fig, clear_figure=True)
with rel_col:
    fig = Figure(figsize=(5, 4.2))
    ax = fig.subplots()
    for series, points in r["reliability"].groupby("series", sort=False):
        ax.plot(points["mean_predicted"], points["observed"], "o-", label=series)
    top = r["reliability"][["mean_predicted", "observed"]].to_numpy().max()
    ax.plot([0, top], [0, top], "--", color="grey", label="perfect calibration")
    ax.set(
        xlabel="Predicted probability of default",
        ylabel="Observed default rate",
        title="Reliability: validation split",
    )
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)
st.caption(
    f"Calibration ({meta['calibration']['method']}) turns the model's scores into PDs that match observed "
    f"default rates: mean test PD {test['mean_pd']:.1%} vs actual {test['default_rate_all']:.1%}."
)

# ---- Cut-off --------------------------------------------------------------------------------
st.header("2. Where the cut-off comes from")
a = meta["assumptions"]
cutoffs = r["cutoffs"]
st.write(
    f"Approving a loan earns the margin ({a['margin']:.0%}) if the customer repays and loses LGD ({a['lgd']:.0%}) "
    f"if they default, so lending breaks even at PD = margin / (margin + LGD) = **{a['break_even_pd']:.0%}**. "
    f"The cut-off that earned the most on the validation split is **{meta['threshold']:.1%}**. Applicants within "
    f"{meta['refer_band']:.0%} of it are referred to an analyst."
)
fig = Figure(figsize=(9, 3.6))
ax = fig.subplots()
profit = r["profit"]
ax.plot(profit["threshold"], profit["realised_profit"] / 1e9, color="#1f618d")
ax.axvline(
    meta["threshold"], color="#1e8449", label=f"profit cut-off ({meta['threshold']:.1%}, used by the app)"
)
ax.axvline(
    cutoffs.loc["balanced_accuracy", "cutoff"],
    color="#b9770e",
    linestyle="--",
    label=f"balanced-accuracy cut-off ({cutoffs.loc['balanced_accuracy', 'cutoff']:.1%})",
)
ax.axvline(a["break_even_pd"], color="grey", linestyle=":", label=f"break-even PD ({a['break_even_pd']:.0%})")
ax.set(
    xlabel="Cut-off: decline when PD is at or above",
    ylabel="Realised profit (bn)",
    xlim=(0, 0.5),
    title="Profit on the validation split at each cut-off",
)
ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)

shown = cutoffs.rename(index={"profit": "Profit (app)", "balanced_accuracy": "Balanced accuracy"}).rename(
    columns={
        "cutoff": "Cut-off",
        "approval_rate": "Approval rate",
        "tpr": "Defaulters caught",
        "tnr": "Good customers approved",
        "precision": "Declined who would default",
        "balanced_accuracy": "Balanced accuracy",
        "realised_profit": "Profit",
    }
)
st.dataframe(
    shown.style.format({"Profit": "{:,.0f}"} | {c: "{:.1%}" for c in shown.columns if c != "Profit"}),
    width="stretch",
)
st.caption(
    "Test split. Balanced accuracy treats one missed defaulter as costing as much as about 11 wrongly declined "
    "good customers (the ratio of good customers to defaulters); the lender's economics say LGD / margin = 4. "
    "So the balanced-accuracy rule declines far more people and earns less."
)

# ---- Fairness ---------------------------------------------------------------------------------
st.header("3. Fairness: are groups treated differently?")
st.write(
    "Sex, age and family status are **never model inputs**, but other inputs can be correlated with them. "
    "This audit checks the automatic decisions (approve when PD is below the cut-off) on the test split, "
    "group by group (Code Bank W04-06)."
)
with st.expander("How to read the table"):
    st.markdown(
        f"- **Disparate-impact ratio**: the group's approval rate ÷ the most-approved group's. Flagged below "
        f"{DI_THRESHOLD} (the four-fifths rule).\n"
        f"- **Parity gap**: how many points fewer are approved than in the most-approved group. Flagged above "
        f"{DP_THRESHOLD:.0%}.\n"
        f"- **Equalised-odds gap**: the larger gap in *good customers approved* or *defaulters approved*. Flagged "
        f"above {EO_THRESHOLD:.0%}. Good customers approved is the equal-opportunity view: are creditworthy "
        "applicants treated alike?\n"
        "- **Mean PD vs default rate**: if they match, the PDs are accurate for that group, and a lower approval "
        "rate reflects higher measured risk rather than a biased score.\n"
        f"- Groups with fewer than {MIN_GROUP_SIZE} test applicants are listed but not judged, and their rates "
        "are not shown."
    )
fairness = r["fairness"]
tabs = st.tabs(list(fairness["attribute"].unique()))
for tab, (attribute, groups) in zip(tabs, fairness.groupby("attribute", sort=False), strict=True):
    with tab:
        view = groups.set_index("group")[
            [
                "n",
                "approval_rate",
                "di_ratio",
                "dp_gap",
                "good_customers_approved",
                "defaulters_approved",
                "eo_gap",
                "mean_pd",
                "default_rate",
                "flags",
            ]
        ].rename(
            columns={
                "n": "Applicants",
                "approval_rate": "Approval rate",
                "di_ratio": "Disparate-impact ratio",
                "dp_gap": "Parity gap",
                "good_customers_approved": "Good customers approved",
                "defaulters_approved": "Defaulters approved",
                "eo_gap": "Equalised-odds gap",
                "mean_pd": "Mean PD",
                "default_rate": "Default rate",
                "flags": "Flags",
            }
        )
        pct = [
            "Approval rate",
            "Parity gap",
            "Good customers approved",
            "Defaulters approved",
            "Equalised-odds gap",
            "Mean PD",
            "Default rate",
        ]
        st.dataframe(
            view.style.format(
                {c: "{:.1%}" for c in pct} | {"Disparate-impact ratio": "{:.2f}", "Applicants": "{:,}"},
                na_rep="–",
            ),
            width="stretch",
        )
        judged = groups[groups["judged"]]
        worst = judged.loc[judged["di_ratio"].idxmin()]
        st.caption(
            f"Lowest disparate-impact ratio: {worst['group']} at {worst['di_ratio']:.2f} "
            f"({'below' if worst['di_ratio'] < DI_THRESHOLD else 'above'} the {DI_THRESHOLD} threshold). "
            f"Largest gap between mean PD and actual default rate: {judged['pd_gap'].abs().max():.1%}."
        )
judged = fairness[fairness["judged"]]
below = judged[judged["di_ratio"] < DI_THRESHOLD]
four_fifths = (
    "No group falls below the four-fifths rule"
    if below.empty
    else f"{len(below)} group(s) fall below the four-fifths rule ({', '.join(below['group'])})"
)
st.info(
    f"**What this shows.** {four_fifths}, and the largest gap between mean PD and the actual default rate in any "
    f"group is {judged['pd_gap'].abs().max():.1%}, so the PDs are accurate within every group. The approval gaps "
    "follow real differences in default rates, and when default rates differ a model can't be both accurate "
    "within every group and equal on error rates. Even so, creditworthy younger applicants are declined more "
    "often than older ones. Most of that comes through the external credit scores, which are correlated with age "
    "(a longer credit history scores better), then loan length, job tenure and time since the ID was renewed. "
    "Whether to accept, reduce or offset it is a policy decision, discussed in the report."
)
