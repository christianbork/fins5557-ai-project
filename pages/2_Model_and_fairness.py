"""Model and fairness page: how the model was chosen, how good its PDs are, where the cut-off
comes from, and whether its decisions treat groups of applicants differently.

Reads only the summary files in models/ and reports/ (no applicant data), so it works on the
deployed app. Regenerate them with `python -m src.train` or `python -m src.reporting`.
"""

import json

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure  # not pyplot: Streamlit serves each session in its own thread
from matplotlib.ticker import FuncFormatter

from src.config import (
    CURRENCY_NOTE,
    DI_THRESHOLD,
    DP_THRESHOLD,
    EO_THRESHOLD,
    METADATA_PATH,
    MIN_GROUP_SIZE,
    REPORTS_DIR,
    ROOT,
)
from src.formatting import count, format_table, money, money_short, number, pct, ratio, score
from src.labels import RULES, flag_text, headers, model_choice_reason, model_label

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
cols[0].metric("Chosen model", model_label(chosen))
cols[1].metric(
    "Test AUC",
    score(test["auc"]),
    help="Chance the model ranks a random defaulter above a random good customer.",
)
cols[2].metric(
    "Test PR-AUC",
    score(test["pr_auc"]),
    help="Ranking quality focused on the rare defaulters. Random = the default rate, about 0.08.",
)
cols[3].metric(
    "Test Brier score",
    score(test["brier"]),
    help="Accuracy of the calibrated probabilities of default (PDs): lower is better.",
)
st.write(model_choice_reason(comparison))

table = headers(
    comparison[["cv_auc", "auc", "pr_auc", "ks", "tpr", "tnr", "precision"]].rename(index=model_label)
)
st.dataframe(
    table.style.format(score).highlight_max(subset=["Validation AUC"], color="#d5f5e3"), width="stretch"
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
                    version: [
                        model_label(m["chosen_model"]),
                        score(t["auc"]),
                        pct(m["threshold"]),
                        pct(t["approval_rate"]),
                        pct(t["default_rate_approved"]),
                    ]
                    for version, m, t in [("Model v1 (28 Oct)", v1, v1t), ("Model v2 (29 Oct)", meta, test)]
                },
                index=["Model", "Test AUC", "Cut-off", "Approval rate", "Default rate among approved"],
            )
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
            label=f"{model_label(model)} (AUC {score(comparison.loc[model, 'auc'])})",
            linewidth=2.2 if model == chosen else 1.2,
        )
    ax.plot([0, 1], [0, 1], "--", color="grey", label=f"Random ({score(0.5)})")
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
        # older report files name the series by model code, e.g. "xgboost (raw score)"
        label = series.replace(chosen, model_label(chosen))
        ax.plot(points["mean_predicted"], points["observed"], "o-", label=label[:1].upper() + label[1:])
    top = r["reliability"][["mean_predicted", "observed"]].to_numpy().max()
    ax.plot([0, top], [0, top], "--", color="grey", label="Perfect calibration")
    ax.set(
        xlabel="Predicted probability of default",
        ylabel="Observed default rate",
        title="Reliability: validation split",
    )
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)
st.caption(
    f"Calibration ({meta['calibration']['method']}) turns the model's scores into probabilities of default "
    f"(PDs) that match observed default rates: mean test PD {pct(test['mean_pd'])} vs actual "
    f"{pct(test['default_rate_all'])}."
)

# ---- Cut-off --------------------------------------------------------------------------------
st.header("2. Where the cut-off comes from")
a = meta["assumptions"]
cutoffs = r["cutoffs"]
st.write(
    f"Approving a loan earns the margin ({pct(a['margin'])}) if the customer repays and loses the loss given "
    f"default (LGD, {pct(a['lgd'])}) if they default, so lending breaks even at PD = margin / (margin + LGD) = "
    f"**{pct(a['break_even_pd'])}**. The cut-off that earned the most on the validation split is "
    f"**{pct(meta['threshold'])}**. Applicants within {pct(meta['refer_band'])} of it are referred to an analyst."
)
fig = Figure(figsize=(9, 3.6))
ax = fig.subplots()
profit = r["profit"]
ax.plot(profit["threshold"], profit["realised_profit"], color="#1f618d")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: money_short(v)))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v)))
ax.axvline(
    meta["threshold"], color="#1e8449", label=f"Profit cut-off ({pct(meta['threshold'])}, used by the app)"
)
ax.axvline(
    cutoffs.loc["balanced_accuracy", "cutoff"],
    color="#b9770e",
    linestyle="--",
    label=f"Balanced-accuracy cut-off ({pct(cutoffs.loc['balanced_accuracy', 'cutoff'])})",
)
ax.axvline(
    a["break_even_pd"], color="grey", linestyle=":", label=f"Break-even PD ({pct(a['break_even_pd'])})"
)
ax.set(
    xlabel="Cut-off: decline when PD is at or above",
    ylabel="Realised profit",
    xlim=(0, 0.5),
    title="Profit on the validation split at each cut-off",
)
ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)

shown = headers(cutoffs.rename(index=RULES).rename(columns={"precision": "Declined who would default"}))
st.dataframe(
    shown.style.format({c: money if c == "Profit" else pct for c in shown.columns}),
    width="stretch",
)
st.caption(
    "Test split. Balanced accuracy treats one missed defaulter as costing as much as about 11 wrongly declined "
    "good customers (the ratio of good customers to defaulters); the lender's economics say LGD / margin = 4. "
    "So the balanced-accuracy rule declines far more people and earns less. " + CURRENCY_NOTE
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
        f"{ratio(DI_THRESHOLD)} (the four-fifths rule).\n"
        f"- **Parity gap**: how many points fewer are approved than in the most-approved group. Flagged above "
        f"{pct(DP_THRESHOLD)}.\n"
        f"- **Equalised-odds gap**: the larger gap in *good customers approved* or *defaulters approved*. Flagged "
        f"above {pct(EO_THRESHOLD)}. Good customers approved is the equal-opportunity view: are creditworthy "
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
        view = groups.assign(flags=groups["flags"].fillna("").map(flag_text)).set_index("group")[
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
        ]
        rates = {c: pct for c in view.columns if c not in ("n", "di_ratio", "flags")}
        st.dataframe(
            headers(format_table(view, rates | {"n": number, "di_ratio": ratio})),
            width="stretch",
        )
        judged = groups[groups["judged"]]
        worst = judged.loc[judged["di_ratio"].idxmin()]
        st.caption(
            f"Lowest disparate-impact ratio: {worst['group']} at {ratio(worst['di_ratio'])} "
            f"({'below' if worst['di_ratio'] < DI_THRESHOLD else 'above'} the {ratio(DI_THRESHOLD)} threshold). "
            f"Largest gap between mean PD and actual default rate: {pct(judged['pd_gap'].abs().max())}."
        )
judged = fairness[fairness["judged"]]
below = judged[judged["di_ratio"] < DI_THRESHOLD]
four_fifths = (
    "No group falls below the four-fifths rule"
    if below.empty
    else f"{count(len(below), 'group falls', 'groups fall')} below the four-fifths rule "
    f"({', '.join(below['group'])})"
)
st.info(
    f"**What this shows.** {four_fifths}, and the largest gap between mean PD and the actual default rate in any "
    f"group is {pct(judged['pd_gap'].abs().max())}, so the PDs are accurate within every group. The approval gaps "
    "follow real differences in default rates, and when default rates differ a model can't be both accurate "
    "within every group and equal on error rates. Even so, creditworthy younger applicants are declined more "
    "often than older ones. Most of that comes through the external credit scores, which are correlated with age "
    "(a longer credit history scores better), then loan length, job tenure and time since the ID was renewed. "
    "Whether to accept, reduce or offset it is a policy decision, discussed in the report."
)
