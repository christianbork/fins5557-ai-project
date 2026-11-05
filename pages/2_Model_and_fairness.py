"""Station 4 (Implementation). Model and fairness page: how the model was chosen, how good its PDs are,
where the cut-off
comes from, and whether its decisions treat groups of applicants differently.

Reads only the summary files in models/ and reports/ (no applicant data), so it works on the
deployed app. Regenerate them with `python -m src.train` or `python -m src.reporting`.
"""

import json

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure  # not pyplot: Streamlit serves each session in its own thread
from matplotlib.ticker import FuncFormatter

from src.app_logging import get_logger
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
from src.ddf import DDF_STATIONS
from src.formatting import count, format_table, money, money_short, number, pct, pct_points, ratio, score
from src.labels import RULES, flag_text, headers, model_choice_reason, model_label

V1_METADATA = ROOT / "archive" / "v1-home-credit-baseline" / "metadata.json"
log = get_logger("app.model")


def count_plain(value) -> str:
    """A whole number with thousands separators (a cost or a count of groups)."""
    return f"{value:,.0f}"


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
except FileNotFoundError:
    log.exception("A report file is missing (regenerate with python -m src.reporting)")
    st.error(
        "A summary file behind this page is missing, so it can't be shown right now. Try reloading the page; if it keeps happening, the app's maintainers can see the details in its log."
    )
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
rule = meta.get("decision_rule", "profit")
st.write(
    "The model gives a PD; the cut-off decides who is approved. Where it sits depends on how the lender weighs "
    "**approving someone who defaults** against **declining someone who would have repaid**. The same model "
    "gives very different decisions under three sourced answers to that question:"
)
st.markdown(
    f"- **{RULES['profit']}**: the cut-off that earned the most on the validation split when a repaid loan earns "
    f"the margin ({pct(a['margin'])}) and a default loses the loss given default (LGD, {pct(a['lgd'])}); "
    f"break-even PD = margin / (margin + LGD) = {pct(a['break_even_pd'])}, a 4 : 1 cost ratio.\n"
    f"- **{RULES['cost_ratio']}**: approving a defaulter counts as {meta.get('cost_ratio', 5):g} times as costly as "
    "declining a good customer, the cost matrix of the German credit data. Approve while "
    f"PD < 1 / (1 + {meta.get('cost_ratio', 5):g}) = {pct(cutoffs.loc['cost_ratio', 'cutoff'])}.\n"
    f"- **{RULES['balanced_accuracy']}**: the cut-off that maximises (TPR + TNR) / 2 on the validation split, "
    "the method taught in Big Data Analytics at CBS. With 8% defaulters it implies about 11 : 1."
)
st.info(
    f"**The app uses {RULES[rule]}**: cut-off {pct(meta['threshold'])}, with applicants within "
    f"{pct_points(meta['refer_band']).lstrip('+')} of it referred to an analyst. It is more cautious than the profit rule at a small "
    "cost in profit, and it also weighs the borrower's harm from a default, not only the lender's loss."
)
fig = Figure(figsize=(9, 3.6))
ax = fig.subplots()
profit = r["profit"]
ax.plot(profit["threshold"], profit["realised_profit"], color="#1f618d")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: money_short(v)))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v)))
styles = {"profit": ("#1f618d", "--"), "cost_ratio": ("#1e8449", "-"), "balanced_accuracy": ("#b9770e", "-.")}
for code, (colour, line) in styles.items():
    cut = cutoffs.loc[code, "cutoff"]
    used = " (used by the app)" if code == rule else ""
    ax.axvline(
        cut,
        color=colour,
        linestyle=line,
        linewidth=2 if used else 1.2,
        label=f"{RULES[code]}: {pct(cut)}{used}",
    )
ax.set(
    xlabel="Cut-off: decline when PD is at or above",
    ylabel="Realised profit",
    xlim=(0, 0.5),
    title="Profit on the validation split at each cut-off",
)
ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)

columns = [
    "cutoff",
    "balanced_accuracy",
    "tpr",
    "good_auto_approved",
    "defaulters_auto_approved",
    "approve_share",
    "refer_share",
    "decline_share",
    "default_rate_approved",
    "realised_profit",
    "cost_at_ratio",
    "lowest_di_ratio",
    "groups_below_four_fifths",
]
# One column per rule, one row per measure: three columns fit the page, thirteen don't
shown = headers(cutoffs[[c for c in columns if c in cutoffs]].rename(index=RULES)).T.rename_axis("Measure")
counts = {"Cost at 5:1", "Groups below four-fifths"}
ratios = {"Balanced accuracy", "Lowest disparate-impact ratio"}
styler = shown.style
for measure in shown.index:
    fmt = (
        money
        if measure == "Profit"
        else count_plain
        if measure in counts
        else ratio
        if measure in ratios
        else pct
    )
    styler = styler.format(fmt, subset=pd.IndexSlice[measure, :])
st.dataframe(styler, width="stretch")
st.caption(
    "Test split. Defaulters caught is the share of eventual defaulters at or above the cut-off; the approve, refer "
    "and decline shares and the automatic approvals include the refer band. Cost at 5:1 counts 5 per approved "
    "defaulter and 1 per declined good customer. The stricter the rule, the more defaulters it stops, but the more "
    "good customers, and especially younger applicants, it turns away (lowest disparate-impact ratio). "
    + CURRENCY_NOTE
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
    "Whether to accept, reduce or offset it is a policy decision, discussed in the report. The stricter the decision "
    "rule, the wider the gap: section 2 shows the lowest ratio under each rule, and borderline applicants are referred "
    "to an analyst rather than declined automatically."
)

# ---- How it is built: the Digital Data Factory ---------------------------------------------
st.header("4. How the system is built: the Digital Data Factory")
st.caption(
    "The course's DDF framework: four stations, each accountable for its own checks. An error at an early "
    "station can't be fixed later, so the checks start with the data, not the model."
)
for station in DDF_STATIONS:
    with st.expander(f"Station {station['station']}: {station['name']} ({station['role']})"):
        st.markdown(f"**What we do.** {station['what_we_do']}")
        st.markdown(f"**Checks owned here.** {station['checks']}")
        st.caption("Code: " + ", ".join(f"`{m}`" for m in station["modules"]))
