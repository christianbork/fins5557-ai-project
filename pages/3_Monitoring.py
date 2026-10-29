"""Monitoring page: has a new batch of applicants drifted away from the training population?

Compares each input's distribution with training (PSI, src/monitoring.py) and shows how the
model's PDs and decisions for the batch compare with the test split. Works on the deployed app:
it needs only models/ and the synthetic sample batches, never the Kaggle data.
"""

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure  # not pyplot: Streamlit serves each session in its own thread
from matplotlib.ticker import FuncFormatter

from src.formatting import count, format_table, pct, pct_points, ratio, score
from src.labels import DAY_FIELDS, STATUSES, decision_label, headers, input_label, status_label
from src.monitoring import PSI_ALERT, PSI_STABLE, SAMPLE_PATHS, bin_detail, psi_table
from src.predict import load_artifacts, predict_batch
from src.validation import read_csv_upload, validate_batch

STATUS_COLOURS = {"stable": "#1e8449", "watch": "#d68910", "alert": "#c0392b"}
SOURCES = {
    "Sample: applicants like the training data (synthetic)": "normal",
    "Sample: a shifted batch, younger and thinner-file (synthetic)": "shifted",
    "Upload a CSV": "upload",
}

st.set_page_config(page_title="Monitoring", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def get_artifacts() -> dict:
    return load_artifacts()


@st.cache_data
def read_sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATHS[name])


try:
    artifacts = get_artifacts()
except (FileNotFoundError, ValueError) as e:
    st.error(f"The model couldn't be loaded, so monitoring isn't available right now. Details: {e}")
    st.stop()
reference, meta = artifacts["reference"], artifacts["metadata"]

st.title("Monitoring: has the applicant population changed?")
st.caption(
    "The model learnt from loans Home Credit had already approved. If the people applying change, for example "
    "after a marketing campaign, its PDs can be wrong even though the model itself hasn't changed. The population "
    f"stability index (PSI) compares each input of a new batch with training: below {ratio(PSI_STABLE)} stable, "
    f"{ratio(PSI_STABLE)} to {ratio(PSI_ALERT)} watch, above {ratio(PSI_ALERT)} alert (Code Bank W02-10, W05-07). "
    "PD is the probability of default."
)

source = SOURCES[st.radio("Batch to check", list(SOURCES), horizontal=True)]
if source == "upload":
    uploaded = st.file_uploader(
        "CSV with one row per applicant and the model's input columns (same format as the sample files)",
        type="csv",
    )
    if uploaded is None:
        st.info("Upload a file, or choose one of the sample batches above.")
        st.stop()
    try:
        batch = read_csv_upload(uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()
else:
    batch = read_sample(source)
    st.download_button(
        "Download this sample (CSV)", batch.to_csv(index=False), file_name=f"monitoring_{source}.csv"
    )

try:
    table = psi_table(batch, reference)
except ValueError as e:
    st.error(f"This file can't be checked: {e}. Optional columns can be left blank but must be there.")
    st.stop()

counts = table["status"].value_counts()
alerts, watch = counts.get("alert", 0), counts.get("watch", 0)
cols = st.columns(4)
cols[0].metric("Applicants in batch", f"{len(batch):,}")
for col, (code, name) in zip(cols[1:], reversed(STATUSES.items()), strict=True):
    col.metric(name, int(counts.get(code, 0)))
if alerts:
    shifted = ", ".join(table.loc[table["status"] == "alert", "input"].map(input_label))
    st.error(
        f"**{count(alerts, 'input has', 'inputs have')} shifted a lot**: {shifted}. The model's PDs may not be "
        "reliable for this batch. Review it, check calibration on recent outcomes, and consider retraining."
    )
elif watch:
    st.warning(
        f"{count(watch, 'input has', 'inputs have')} moved somewhat. No action needed yet, but keep an eye on them."
    )
else:
    st.success("All inputs look like the training data. The model can be used as validated.")

chart_col, table_col = st.columns([3, 2])
with chart_col:
    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    shown = table.iloc[::-1]
    ax.barh(
        shown["input"].map(input_label),
        shown["psi"].clip(lower=1e-4),
        color=shown["status"].map(STATUS_COLOURS),
    )
    ax.axvline(PSI_STABLE, color=STATUS_COLOURS["watch"], linestyle="--", linewidth=1)
    ax.axvline(PSI_ALERT, color=STATUS_COLOURS["alert"], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(
        f"PSI (log scale): dashed lines at {ratio(PSI_STABLE)} (watch) and {ratio(PSI_ALERT)} (alert)"
    )
    ax.set_title("Drift per input", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)
with table_col:
    shown = table.assign(input=table["input"].map(input_label), status=table["status"].map(status_label))
    shown = shown.set_index("input")[["psi", "status", "missing_train", "missing_batch"]]
    st.dataframe(
        headers(format_table(shown, {"psi": score, "missing_train": pct, "missing_batch": pct})),
        height=560,
        width="stretch",
    )

st.subheader("Where does the shift come from?")
field = st.selectbox(
    "Input",
    table["input"].tolist(),
    format_func=input_label,
    help="Inputs are ordered by PSI, largest first.",
)
detail = bin_detail(batch, reference, field)
fig = Figure(figsize=(9, 3.2))
ax = fig.subplots()
x = range(len(detail))
ax.bar([i - 0.2 for i in x], detail["training"], width=0.4, label="Training", color="#85929e")
ax.bar([i + 0.2 for i in x], detail["batch"], width=0.4, label="This batch", color="#2471a3")
ax.set_xticks(list(x), detail["bin"], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Share of applicants")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig, clear_figure=True)
if field in DAY_FIELDS:
    st.caption("Day counts are as Home Credit stores them: negative, counted back from the application date.")

st.subheader("What the model does with this batch")
valid, errors = validate_batch(batch, reference) if len(batch) else (batch, pd.DataFrame())
if errors is not None and len(errors):
    st.warning(
        f"{count(errors['row'].nunique(), 'row has', 'rows have')} input errors and "
        f"{'is' if errors['row'].nunique() == 1 else 'are'} left out of the scoring below."
    )
if len(valid):
    with st.spinner("Scoring the batch..."):
        scored = predict_batch(valid, artifacts)
    test = meta["test_metrics"]
    shares = scored["decision"].value_counts(normalize=True)
    cols = st.columns(4)
    cols[0].metric(
        "Mean PD",
        pct(scored["pd"].mean()),
        f"{pct_points(scored['pd'].mean() - test['mean_pd'])} vs test",
        delta_color="inverse",
    )
    for col, decision in zip(cols[1:], ("approve", "refer", "decline"), strict=True):
        share = shares.get(decision, 0.0)
        col.metric(
            decision_label(decision),
            pct(share),
            f"{pct_points(share - test['decision_shares'][decision])} vs test",
            delta_color="inverse" if decision == "decline" else "normal",
        )
    st.caption(
        f"Compared with the test split (mean PD {pct(test['mean_pd'])}). {pct((scored['warnings'] != '').mean())} of "
        "this batch had an input outside the training range and was referred by the guard. A higher mean PD on a "
        "shifted batch is the model reacting to riskier inputs; whether those PDs are still accurate can only be "
        "checked once the batch's repayment outcomes are known."
    )
