"""Monitoring page: has a new batch of applicants drifted away from the training population?

Compares each input's distribution with training (PSI, src/monitoring.py) and shows how the
model's PDs and decisions for the batch compare with the test split. Works on the deployed app:
it needs only models/ and the synthetic sample batches, never the Kaggle data.
"""

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure  # not pyplot: Streamlit serves each session in its own thread

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
    f"stability index (PSI) compares each input of a new batch with training: below {PSI_STABLE} stable, "
    f"{PSI_STABLE}-{PSI_ALERT} watch, above {PSI_ALERT} alert (Code Bank W02-10, W05-07)."
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
cols[1].metric("Alert", int(alerts))
cols[2].metric("Watch", int(watch))
cols[3].metric("Stable", int(counts.get("stable", 0)))
if alerts:
    shifted = ", ".join(table.loc[table["status"] == "alert", "input"])
    st.error(
        f"**{alerts} input(s) have shifted a lot**: {shifted}. The model's PDs may not be reliable for this "
        "batch. Review it, check calibration on recent outcomes, and consider retraining."
    )
elif watch:
    st.warning(f"{watch} input(s) have moved somewhat. No action needed yet, but keep an eye on them.")
else:
    st.success("All inputs look like the training data. The model can be used as validated.")

chart_col, table_col = st.columns([3, 2])
with chart_col:
    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    shown = table.iloc[::-1]
    ax.barh(shown["input"], shown["psi"].clip(lower=1e-4), color=shown["status"].map(STATUS_COLOURS))
    ax.axvline(PSI_STABLE, color=STATUS_COLOURS["watch"], linestyle="--", linewidth=1)
    ax.axvline(PSI_ALERT, color=STATUS_COLOURS["alert"], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("PSI (log scale): dashed lines at 0.10 (watch) and 0.25 (alert)")
    ax.set_title("Drift per input", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, clear_figure=True)
with table_col:
    st.dataframe(
        table.set_index("input")[["psi", "status", "missing_train", "missing_batch"]]
        .rename(
            columns={
                "psi": "PSI",
                "status": "Status",
                "missing_train": "Missing (training)",
                "missing_batch": "Missing (batch)",
            }
        )
        .style.format(
            {"PSI": "{:.3f}", "Missing (training)": "{:.1%}", "Missing (batch)": "{:.1%}"}, na_rep="–"
        ),
        height=560,
        width="stretch",
    )

st.subheader("Where does the shift come from?")
field = st.selectbox("Input", table["input"].tolist(), help="Inputs are ordered by PSI, largest first.")
detail = bin_detail(batch, reference, field)
fig = Figure(figsize=(9, 3.2))
ax = fig.subplots()
x = range(len(detail))
ax.bar([i - 0.2 for i in x], detail["training"], width=0.4, label="training", color="#85929e")
ax.bar([i + 0.2 for i in x], detail["batch"], width=0.4, label="this batch", color="#2471a3")
ax.set_xticks(list(x), detail["bin"], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Share of applicants")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig, clear_figure=True)

st.subheader("What the model does with this batch")
valid, errors = validate_batch(batch, reference) if len(batch) else (batch, pd.DataFrame())
if errors is not None and len(errors):
    st.warning(f"{errors['row'].nunique():,} row(s) have input errors and are left out of the scoring below.")
if len(valid):
    with st.spinner("Scoring the batch..."):
        scored = predict_batch(valid, artifacts)
    test = meta["test_metrics"]
    shares = scored["decision"].value_counts(normalize=True)
    cols = st.columns(4)
    cols[0].metric(
        "Mean PD",
        f"{scored['pd'].mean():.1%}",
        f"{scored['pd'].mean() - test['mean_pd']:+.1%} vs test",
        delta_color="inverse",
    )
    for col, decision in zip(cols[1:], ("approve", "refer", "decline"), strict=True):
        share = shares.get(decision, 0.0)
        col.metric(
            decision.title(),
            f"{share:.1%}",
            f"{share - test['decision_shares'][decision]:+.1%} vs test",
            delta_color="inverse" if decision == "decline" else "normal",
        )
    st.caption(
        f"Compared with the test split (mean PD {test['mean_pd']:.1%}). {(scored['warnings'] != '').mean():.1%} of "
        "this batch had an input outside the training range and was referred by the guard. A higher mean PD on a "
        "shifted batch is the model reacting to riskier inputs; whether those PDs are still accurate can only be "
        "checked once the batch's repayment outcomes are known."
    )
