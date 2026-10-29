"""Batch scoring page: upload a CSV of applicants, score every valid row, download the results.

Uses the same validation, out-of-range guard, model and reasons as the screening page
(src/validation.py, src/predict.py), so a batch is judged exactly like single applicants. Rows
with input errors are listed and left out; the rest are still scored.
"""

import time

import pandas as pd
import streamlit as st

from src.applicant import label
from src.config import DATA_DIR, MAX_BATCH_ROWS, MAX_UPLOAD_MB, RAW_INPUTS
from src.predict import load_artifacts, predict_batch
from src.validation import read_csv_upload, validate_batch

EXAMPLES_PATH = DATA_DIR / "sample" / "sample_applicants.csv"
SOURCES = {"Upload a CSV": "upload", "Use the 6 invented example applicants": "examples"}

st.set_page_config(page_title="Batch scoring", page_icon="🗂️", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def get_artifacts() -> dict:
    return load_artifacts()


@st.cache_data
def read_examples() -> pd.DataFrame:
    return pd.read_csv(EXAMPLES_PATH)


try:
    artifacts = get_artifacts()
except (FileNotFoundError, ValueError) as e:
    st.error(f"The model couldn't be loaded, so batches can't be scored right now. Details: {e}")
    st.stop()
meta = artifacts["metadata"]
lgd = meta["assumptions"]["lgd"]

st.title("Batch scoring")
st.caption(
    "Score many applicants at once: the same checks, model and decision rule as the screening page. "
    f"Up to {MAX_BATCH_ROWS:,} rows and {MAX_UPLOAD_MB} MB per file."
)

with st.expander("What the file needs"):
    st.markdown(
        "One row per applicant, with a header row containing these columns (extra columns, such as your own "
        "reference, are kept and returned):\n\n"
        + ", ".join(f"`{c}`" for c in RAW_INPUTS)
        + "\n\nOptional values (external scores, car age, occupation, goods price, bureau enquiries) can be left "
        "blank, but the columns must be there. Day counts are negative (days before the application), as in the "
        "Home Credit data; `DAYS_EMPLOYED = 365243` means not employed."
    )
    cols = st.columns(2)
    cols[0].download_button(
        "Download an empty template (CSV)", ",".join(RAW_INPUTS) + "\n", file_name="batch_template.csv"
    )
    cols[1].download_button(
        "Download the example file (CSV)",
        read_examples().to_csv(index=False),
        file_name="example_applicants.csv",
    )

source = SOURCES[st.radio("Applicants", list(SOURCES), horizontal=True)]
if source == "examples":
    batch = read_examples()
else:
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is None:
        st.info("Upload a file, or use the example applicants.")
        st.stop()
    try:
        batch = read_csv_upload(uploaded.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()

try:
    valid, errors = validate_batch(batch, artifacts["reference"])
except ValueError as e:  # whole-file problems: no rows, too many rows, missing columns
    st.error(f"This file can't be scored: {e}")
    st.stop()

if len(errors):
    rejected = errors["row"].nunique()
    st.warning(f"{rejected:,} of {len(batch):,} row(s) have input errors and were not scored. The rest were.")
    with st.expander(f"Rows with errors ({len(errors):,} problem(s))", expanded=rejected <= 20):
        shown = errors.assign(
            csv_line=errors["row"] + 2,  # +1 for the header, +1 because CSV lines count from 1
            field_label=errors["field"].map(label),
        )[["csv_line", "field", "field_label", "problem"]]
        st.dataframe(
            shown.rename(
                columns={
                    "csv_line": "Line in file",
                    "field": "Column",
                    "field_label": "Field",
                    "problem": "Problem",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download the error list (CSV)", shown.to_csv(index=False), file_name="batch_errors.csv"
        )
if valid.empty:
    st.error("No row could be scored. Fix the errors listed above and upload the file again.")
    st.stop()

try:
    start = time.perf_counter()
    with st.spinner(f"Scoring {len(valid):,} applicant(s)..."):
        scored = predict_batch(valid, artifacts, with_reasons=True)
    seconds = time.perf_counter() - start
except Exception as e:  # noqa: BLE001 (show any scoring failure instead of crashing the page)
    st.error(f"Something went wrong while scoring this file: {e}")
    st.stop()
scored["expected_loss"] = scored["pd"] * lgd * scored["AMT_CREDIT"]

st.success(f"Scored {len(scored):,} applicant(s) in {seconds:.1f} s.")
shares = scored["decision"].value_counts(normalize=True)
cols = st.columns(5)
cols[0].metric("Approve", f"{shares.get('approve', 0):.1%}")
cols[1].metric("Refer", f"{shares.get('refer', 0):.1%}")
cols[2].metric("Decline", f"{shares.get('decline', 0):.1%}")
cols[3].metric(
    "Mean PD", f"{scored['pd'].mean():.1%}", help=f"Test split: {meta['test_metrics']['mean_pd']:.1%}"
)
cols[4].metric(
    "Expected loss", f"{scored['expected_loss'].sum():,.0f}", help="Sum of PD × LGD × loan amount."
)
if (scored["warnings"] != "").any():
    st.warning(
        f"{(scored['warnings'] != '').sum():,} applicant(s) had an input outside the range of past approved "
        "applicants and were referred to an analyst instead of approved (see the warnings column)."
    )

st.subheader("Results")
decisions = st.multiselect("Show", ["approve", "refer", "decline"], default=["approve", "refer", "decline"])
id_cols = [
    c
    for c in scored.columns
    if c not in RAW_INPUTS and c not in ("pd", "decision", "warnings", "reasons", "expected_loss")
]
view = scored[scored["decision"].isin(decisions)][
    [*id_cols, "AMT_CREDIT", "pd", "decision", "reasons", "warnings"]
]
st.dataframe(
    view.head(500),
    column_config={
        "AMT_CREDIT": st.column_config.NumberColumn("Loan amount", format="%,.0f"),
        "pd": st.column_config.ProgressColumn("PD", format="percent", min_value=0, max_value=1),
        "decision": "Decision",
        "reasons": "Main reasons (refer / decline)",
        "warnings": "Warnings",
    },
    hide_index=True,
    width="stretch",
)
if len(view) > 500:
    st.caption(f"Showing the first 500 of {len(view):,} rows; the download has all of them.")
st.download_button(
    "Download all results (CSV)",
    scored.to_csv(index=False),
    file_name="batch_results.csv",
    type="primary",
    help="Your columns plus pd, decision, reasons, warnings and expected_loss.",
)
