"""Batch scoring page: upload a CSV of applicants, score every valid row, download the results.

Uses the same validation, out-of-range guard, model and reasons as the screening page
(src/validation.py, src/predict.py), so a batch is judged exactly like single applicants. Rows
with input errors are listed and left out; the rest are still scored.
"""

import time

import pandas as pd
import streamlit as st

from src.applicant import label
from src.config import CURRENCY_NOTE, DATA_DIR, MAX_BATCH_ROWS, MAX_UPLOAD_MB, RAW_INPUTS
from src.formatting import MONEY_COLUMN, PCT_COLUMN, count, money, pct
from src.labels import COLUMNS, DECISIONS, column_label, decision_label
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
    f"Up to {count(MAX_BATCH_ROWS, 'row')} and {MAX_UPLOAD_MB} MB per file."
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
    st.warning(
        f"{rejected:,} of {count(len(batch), 'row has', 'rows have')} input errors and were not scored. "
        "The rest were."
    )
    with st.expander(f"Rows with errors ({count(len(errors), 'problem')})", expanded=rejected <= 20):
        shown = errors.assign(
            csv_line=errors["row"] + 2,  # +1 for the header, +1 because CSV lines count from 1
            field_label=errors["field"].map(label),
        )[["csv_line", "field", "field_label", "problem"]].rename(columns=COLUMNS)
        st.dataframe(shown, hide_index=True, width="stretch")
        st.download_button(
            "Download the error list (CSV)", shown.to_csv(index=False), file_name="batch_errors.csv"
        )
if valid.empty:
    st.error("No row could be scored. Fix the errors listed above and upload the file again.")
    st.stop()

try:
    start = time.perf_counter()
    with st.spinner(f"Scoring {count(len(valid), 'applicant')}..."):
        scored = predict_batch(valid, artifacts, with_reasons=True)
    seconds = time.perf_counter() - start
except Exception as e:  # noqa: BLE001 (show any scoring failure instead of crashing the page)
    st.error(f"Something went wrong while scoring this file: {e}")
    st.stop()
scored["expected_loss"] = scored["pd"] * lgd * scored["AMT_CREDIT"]

st.success(f"Scored {count(len(scored), 'applicant')} in {seconds:.1f} s.")
shares = scored["decision"].value_counts(normalize=True)
cols = st.columns(5)
for col, (code, name) in zip(cols, DECISIONS.items(), strict=False):
    col.metric(name, pct(shares.get(code, 0)))
cols[3].metric(
    "Mean PD",
    pct(scored["pd"].mean()),
    help=f"Probability of default, averaged over the batch. Test split: {pct(meta['test_metrics']['mean_pd'])}.",
)
cols[4].metric(
    "Expected loss",
    money(scored["expected_loss"].sum()),
    help="Sum of PD × LGD × loan amount (LGD = loss given default).",
)
st.caption(CURRENCY_NOTE)
guarded = (scored["warnings"] != "").sum()
if guarded:
    st.warning(
        f"{count(guarded, 'applicant')} had an input outside the range of past approved applicants and "
        f"{'was' if guarded == 1 else 'were'} referred to an analyst instead of approved (see the Warnings column)."
    )

st.subheader("Results")
decisions = st.multiselect("Show", list(DECISIONS), default=list(DECISIONS), format_func=decision_label)
id_cols = [
    c
    for c in scored.columns
    if c not in RAW_INPUTS and c not in ("pd", "decision", "warnings", "reasons", "expected_loss")
]
view = scored[scored["decision"].isin(decisions)][
    [*id_cols, "AMT_CREDIT", "pd", "decision", "reasons", "warnings"]
].assign(pd=lambda d: d["pd"] * 100, decision=lambda d: d["decision"].map(decision_label))
st.dataframe(
    view.head(500),
    column_config={c: column_label(c) for c in view.columns}
    | {
        "AMT_CREDIT": st.column_config.NumberColumn("Loan amount", format=MONEY_COLUMN),
        "pd": st.column_config.ProgressColumn("PD", format=PCT_COLUMN, min_value=0, max_value=100),
    },
    hide_index=True,
    width="stretch",
)
if len(view) > 500:
    st.caption(f"Showing the first 500 of {len(view):,} rows; the download has all of them.")
# The input columns keep their codes, so the file can be uploaded again; the added columns get headers.
added = ["pd", "decision", "reasons", "warnings", "expected_loss"]
results = scored.assign(decision=scored["decision"].map(decision_label)).rename(
    columns={c: COLUMNS[c] for c in added}
)
st.download_button(
    "Download all results (CSV)",
    results.round(4).to_csv(index=False),
    file_name="batch_results.csv",
    type="primary",
    help="Your columns plus PD, decision, reasons, warnings and expected loss (USD display assumption).",
)
