"""Loan default screener: the screening page (Streamlit entry point).

Run locally:  streamlit run app.py

An analyst enters one applicant (or loads an invented example) and sees the calibrated PD, the
approve / refer / decline decision, the main reasons (SHAP) and, for declines and refers, a draft
letter. All logic lives in src/; this file only lays out the page.

Performance: Streamlit reruns this whole script on every click, so the model, its SHAP explainer
and the example applicants are cached (st.cache_resource / st.cache_data) and loaded once per
server instead of on every rerun. Timings are in reports/performance.md.
"""

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure  # not pyplot: Streamlit serves each session in its own thread

from src.app_logging import get_logger
from src.applicant import FIELDS, NOT_EMPLOYED, friendly_error, to_form_values, to_model_inputs
from src.audit import log_assessment, log_letter
from src.config import CURRENCY_NOTE, DATA_DIR, LLM_BACKEND, OLLAMA_MODEL
from src.explain import contributions_by_reason
from src.formatting import money, pct, score
from src.labels import decision_label, model_label
from src.llm import build_prompt, draft_letter
from src.predict import load_artifacts, predict
from src.validation import validate_applicant

SAMPLE_PATH = DATA_DIR / "sample" / "sample_applicants.csv"
log = get_logger("app.screening")
BADGE_COLOURS = {"approve": "green", "refer": "orange", "decline": "red"}
INT_FIELDS = {
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OWN_CAR_AGE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "REGION_RATING_CLIENT_W_CITY",
}

st.set_page_config(page_title="Loan default screener", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def get_artifacts() -> dict:
    """Model, metadata, training ranges and SHAP explainer: loaded once per server, not on every rerun."""
    return load_artifacts()


def show_letter_agent(letter: dict, result: dict) -> None:
    """Show how the letter was made: the fixed input the language model gets and the guardrail checks.
    This is the AI part of the letter made visible, so an analyst (or a reviewer) can see that no
    applicant data reaches the model and why a model letter was accepted or replaced."""
    prompt = letter.get("prompt") or build_prompt(result["decision"], result["reasons"])
    with st.expander("What the language model sees: the Letter Agent's fixed input"):
        if letter.get("prompt") is None:
            st.caption(
                "The standard template wrote this letter, so nothing was sent to a model. This is what a model would get."
            )
        st.markdown(
            "Everything above **INPUT** is the same for every applicant: the agent's role, task, rules and "
            "output format. The **INPUT** holds only the decision and reason phrases from a fixed list. "
            "No PD, amounts, name or personal details are sent, so no applicant data reaches the model "
            "provider, and nothing typed into the form can change the model's instructions."
        )
        st.code(prompt, language=None, wrap_lines=True)
    with st.expander("Guardrail checks on the model's letter"):
        checks = letter.get("checks")
        if not checks:
            st.write(
                "No model letter to check: the standard template wording was used. It is fixed text that "
                "compliance can sign off once."
            )
            return
        passed = sum(c["passed"] for c in checks)
        outcome = (
            "so the model's letter is shown."
            if passed == len(checks)
            else "so the standard letter is shown instead."
        )
        st.markdown(f"**Passed {passed} of {len(checks)} checks**, {outcome}")
        st.markdown(
            "\n".join(
                f"- {'✅' if c['passed'] else '❌'} {c['check']}"
                + ("" if c["passed"] else f": {c['problem']}")
                for c in checks
            )
        )
        st.caption(
            "Plain rules, not another language model: they behave the same way every time and are tested."
        )


@st.cache_data
def get_examples() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def load_example() -> None:
    """Copy the chosen example applicant into the form fields."""
    row = get_examples().set_index("example").loc[st.session_state["example"]].to_dict()
    for field, value in to_form_values(row).items():
        value = value.item() if isinstance(value, np.generic) else value
        if value is not None and field != NOT_EMPLOYED:
            value = int(value) if field in INT_FIELDS else value
        st.session_state[f"f_{field}"] = value
    for key in ("result", "letter", "letter_text"):
        st.session_state.pop(key, None)


# ---- Widgets -----------------------------------------------------------------------------
def number(field: str, step: float = 1000.0, low: float = 0.0, high: float | None = None, fmt: str = "%.0f"):
    lbl, hlp = FIELDS[field]
    if field in INT_FIELDS:
        st.number_input(
            lbl,
            min_value=int(low),
            max_value=None if high is None else int(high),
            step=1,
            key=f"f_{field}",
            help=hlp,
        )
    else:
        st.number_input(lbl, min_value=low, max_value=high, step=step, format=fmt, key=f"f_{field}", help=hlp)


def choice(field: str, options: list, optional: bool = False):
    lbl, hlp = FIELDS[field]
    if optional:
        options = [None, *options]
    st.selectbox(
        lbl,
        options,
        key=f"f_{field}",
        help=hlp,
        format_func=lambda v: "Unknown / not given" if v is None else str(v),
    )


def yes_no(field: str):
    lbl, hlp = FIELDS[field]
    st.radio(
        lbl,
        ["Y", "N"],
        key=f"f_{field}",
        help=hlp,
        horizontal=True,
        format_func=lambda v: "Yes" if v == "Y" else "No",
    )


def categories(field: str) -> list[str]:
    """Categories seen in training, most common first (validation rejects anything else)."""
    shares = artifacts["reference"]["categorical"][field]
    return [c for c in sorted(shares, key=shares.get, reverse=True) if c != "Unknown"]


# ---- Page ----------------------------------------------------------------------------------
try:
    artifacts = get_artifacts()
except (FileNotFoundError, ValueError):
    log.exception("Model files couldn't be loaded")
    st.error(
        "The model couldn't be loaded, so applicants can't be scored right now. Try reloading the page; if it keeps happening, the app's maintainers can see the details in its log."
    )
    st.stop()
meta = artifacts["metadata"]
threshold, band = meta["threshold"], meta["refer_band"]
assumptions = meta["assumptions"]

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Load an example applicant, or enter one yourself.\n"
        "2. Press **Assess applicant**.\n"
        "3. Read the probability of default (PD), the decision and the reasons.\n"
        "4. For a decline or refer, draft the letter, **check and edit it**, then download it."
    )
    st.header("Scope")
    st.info(
        "A **second-stage screen** for applicants who already pass the lender's basic policy checks. "
        "The model learnt from loans Home Credit had already approved, so it has never seen the applicants "
        "those checks turned away. Unusual inputs are referred to an analyst."
    )
    st.header("Model")
    st.markdown(
        f"- {model_label(meta['chosen_model'])}, calibrated ({meta['calibration']['method']})\n"
        f"- Test AUC {score(meta['test_metrics']['auc'])}\n"
        f"- Decline above a PD of {pct(threshold + band)}, approve below {pct(threshold - band)}, "
        "refer in between\n"
        f"- Break-even PD {pct(assumptions['break_even_pd'])} (loss given default, LGD, "
        f"{pct(assumptions['lgd'])}; margin {pct(assumptions['margin'])})\n"
        f"- Trained {meta['trained_at'][:10]}"
    )
    st.caption(
        "Data: Home Credit Default Risk (Kaggle, 2018). The example applicants are invented, not real people."
    )

st.title("Loan default screener")
st.caption(
    "Probability of default, a credit decision and the reasons behind it, for one consumer loan applicant."
)

examples = get_examples()["example"].tolist()
if "f_AMT_CREDIT" not in st.session_state:  # first visit: start from the first example
    st.session_state["example"] = examples[0]
    load_example()
st.selectbox("Load an example applicant (invented)", examples, key="example", on_change=load_example)

with st.form("applicant"):
    loan, income, credit, household = st.columns(4)
    with loan:
        st.subheader("Loan")
        choice("NAME_CONTRACT_TYPE", categories("NAME_CONTRACT_TYPE"))
        number("AMT_CREDIT")
        number("AMT_ANNUITY", step=100.0)
        number("AMT_GOODS_PRICE")
    with income:
        st.subheader("Income and employment")
        number("AMT_INCOME_TOTAL")
        choice("NAME_INCOME_TYPE", categories("NAME_INCOME_TYPE"))
        st.checkbox(FIELDS[NOT_EMPLOYED][0], key=f"f_{NOT_EMPLOYED}", help=FIELDS[NOT_EMPLOYED][1])
        number("YEARS_EMPLOYED", step=0.5, high=70.0, fmt="%.1f")
        choice("OCCUPATION_TYPE", categories("OCCUPATION_TYPE"), optional=True)
        choice("ORGANIZATION_TYPE", categories("ORGANIZATION_TYPE"))
        choice("NAME_EDUCATION_TYPE", categories("NAME_EDUCATION_TYPE"))
    with credit:
        st.subheader("Credit history")
        for field in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
            number(field, step=0.01, high=1.0, fmt="%.2f")
        number("AMT_REQ_CREDIT_BUREAU_YEAR", high=100)
    with household:
        st.subheader("Household and stability")
        number("CNT_CHILDREN", high=20)
        number("CNT_FAM_MEMBERS", low=1, high=25)
        choice("NAME_HOUSING_TYPE", categories("NAME_HOUSING_TYPE"))
        yes_no("FLAG_OWN_REALTY")
        yes_no("FLAG_OWN_CAR")
        number("OWN_CAR_AGE", high=100)
        choice("REGION_RATING_CLIENT_W_CITY", [1, 2, 3])
        number("YEARS_SINCE_REGISTRATION", step=0.5, high=80.0, fmt="%.1f")
        number("YEARS_SINCE_ID", step=0.5, high=80.0, fmt="%.1f")
    submitted = st.form_submit_button("Assess applicant", type="primary")

if submitted:
    for key in ("result", "letter", "letter_text"):
        st.session_state.pop(key, None)
    applicant = to_model_inputs({f: st.session_state.get(f"f_{f}") for f in FIELDS})
    errors = validate_applicant(applicant, artifacts["reference"])
    if errors:
        st.error(
            "Please fix these inputs before the applicant can be scored:\n\n"
            + "\n".join(f"- {friendly_error(e)}" for e in errors)
        )
    else:
        try:
            with st.spinner("Scoring the applicant..."):
                st.session_state["result"] = predict(applicant, artifacts)
                st.session_state["assessment_id"] = log_assessment(
                    applicant, st.session_state["result"], meta
                )
                st.session_state["ead"] = applicant["AMT_CREDIT"]
        except Exception:  # show any scoring failure instead of crashing the page
            log.exception("Scoring one applicant failed")
            st.error(
                "Something went wrong while scoring this applicant, so no decision was made. Try reloading the page; if it keeps happening, the app's maintainers can see the details in its log."
            )

if "result" in st.session_state:
    result = st.session_state["result"]
    ead = st.session_state["ead"]
    expected_loss = result["pd"] * assumptions["lgd"] * ead
    expected_profit = (1 - result["pd"]) * assumptions["margin"] * ead - expected_loss

    st.divider()
    st.subheader("Assessment")
    pd_col, decision_col, el_col, profit_col = st.columns(4)
    pd_col.metric(
        "Probability of default",
        pct(result["pd"]),
        help="Calibrated: of 100 applicants with this PD, about this many have payment difficulties.",
    )
    with decision_col:
        st.markdown("**Decision**")
        st.badge(decision_label(result["decision"]), color=BADGE_COLOURS[result["decision"]])
    el_col.metric(
        "Expected loss",
        money(expected_loss),
        help="PD × LGD × EAD: loss given default × exposure at default (the loan amount).",
    )
    profit_col.metric(
        "Expected profit",
        money(expected_profit),
        help="(1 − PD) × margin × EAD − expected loss. Negative above the break-even PD.",
    )

    if result["decision"] == "approve":
        st.success("Expected to be profitable at the lender's assumptions: approve.")
    elif result["decision"] == "decline":
        st.error(f"PD is above {pct(threshold + band)}: the expected loss outweighs the margin, so decline.")
    for warning in result["warnings"]:
        st.warning(
            f"Referred for review: {warning}. The model learnt only from past approved loans, so it can't be "
            "relied on for applicants unlike them."
        )
    if result["decision"] == "refer" and not result["warnings"]:
        st.warning(
            f"PD is within {pct(band)} of the {pct(threshold)} cut-off: too close to call, so an analyst decides."
        )
    st.caption(CURRENCY_NOTE)

    reasons_col, chart_col = st.columns([1, 2])
    with reasons_col:
        st.markdown("**Main reasons**" if result["decision"] != "approve" else "**Main risk factors**")
        if result["reasons"]:
            st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(result["reasons"], 1)))
        else:
            st.write("No factor raised this applicant's risk above average.")
        st.caption(
            "Factors that raised the risk most (SHAP). They name what mattered, not whether it was high or low."
        )
    with chart_col:
        by_reason = contributions_by_reason(result["shap"]).head(8).iloc[::-1]
        fig = Figure(figsize=(7, 3.4))
        ax = fig.subplots()
        ax.barh(
            by_reason.index, by_reason.values, color=["#c0392b" if v > 0 else "#2471a3" for v in by_reason]
        )
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Effect on the risk score (log-odds): right = riskier, left = safer")
        ax.set_title("What moved this applicant's score", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, clear_figure=True)

    if result["decision"] in ("decline", "refer"):
        st.divider()
        st.subheader("Letter to the applicant")
        writers = {
            "ollama": f"the local language model ({OLLAMA_MODEL})",
            "gemini": "Google Gemini",
            "template": "the standard template",
        }
        writer = writers.get(LLM_BACKEND, writers["template"])
        if st.button("Draft letter"):
            wait = " This can take up to a minute." if LLM_BACKEND == "ollama" else ""
            with st.spinner(f"Drafting the letter with {writer}...{wait}"):
                letter = draft_letter(result["decision"], result["reasons"])
            log_letter(st.session_state.get("assessment_id"), letter["backend"])
            st.session_state["letter"] = letter
            st.session_state["letter_text"] = letter["text"]
        if "letter" in st.session_state:
            letter = st.session_state["letter"]
            if letter["warning"]:
                st.warning(letter["warning"])
            st.caption(
                f"Written by {writers[letter['backend']]}. This is a draft: check it and edit it before sending."
            )
            st.text_area("Draft letter", key="letter_text", height=380)
            st.download_button(
                "Download letter (.txt)", st.session_state["letter_text"], file_name="letter.txt"
            )
            show_letter_agent(letter, result)
