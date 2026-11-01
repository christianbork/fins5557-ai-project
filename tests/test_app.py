"""Smoke tests for the screening page (app.py) with Streamlit's AppTest: no browser or server needed.

They use the committed model in models/ and the invented examples in data/sample/.
"""

import json
import re

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import llm
from src.config import DATA_DIR, ROOT

EXAMPLES = pd.read_csv(DATA_DIR / "sample" / "sample_applicants.csv")["example"].tolist()


@pytest.fixture
def app(monkeypatch):
    # Whatever LLM_BACKEND is set locally, never call a real model from the tests
    monkeypatch.setattr(llm, "_call_ollama", lambda prompt: (_ for _ in ()).throw(TimeoutError()))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()


def assess(app: AppTest, example: str) -> AppTest:
    app.selectbox(key="example").set_value(example).run()
    return app.button(key="FormSubmitter:applicant-Assess applicant").click().run()


def test_page_loads_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "Loan default screener"


@pytest.mark.parametrize(
    ("example", "badge"),
    [
        (EXAMPLES[0], ":green-badge[Approve]"),
        (EXAMPLES[1], ":orange-badge[Refer]"),
        (EXAMPLES[2], ":red-badge[Decline]"),
        (EXAMPLES[5], ":orange-badge[Refer]"),
    ],
)
def test_examples_get_their_decision(app, example, badge):
    assess(app, example)
    assert not app.exception
    assert badge in [m.value for m in app.markdown]


def test_unusual_applicant_shows_the_guard_warning(app):
    assess(app, EXAMPLES[5])
    assert any("past approved applicants" in w.value for w in app.warning)


def test_decline_letter_is_a_draft_with_fallback_note(app):
    assess(app, EXAMPLES[2])
    next(b for b in app.button if b.label == "Draft letter").click().run()
    assert not app.exception
    letter = app.text_area(key="letter_text").value
    assert letter.startswith("Dear Applicant,")
    assert any("draft" in c.value.lower() for c in app.caption)


def test_invalid_input_shows_an_error_not_a_crash(app):
    app.number_input(key="f_AMT_CREDIT").set_value(0.0)  # form values only apply on submit
    app.button(key="FormSubmitter:applicant-Assess applicant").click().run()
    assert not app.exception
    assert any("Loan amount ($): must be at least 1" in e.value for e in app.error)
    assert not app.metric  # nothing scored


def test_scoring_failure_shows_a_short_message_and_logs_the_detail(app, monkeypatch, caplog):
    import src.predict

    monkeypatch.setattr(
        src.predict, "predict", lambda *a: (_ for _ in ()).throw(RuntimeError("internal detail"))
    )
    with caplog.at_level("ERROR", logger="app.screening"):
        app.button(key="FormSubmitter:applicant-Assess applicant").click().run()
    assert not app.exception
    assert "Something went wrong while scoring this applicant" in app.error[0].value
    assert "internal detail" not in app.error[0].value
    assert "internal detail" in caplog.text  # the traceback is in the log


@pytest.mark.parametrize("page", ["app.py", "pages/1_Batch_scoring.py", "pages/3_Monitoring.py"])
def test_missing_model_shows_a_short_message_without_the_path(monkeypatch, page):
    import src.predict

    def missing(*args):
        raise FileNotFoundError("Model files not found: ['C:/secret/models/model.joblib']")

    monkeypatch.setattr(src.predict, "load_artifacts", missing)
    st.cache_resource.clear()  # earlier tests cached the real model
    try:
        at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    finally:
        st.cache_resource.clear()
    assert not at.exception
    assert "The model couldn't be loaded" in at.error[0].value
    assert "secret" not in at.error[0].value


def test_model_and_fairness_page_loads():
    page = AppTest.from_file(str(ROOT / "pages" / "2_Model_and_fairness.py"), default_timeout=120).run()
    assert not page.exception and not page.error
    assert [t.label for t in page.tabs] == ["Sex", "Age band", "Family status", "Education"]
    assert page.metric[0].value == "XGBoost"


def monitoring_page() -> AppTest:
    return AppTest.from_file(str(ROOT / "pages" / "3_Monitoring.py"), default_timeout=120).run()


def test_monitoring_page_normal_sample_is_stable():
    page = monitoring_page()
    assert not page.exception and not page.error
    assert page.success and "look like the training data" in page.success[0].value


def test_monitoring_page_shifted_sample_alerts():
    page = monitoring_page()
    page.radio[0].set_value(page.radio[0].options[1]).run()
    assert not page.exception
    assert page.error and "shifted a lot" in page.error[0].value
    assert {m.label: m.value for m in page.metric}["Alert"] != "0"


def test_monitoring_page_asks_for_an_upload():
    page = monitoring_page()
    page.radio[0].set_value(page.radio[0].options[2]).run()
    assert not page.exception and "Upload a file" in page.info[0].value


def batch_page() -> AppTest:
    return AppTest.from_file(str(ROOT / "pages" / "1_Batch_scoring.py"), default_timeout=120).run()


def use_examples(page: AppTest) -> AppTest:
    return page.radio[0].set_value(page.radio[0].options[1]).run()


def test_batch_page_asks_for_an_upload():
    page = batch_page()
    assert not page.exception and "Upload a file" in page.info[0].value


def test_batch_page_scores_the_examples():
    page = use_examples(batch_page())
    assert not page.exception and not page.error
    assert "Scored 6 applicants" in page.success[0].value
    results = page.dataframe[0].value
    assert list(results["Decision"]) == ["Approve", "Refer", "Decline", "Approve", "Approve", "Refer"]
    assert (results.loc[results["Decision"] != "Approve", "Main reasons (refer / decline)"] != "").all()


def test_batch_page_lists_bad_rows_and_scores_the_rest(tmp_path, monkeypatch):
    examples = pd.read_csv(DATA_DIR / "sample" / "sample_applicants.csv")
    bad = examples.iloc[[0]].assign(example="Bad row", EXT_SOURCE_2=1.5)
    (tmp_path / "sample").mkdir()
    pd.concat([examples, bad]).to_csv(tmp_path / "sample" / "sample_applicants.csv", index=False)
    monkeypatch.setattr("src.config.DATA_DIR", tmp_path)
    st.cache_data.clear()  # earlier tests cached the real example file
    page = use_examples(batch_page())
    st.cache_data.clear()
    assert not page.exception
    assert "1 of 7 rows have input errors" in page.warning[0].value
    assert "Scored 6 applicants" in page.success[0].value
    error_table = page.dataframe[0].value
    assert error_table.iloc[0]["Column"] == "EXT_SOURCE_2" and error_table.iloc[0]["Line in file"] == 8


# ---- What users see: display labels and formats on every page (Step 13b) -------------------
RAW_CODE = re.compile(r"_|^[a-z]")  # a code such as AMT_CREDIT, approval_rate or pd
RAW_DECISION = re.compile(r"\b(approve|refer|decline)\b")
EMPTY_TEXT = re.compile(r"\b(None|nan|NaN)\b")
TEXT_ELEMENTS = ("markdown", "caption", "info", "warning", "error", "success", "subheader", "header", "title")


def every_page_in_use(app) -> list[AppTest]:
    """Each page with its results showing: a declined applicant with a letter, the example batch,
    the model page and the shifted monitoring batch."""
    assess(app, EXAMPLES[2])
    next(b for b in app.button if b.label == "Draft letter").click().run()
    monitoring = monitoring_page()
    monitoring.radio[0].set_value(monitoring.radio[0].options[1]).run()
    model = AppTest.from_file(str(ROOT / "pages" / "2_Model_and_fairness.py"), default_timeout=120).run()
    return [app, use_examples(batch_page()), model, monitoring]


def test_tables_show_display_labels_and_no_empty_markers(app):
    for page in every_page_in_use(app):
        assert not page.exception
        for table in page.dataframe:
            df = table.value
            names = [str(c) for c in df.columns] + [str(n) for n in df.index.names if n is not None]
            assert not [n for n in names if RAW_CODE.search(n)], f"raw column names: {names}"
            cells = df.astype(str).to_numpy().ravel()
            assert not [c for c in cells if EMPTY_TEXT.fullmatch(c)], f"empty markers in {names}"
            assert not df.isna().to_numpy().any(), f"empty cells (shown as None) in {names}"
            if "Decision" in df.columns:
                assert set(df["Decision"]) <= {"Approve", "Refer", "Decline"}


def test_text_has_no_empty_markers_or_lower_case_decision_labels(app):
    for page in every_page_in_use(app):
        texts = [e.value for kind in TEXT_ELEMENTS for e in getattr(page, kind)]
        texts += [f"{m.label} {m.value}" for m in page.metric]
        assert not [t for t in texts if EMPTY_TEXT.search(t)]
        labels = [m.label for m in page.metric] + [o for s in page.multiselect for o in s.options]
        assert not [t for t in labels if RAW_DECISION.search(t)]


def test_every_money_value_shows_a_dollar_sign(app):
    pages = every_page_in_use(app)
    screening, batch = pages[0], pages[1]
    money = [m.value for m in screening.metric if m.label.startswith("Expected")]
    money += [m.value for m in batch.metric if m.label == "Expected loss"]
    assert len(money) == 3 and all(v.startswith(("$", "-$")) for v in money), money
    for page in pages[:3]:  # every page that shows money says which currency it is shown in
        assert any("Amounts in USD" in c.value for c in page.caption)


def test_assessment_and_letter_are_written_to_the_decision_log(app, monkeypatch, tmp_path):
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr("src.audit.DECISION_LOG", True)
    monkeypatch.setattr("src.audit.DECISION_LOG_PATH", path)
    assess(app, EXAMPLES[2])
    next(b for b in app.button if b.label == "Draft letter").click().run()
    assessment, letter = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    assert assessment["decision"] == "decline" and assessment["inputs"]["AMT_CREDIT"] == 450000
    assert letter["assessment_id"] == assessment["id"] and letter["letter_backend"] == "template"
