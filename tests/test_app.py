"""Smoke tests for the screening page (app.py) with Streamlit's AppTest: no browser or server needed.

They use the committed model in models/ and the invented examples in data/sample/.
"""

import pandas as pd
import pytest
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
        (EXAMPLES[1], ":orange-badge[Refer to an analyst]"),
        (EXAMPLES[2], ":red-badge[Decline]"),
        (EXAMPLES[5], ":orange-badge[Refer to an analyst]"),
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
    assert any("Loan amount: must be at least 1" in e.value for e in app.error)
    assert not app.metric  # nothing scored


def test_model_and_fairness_page_loads():
    page = AppTest.from_file(str(ROOT / "pages" / "2_Model_and_fairness.py"), default_timeout=120).run()
    assert not page.exception and not page.error
    assert [t.label for t in page.tabs] == ["Sex", "Age band", "Family status", "Education"]
    assert page.metric[0].value == "XGBoost"
