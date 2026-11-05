"""Tests for src/model_card.py, on the committed model's metadata and reports."""

import re

from src.config import ROOT
from src.model_card import MODEL_CARD_PATH, write_model_card


def test_model_card_has_every_section_and_no_raw_values(tmp_path):
    card = write_model_card(out_path=tmp_path / "MODEL_CARD.md").read_text(encoding="utf-8")
    for section in [
        "Intended use and scope",
        "Data and terms",
        "Inputs",
        "Protected attributes",
        "Training and tuning",
        "Performance",
        "Calibration",
        "Cut-off and assumptions",
        "Fairness",
        "Limitations",
    ]:
        assert f"## {section}" in card
    assert "MODEL_HISTORY.md" in card and "Amounts in currency units (CU)" in card
    assert not re.search(r"\b(None|nan)\b|xgboost|logistic_regression", card)


def test_committed_model_card_is_up_to_date(tmp_path):
    """docs/MODEL_CARD.md must match the committed model: regenerate it with
    python -m src.reporting model-card after retraining or changing src/model_card.py."""
    fresh = write_model_card(out_path=tmp_path / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert MODEL_CARD_PATH == ROOT / "docs" / "MODEL_CARD.md"
    assert MODEL_CARD_PATH.read_text(encoding="utf-8") == fresh
