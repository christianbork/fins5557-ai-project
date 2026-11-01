"""Tests for src/audit.py: the decision log, written to a temp file."""

import json

import numpy as np
import pandas as pd

from src.audit import log_assessment, log_batch, log_letter

META = {
    "chosen_model": "xgboost",
    "trained_at": "2026-10-29T10:00:00+00:00",
    "input_columns": ["AMT_CREDIT", "EXT_SOURCE_1"],
}
RESULT = {
    "pd": np.float64(0.4754),
    "decision": "decline",
    "reasons": ["External credit scores"],
    "warnings": [],
}


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_assessment_is_one_json_line_with_everything_needed_to_trace_it(tmp_path):
    path = tmp_path / "logs" / "decisions.jsonl"
    applicant = {"AMT_CREDIT": np.int64(450000), "EXT_SOURCE_1": np.nan}
    entry_id = log_assessment(applicant, RESULT, META, path=path, enabled=True)
    [entry] = lines(path)
    assert entry["id"] == entry_id and entry["event"] == "assessment" and entry["source"] == "screening"
    assert entry["model"] == {"name": "xgboost", "trained_at": "2026-10-29T10:00:00+00:00"}
    assert entry["inputs"] == {"AMT_CREDIT": 450000, "EXT_SOURCE_1": None}  # NaN is written as null
    assert entry["pd"] == 0.4754 and entry["decision"] == "decline"
    assert entry["reasons"] == ["External credit scores"] and entry["warnings"] == []
    assert entry["timestamp"].endswith("+00:00")


def test_letter_points_back_to_its_assessment(tmp_path):
    path = tmp_path / "decisions.jsonl"
    entry_id = log_assessment({"AMT_CREDIT": 1}, RESULT, META, path=path, enabled=True)
    assert log_letter(entry_id, "gemini", path=path, enabled=True)
    letter = lines(path)[1]
    assert letter["event"] == "letter" and letter["assessment_id"] == entry_id
    assert letter["letter_backend"] == "gemini"


def test_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "decisions.jsonl"
    for _ in range(3):
        log_assessment({"AMT_CREDIT": 1}, RESULT, META, path=path, enabled=True)
    assert len(lines(path)) == 3


def test_off_writes_nothing(tmp_path):
    path = tmp_path / "decisions.jsonl"
    assert log_assessment({"AMT_CREDIT": 1}, RESULT, META, path=path, enabled=False) is None
    assert not log_letter(None, "template", path=path, enabled=True)
    assert not path.exists()


def test_off_by_default_in_tests(tmp_path):
    """conftest turns the log off, so the app tests never write to logs/."""
    assert log_assessment({"AMT_CREDIT": 1}, RESULT, META) is None


def test_batch_logs_one_line_per_row(tmp_path):
    path = tmp_path / "decisions.jsonl"
    scored = pd.DataFrame(
        {
            "AMT_CREDIT": [450000, 270000],
            "EXT_SOURCE_1": [0.7, np.nan],
            "pd": [0.01, 0.48],
            "decision": ["approve", "decline"],
            "reasons": ["", "External credit scores; Length of the loan"],
            "warnings": ["", ""],
        }
    )
    assert log_batch(scored, META, path=path, enabled=True) == 2
    approved, declined = lines(path)
    assert approved["source"] == "batch" and approved["reasons"] == [] and approved["warnings"] == []
    assert declined["reasons"] == ["External credit scores", "Length of the loan"]
    assert declined["inputs"]["EXT_SOURCE_1"] is None


def test_unwritable_log_does_not_stop_the_app(tmp_path, caplog):
    blocked = tmp_path / "a_file"
    blocked.write_text("not a folder")
    assert (
        log_assessment({"AMT_CREDIT": 1}, RESULT, META, path=blocked / "decisions.jsonl", enabled=True)
        is None
    )
    assert "Couldn't write the decision log" in caplog.text
