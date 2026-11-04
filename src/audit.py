"""Station 4 (Implementation). Decision log: one JSON line per assessed applicant, the audit trail a real
lender would keep.

    from src.audit import log_assessment, log_letter
    entry_id = log_assessment(applicant, result, metadata, source="screening")
    log_letter(entry_id, letter["backend"])           # when a letter is drafted for it

Each line records when, which model version, the inputs, the PD, the decision, the reasons and the
warnings, so any decision can be traced and explained later. Letters are drafted after the
assessment, so they get their own line pointing back to it by id.

Written to logs/decisions.jsonl (gitignored). Controlled by DECISION_LOG: on locally, off on the
public demo (set DECISION_LOG = "off" in the app's Secrets), where a log would only collect what
visitors type in. Logging never stops the app: if the file can't be written, it logs a warning.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DECISION_LOG, DECISION_LOG_PATH

log = logging.getLogger(__name__)


def _plain(value):
    """numpy numbers -> Python numbers and NaN -> None, so every line is valid JSON."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _write(entries: list[dict], path: Path | None, enabled: bool | None) -> bool:
    """Append entries; path and enabled default to the config settings (read at call time)."""
    path = DECISION_LOG_PATH if path is None else path
    if not (DECISION_LOG if enabled is None else enabled) or not entries:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in entries)
        return True
    except OSError:
        log.warning("Couldn't write the decision log to %s", path, exc_info=True)
        return False


def _entry(
    inputs: dict, pd_: float, decision: str, reasons, warnings, metadata: dict, source: str, policy=()
) -> dict:
    return {
        "event": "assessment",
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "model": {"name": metadata["chosen_model"], "trained_at": metadata["trained_at"]},
        "inputs": {k: _plain(v) for k, v in inputs.items()},
        "pd": round(float(pd_), 6),
        "decision": decision,
        "reasons": list(reasons),
        "warnings": list(warnings),
        "policy": list(policy),  # lending policy rules broken (stage 1); any one means decline
    }


def log_assessment(
    applicant: dict,
    result: dict,
    metadata: dict,
    source: str = "screening",
    path: Path | None = None,
    enabled: bool | None = None,
) -> str | None:
    """Log one assessment (src.predict.predict's result). Returns its id, or None if not logged."""
    entry = _entry(
        applicant,
        result["pd"],
        result["decision"],
        result["reasons"],
        result["warnings"],
        metadata,
        source,
        policy=[f["message"] for f in result.get("policy", [])],
    )
    return entry["id"] if _write([entry], path, enabled) else None


def log_batch(
    scored: pd.DataFrame,
    metadata: dict,
    path: Path | None = None,
    enabled: bool | None = None,
) -> int:
    """Log every row of a scored batch (src.predict.predict_batch's output). Returns the lines written."""
    inputs = scored[metadata["input_columns"]].to_dict("records")
    reasons = scored["reasons"] if "reasons" in scored else pd.Series("", index=scored.index)
    policy = scored["policy"] if "policy" in scored else pd.Series("", index=scored.index)
    entries = [
        _entry(
            row,
            pd_,
            decision,
            [r for r in reason.split("; ") if r],
            [w for w in warning.split("; ") if w],
            metadata,
            "batch",
            policy=[rule] if rule else [],
        )
        for row, pd_, decision, reason, warning, rule in zip(
            inputs, scored["pd"], scored["decision"], reasons, scored["warnings"], policy, strict=True
        )
    ]
    return len(entries) if _write(entries, path, enabled) else 0


def log_letter(
    assessment_id: str | None,
    backend: str,
    path: Path | None = None,
    enabled: bool | None = None,
) -> bool:
    """Log that a letter was drafted for an assessment, and which backend wrote it."""
    if assessment_id is None:
        return False
    entry = {
        "event": "letter",
        "assessment_id": assessment_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "letter_backend": backend,
    }
    return _write([entry], path, enabled)
