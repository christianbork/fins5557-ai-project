"""Station 4 (Implementation). Logging for the app: error details go to the server log, never onto the page.

    from src.app_logging import get_logger
    log = get_logger("pages.batch")
    log.exception("Scoring failed")     # full traceback in the log; the page shows a short message

Locally the log is the terminal running `streamlit run`; on Streamlit Cloud it is the app's
"Manage app" log, which only the app's owner can see.
"""

from __future__ import annotations

import logging

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """A logger for one page or module. Configures the root logger the first time (basicConfig does
    nothing once a handler exists, so calling this on every Streamlit rerun is safe)."""
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    return logging.getLogger(name)
