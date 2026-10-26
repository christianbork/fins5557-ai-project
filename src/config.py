"""Project settings in one place: paths, data constants and API keys.

Locally, keys come from a .env file (see .env.example).
On Streamlit Cloud, add the same keys under App settings -> Secrets.
Keys are never hardcoded or printed.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "uci_credit_default.csv"  # gitignored, downloaded on first use
MODELS_DIR = ROOT / "models"

# ---- Data -----------------------------------------------------------------
UCI_DATASET_ID = 350  # UCI "Default of Credit Card Clients" (Taiwan, 2005), CC BY 4.0
TARGET = "default"  # 1 = defaulted on next month's payment
# Kept for the fairness audit only, never used as model inputs.
PROTECTED = ["SEX", "MARRIAGE", "AGE"]
RANDOM_STATE = 100  # same seed as the notebooks, so results are reproducible
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # share of the full dataset; the remaining 60% is training

# ---- Features ---------------------------------------------------------------
# Engineered in src/features.py from the raw bureau-style columns.
ENGINEERED_FEATURES = [
    "utilisation",  # last bill / credit limit
    "payment_ratio",  # last payment / the bill it paid (BILL_AMT2), 1.0 when nothing was owed
    "months_late_6m",  # number of the last 6 months with a late payment
    "max_months_late",  # worst arrears in the last 6 months
    "avg_utilisation_6m",  # average bill / credit limit over 6 months
    "bill_trend",  # BILL_AMT1 - BILL_AMT6: positive means debt is growing
]
_RAW_NUMERIC = [
    "LIMIT_BAL",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
]
# Each feature set lists the numeric and categorical model inputs. Protected attributes
# (PROTECTED) never appear here. A reduced "thin-file" set can be added as another entry.
FEATURE_SETS = {
    "full": {
        "numeric": [*_RAW_NUMERIC, *ENGINEERED_FEATURES],
        "categorical": ["EDUCATION"],
    },
}
FEATURE_SET = "full"  # change this one setting to train on a different feature set


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return a secret from the environment, then Streamlit secrets, else default."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 (no secrets file locally raises different errors)
        return default


# ---- API keys and AI settings ----------------------------------------------
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
# Which LLM writes the decline letter: "template" (no AI, always works), "gemini" or "ollama".
LLM_BACKEND = get_secret("LLM_BACKEND", "template")
