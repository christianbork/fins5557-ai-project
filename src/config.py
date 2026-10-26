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
