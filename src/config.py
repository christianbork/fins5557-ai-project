"""Project settings in one place: paths, data constants and API keys.

Locally, keys come from a .env file (see .env.example).
On Streamlit Cloud, add the same keys under App settings -> Secrets.
Keys are never hardcoded or printed.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "uci_credit_default.csv"  # gitignored, downloaded on first use
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"

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

# ---- Training ---------------------------------------------------------------
CV_FOLDS = 5  # stratified K-fold on the training split, used for tuning only
# Model choice rule: take the best validation AUC, but only pick a tree model over logistic
# regression if it beats it by at least this much. Otherwise the simpler, more explainable
# model wins (easier to justify to a regulator and to explain to a declined customer).
MIN_AUC_GAIN = 0.01

# ---- Credit decision assumptions (stated in the report) ---------------------------
# Profit of approving one customer: no default -> earn MARGIN x EAD; default -> lose LGD x EAD.
# Expected loss = PD x LGD x EAD (the Basel formula).
LGD = 0.60  # loss given default: share of the exposure lost after recoveries
MARGIN = 0.12  # net annual margin on the exposure, after funding and operating costs
EAD_COLUMN = "LIMIT_BAL"  # exposure at default = the credit limit (conservative: assumes fully drawn)
# Break-even PD = MARGIN / (MARGIN + LGD) = 16.7%: above it, a customer is expected to lose money.
REFER_BAND = 0.02  # PDs within +/- 2 points of the cut-off go to an analyst (about 15% of applicants)
CALIBRATION_METHODS = ["sigmoid", "isotonic"]  # the one with the lower Brier score is used


def get_secret(name: str, default: str | None = None) -> str | None:
    """Return a secret from the environment, then Streamlit secrets, else default.

    Streamlit secrets are only checked when Streamlit is already loaded (i.e. inside the app).
    Importing Streamlit anywhere else has a side effect: it switches matplotlib to a backend
    that can't display charts, which silently breaks the plots in the notebook.
    """
    value = os.getenv(name)
    if value:
        return value
    if "streamlit" not in sys.modules:
        return default
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 (no secrets file locally raises different errors)
        return default


# ---- API keys and AI settings ----------------------------------------------
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
# Which LLM writes the decline letter: "template" (no AI, always works), "gemini" or "ollama".
LLM_BACKEND = get_secret("LLM_BACKEND", "template")
