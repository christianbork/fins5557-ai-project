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
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"

# ---- Data -----------------------------------------------------------------
# Home Credit Default Risk (Kaggle, 2018): 307,511 consumer loan applications, one row each.
# Source: https://www.kaggle.com/competitions/home-credit-default-risk/data (competition rules:
# the data is not redistributed, so data/raw/ is gitignored; see docs/BUILD_PLAN.md for how to get it).
RAW_PATH = DATA_DIR / "raw" / "application_train.csv"
KAGGLE_URL = "https://www.kaggle.com/competitions/home-credit-default-risk/data"
ID_COLUMN = "SK_ID_CURR"
RAW_TARGET = "TARGET"  # Home Credit's name: 1 = payment difficulties on the loan
TARGET = "default"  # our name for it, used everywhere else in the code
# Kept for the fairness audit only, never used as model inputs.
# DAYS_BIRTH is age in days, counted backwards from the application (e.g. -12,000 = about 33 years).
PROTECTED = ["CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS"]
RANDOM_STATE = 100  # same seed as the notebooks, so results are reproducible
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # share of the full dataset; the remaining 60% is training

# ---- Features ---------------------------------------------------------------
# Raw inputs the model uses: information a lender has at the time of application, kept to a
# number an analyst can enter in the app. Deliberately left out:
#   - DEF_30/60_CNT_SOCIAL_CIRCLE (defaults among the applicant's social circle): judging
#     someone by their friends' behaviour is hard to justify to a customer or regulator.
#   - Building/apartment statistics (about 50 mostly-empty columns) and document flags.
NUMERIC_INPUTS = [
    "AMT_INCOME_TOTAL",  # annual income
    "AMT_CREDIT",  # loan amount (also the exposure at default)
    "AMT_ANNUITY",  # regular loan repayment
    "AMT_GOODS_PRICE",  # price of the goods the loan pays for (consumer loans)
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_EMPLOYED",  # days in current job, negative; 365243 means not employed (e.g. pensioners)
    "DAYS_REGISTRATION",  # days since the applicant changed their registration
    "DAYS_ID_PUBLISH",  # days since the applicant changed their ID document
    "OWN_CAR_AGE",  # empty if no car
    "REGION_RATING_CLIENT_W_CITY",  # Home Credit's rating of the applicant's region (1-3)
    "EXT_SOURCE_1",  # scores from external data sources (credit bureaus), 0-1, often missing:
    "EXT_SOURCE_2",  # a missing score is itself a sign of a thin credit file
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR",  # credit bureau enquiries in the last year
]
CATEGORICAL_INPUTS = [
    "NAME_CONTRACT_TYPE",  # cash loan or revolving loan
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",  # working, pensioner, state servant...
    "NAME_EDUCATION_TYPE",  # also audited for fairness
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",  # 58 types; rare ones are grouped by the encoder
]
RAW_INPUTS = [*NUMERIC_INPUTS, *CATEGORICAL_INPUTS]
DAYS_EMPLOYED_SENTINEL = 365243  # Home Credit's code for "not employed"

# Engineered in src/features.py from the raw inputs.
ENGINEERED_FEATURES = [
    "credit_income_ratio",  # loan amount / annual income
    "annuity_income_ratio",  # repayment / income: the affordability (debt-service) ratio
    "credit_term",  # loan amount / repayment: roughly the number of repayments
    "goods_credit_ratio",  # goods price / loan amount: below 1 means extra cash on top of the goods
    "years_employed",  # DAYS_EMPLOYED in years, empty when not employed
    "not_employed",  # 1 when DAYS_EMPLOYED is the "not employed" code or missing
    "ext_source_mean",  # average of the external scores that are available
    "ext_source_missing",  # how many of the three external scores are missing (0-3): thin-file signal
    "income_per_person",  # income / family members
]
# Each feature set lists the numeric and categorical model inputs. Protected attributes
# (PROTECTED) never appear here. A reduced "thin-file" set (e.g. without EXT_SOURCE_*)
# can be added as another entry.
FEATURE_SETS = {
    "full": {
        "numeric": [*NUMERIC_INPUTS, *ENGINEERED_FEATURES],
        "categorical": CATEGORICAL_INPUTS,
    },
}
FEATURE_SET = "full"  # change this one setting to train on a different feature set
RARE_CATEGORY_SHARE = 0.01  # categories rarer than 1% of training rows are grouped together

# ---- Training ---------------------------------------------------------------
CV_FOLDS = 3  # stratified K-fold on the training split, used for tuning only (3 keeps it to minutes on 184k rows)
# Model choice rule: take the best validation AUC, but only pick a tree model over logistic
# regression if it beats it by at least this much. Otherwise the simpler, more explainable
# model wins (easier to justify to a regulator and to explain to a declined customer).
MIN_AUC_GAIN = 0.01

# ---- Credit decision assumptions (stated in the report) ---------------------------
# Profit of approving one loan: no default -> earn MARGIN x EAD over the loan's life;
# default -> lose LGD x EAD. Expected loss = PD x LGD x EAD (the Basel formula).
# The target covers payment difficulties on the loan, so both the PD and the margin refer to
# the life of the loan (no one-month vs annual mismatch as with the Taiwan data).
LGD = 0.60  # loss given default: share of the loan lost after recoveries (unsecured consumer credit)
MARGIN = 0.15  # net margin over the loan's life, after funding and operating costs, as a share of the loan
EAD_COLUMN = "AMT_CREDIT"  # exposure at default = the loan amount (conservative: no repayments yet)
# Break-even PD = MARGIN / (MARGIN + LGD) = 20%: above it, a loan is expected to lose money.
REFER_BAND = 0.02  # PDs within +/- 2 points of the cut-off go to an analyst (check the share after training)
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
