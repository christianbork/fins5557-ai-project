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
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"  # training ranges and shares, no rows
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
CV_FOLDS = (
    3  # stratified K-fold on the training split, used for tuning only (3 keeps it to minutes on 184k rows)
)
N_ITER = 30  # random settings tried per tree model (RandomizedSearchCV); logistic regression uses a full grid
# CV scores for tuning: refit (choose settings) on AUC, also record PR-AUC, which focuses on the
# rare defaulters. Model v2 (Step 5c); v1 used small fixed grids and fixed class weights.
TUNING_SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
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
# Which rule sets the cut-off. All three are computed and compared (reports/threshold_comparison.csv,
# the Model and fairness page); DECISION_RULE is the one the app uses. Decided (5 Nov): "cost_ratio".
# - "profit": the cut-off that maximises realised profit on the validation split under LGD and MARGIN
#   (Thomas, 2000; Verbraken et al., 2014). Implied cost ratio LGD / MARGIN = 4 : 1.
# - "cost_ratio": approving a defaulter is treated as COST_RATIO times as costly as declining a good
#   customer, the cost matrix of the German credit data (Hofmann, 1994). Approve while
#   PD x COST_RATIO < (1 - PD), i.e. below PD = 1 / (1 + COST_RATIO) = 16.7%. More conservative than
#   "profit", and it also counts the borrower's harm from a default, not only the lender's loss.
# - "balanced_accuracy": the cut-off that maximises (TPR + TNR) / 2 on the validation split
#   (Brodersen et al., 2010), the method taught in Big Data Analytics at CBS (A. Sorensen). With 8%
#   defaulters it implies a cost ratio of about 11 : 1, set by the class balance, not the economics.
DECISION_RULES = ("profit", "cost_ratio", "balanced_accuracy")
DECISION_RULE = "cost_ratio"
COST_RATIO = 5.0  # Hofmann (1994): misclassifying a bad customer as good costs 5, a good one as bad costs 1
REFER_BAND = 0.02  # PDs within +/- 2 points of the cut-off go to an analyst (check the share after training)
# Display only. Home Credit's data doesn't state its currency. Until 5 Nov amounts were shown in USD; user
# testing showed the $ sign made people type US-dollar amounts on the wrong scale, so amounts are now shown in
# neutral currency units (CU). Earlier note: amounts were shown in USD
# for readability, with this note on every page that shows money. No conversion is applied.
CURRENCY_SYMBOL = "CU "
CURRENCY_CODE = "CU"
CURRENCY_NOTE = (
    "Amounts in currency units (CU): Home Credit's data doesn't state its currency, so amounts are the "
    "dataset's own numbers, not dollars."
)
CALIBRATION_METHODS = ["sigmoid", "isotonic"]  # the one with the lower Brier score is used

# ---- Stage 1: lending policy limits, checked before the model (src/policy.py) ----------
# Hard rules a lender applies to every application before any scoring. An application that breaks
# one is declined by policy, whatever the model says: a tree model can't extrapolate, so for a loan
# unlike anything Home Credit approved (e.g. a CU 1m loan repaid at CU 500 per instalment) its PD can be
# meaninglessly low. Decided (4 Nov): limits just beyond the range of Home Credit's approved loans,
# so the rules only catch applications the model has no evidence about and leave the model's
# reported test results unchanged. A real lender would set them from its own responsible-lending
# policy (affordability, NCCP Act).
POLICY_MAX_INSTALMENTS = (
    45  # loan / repayment; Home Credit's approved loans needed at most 45 (was 48 until 5 Nov)
)
POLICY_MAX_REPAYMENT_TO_INCOME = 1.0  # a repayment larger than the applicant's income can't be afforded
POLICY_MAX_LOAN_TO_INCOME = 30.0  # loan more than 30x income (0.0075% of approved loans)

# ---- Input validation and the out-of-range guard ---------------------------------
# The model only saw loans Home Credit had already approved (selection bias, see BUILD_PLAN).
# An input below the lowest or above the highest value in the training data is outside what the
# model knows (tree models can't extrapolate), so such an applicant is referred to an analyst
# instead of being approved automatically. Decided (28 Oct): the full training range, not the
# middle 99%, which would flag 1% per field even for ordinary applicants (7.8% of validation
# applicants, who defaulted less than average) and triple the analysts' refer workload.
GUARD_QUANTILES = (0.0, 1.0)
# Decided (5 Nov, after user testing): the guard also checks the ratios the model uses, because every
# typed field can be normal on its own while the combination is not (a tester's 48-instalment loan at
# twice the goods price, with no credit scores, got a PD of 2.8%). Ratios outside the middle 99.8% of
# past approved loans, or no external credit score from any source (0.06% of past applicants), are
# referred to an analyst rather than approved.
RATIO_GUARD_QUANTILES = (0.001, 0.999)
# ratio name -> (numerator, denominator, label); the model's engineered features (src/features.py)
GUARD_RATIOS = {
    "instalments": ("AMT_CREDIT", "AMT_ANNUITY", "Number of instalments (loan ÷ repayment)"),
    "loan_to_goods": ("AMT_CREDIT", "AMT_GOODS_PRICE", "Loan relative to the price of the goods"),
    "repayment_to_income": ("AMT_ANNUITY", "AMT_INCOME_TOTAL", "Repayment relative to income"),
    "loan_to_income": ("AMT_CREDIT", "AMT_INCOME_TOTAL", "Loan relative to income"),
}
# The form shows the middle 80% of past applicants under each amount (deciles 1 and 9), and a note
# when an amount is outside the middle 98% (TYPICAL_QUANTILES), so amounts are entered on the data's scale.
TYPICAL_QUANTILES = (0.01, 0.99)
MAX_BATCH_ROWS = 10_000  # batch scoring limit, keeps the app responsive
MAX_UPLOAD_MB = 5

# ---- Fairness audit (Step 10, src/fairness.py) ---------------------------------------------
DI_THRESHOLD = 0.80  # disparate-impact ratio below this is flagged (the "four-fifths rule")
DP_THRESHOLD = 0.05  # approval-rate gap above this is flagged (CFPB safe harbour, Code Bank W04-06)
EO_THRESHOLD = 0.05  # equalised-odds gap above this is flagged (same threshold as W04-06 uses for parity)
MIN_GROUP_SIZE = 100  # smaller groups are reported but not judged: their rates are too noisy


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
# Which backend writes the decline letter (src/llm.py): "template" (fixed wording, no language
# model, always works), "ollama" (a local model: for real customer data, nothing leaves the
# machine) or "gemini" (Google's API: for the public demo, which only handles invented applicants).
# Whatever the backend, the prompt holds only the decision and fixed reason phrases, no applicant data.
LLM_BACKEND = get_secret("LLM_BACKEND", "template")
# Tested 29 Oct on 6 letters: 3.1-flash-lite passed all 6 (4-14 s); 3.5-flash lost 2 to "overloaded" (503)
# errors; 2.5-flash no longer serves requests. The backup is tried once if the main model is overloaded
# or out of quota.
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_BACKUP_MODEL = get_secret("GEMINI_BACKUP_MODEL", "gemini-3.5-flash")
OLLAMA_URL = get_secret("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = get_secret("OLLAMA_MODEL", "llama3.2:3b")  # pulled once with: ollama pull llama3.2:3b
LLM_TIMEOUT_SECONDS = 60  # a small local model on a laptop CPU takes about 10-30 s per letter
LENDER_NAME = "Example Consumer Finance"  # invented: the app is a demo, not a real lender

# ---- Decision log (src/audit.py) ---------------------------------------------------------------
# One JSON line per assessed applicant: on locally, off on the public demo (DECISION_LOG = "off" in
# the app's Secrets), where it would only collect what visitors type in.
DECISION_LOG = str(get_secret("DECISION_LOG", "on")).strip().lower() in ("on", "true", "1", "yes")
DECISION_LOG_PATH = ROOT / "logs" / "decisions.jsonl"  # gitignored
