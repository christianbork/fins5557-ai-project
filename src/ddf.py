"""Station 4 (Implementation). The course's Digital Data Factory (DDF): which part of this project sits at
which station.

The DDF (FINS5557 Week 1, Hric & Lin 2026) treats an AI system as four stations, each with its own
accountability: 1 ETL, 2 Feature Engineering, 3 Model Design, 4 Implementation. An error at an
early station can't be fixed later ("if data is corrupted at Station 1, no amount of modelling
sophistication at Station 3 can correct it"), so each station lists the checks it owns.

This table is the single source for the map shown on the "Model and fairness" page and in the README.
tests/test_ddf.py checks that every module in src/ (and every page) belongs to exactly one station;
src/config.py holds the settings all stations share.
"""

from __future__ import annotations

DDF_STATIONS: list[dict] = [
    {
        "station": 1,
        "name": "ETL",
        "role": "Gate keepers: data authenticity and integrity",
        "what_we_do": (
            "Load the Home Credit applications, check the columns and types, clean the 365,243-day "
            "employment code, and split 60/20/20 (stratified) before anything is fitted."
        ),
        "checks": (
            "Data source and competition terms documented; raw data never committed; split first to prevent "
            "leakage; selection bias recorded (only previously approved loans have outcomes)."
        ),
        "modules": ["src/data.py"],
    },
    {
        "station": 2,
        "name": "Feature Engineering",
        "role": "Model-ready features with documented transformation logic",
        "what_we_do": (
            "Nine credit-risk ratios (loan-to-income, repayment-to-income, loan term...), missing-value "
            "flags, training-set medians, rare categories pooled, one-hot encoding in one sklearn pipeline."
        ),
        "checks": (
            "Bias check before modelling: sex, age and family status are never inputs; proxies such as "
            "'not employed' and education are analysed (notebook section 6)."
        ),
        "modules": ["src/features.py"],
    },
    {
        "station": 3,
        "name": "Model Design",
        "role": "Train, validate and stress-test against acceptance criteria; answer 'so what?'",
        "what_we_do": (
            "Compare logistic regression, random forest and XGBoost with cross-validation; calibrate the PDs; "
            "set the approve / refer / decline cut-off from the lender's economics, behind hard lending policy "
            "limits checked first; audit fairness."
        ),
        "checks": (
            "A complex model must beat the logistic baseline by 0.01 AUC; calibration checked by Brier score; "
            "four-fifths rule and error-rate gaps by group; model card for independent review (SR 26-2)."
        ),
        "modules": [
            "src/train.py",
            "src/metrics.py",
            "src/threshold.py",
            "src/policy.py",
            "src/fairness.py",
            "src/reporting.py",
            "src/model_card.py",
        ],
    },
    {
        "station": 4,
        "name": "Implementation",
        "role": "A product for the business, with monitoring and a feedback loop to Station 3",
        "what_we_do": (
            "The Streamlit app: validate inputs, score, explain with SHAP, draft the letter with the Letter "
            "Agent behind guardrails, batch scoring, PSI drift monitoring and a decision log."
        ),
        "checks": (
            "Unusual applicants referred to a human; no applicant data sent to the language model; every letter "
            "checked and edited by an analyst; PSI alerts trigger a model review; every decision logged."
        ),
        "modules": [
            "app.py",
            "pages/1_Batch_scoring.py",
            "pages/2_Model_and_fairness.py",
            "pages/3_Monitoring.py",
            "src/applicant.py",
            "src/validation.py",
            "src/predict.py",
            "src/explain.py",
            "src/llm.py",
            "src/monitoring.py",
            "src/audit.py",
            "src/app_logging.py",
            "src/formatting.py",
            "src/labels.py",
            "src/ddf.py",
        ],
    },
]


def station_of(module: str) -> int:
    """The DDF station a module (e.g. "src/llm.py") belongs to."""
    for station in DDF_STATIONS:
        if module in station["modules"]:
            return station["station"]
    raise KeyError(f"{module} is not assigned to a DDF station in src/ddf.py")
