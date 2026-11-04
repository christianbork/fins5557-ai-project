"""Station 3 (Model Design). Stage 1 of the credit decision: the lender's policy limits.

Every application is checked against hard rules before the model's PD is used. An application
that breaks a rule is declined by policy, and the model's PD is shown for information only.

Why: the model learnt only from loans Home Credit had already approved, and tree models can't
extrapolate. For an application far outside that range, such as a $1,000,000 loan on a $30,000
income repaid at $500 per instalment (2,000 instalments), the model returned a PD of about 7%,
because nothing in its training data looks like that. The out-of-range guard (src/validation.py)
only refers such cases to an analyst; a lender's own responsible-lending rules would decline them
outright. Splitting the decision into policy rules (stage 1) and the model (stage 2) is how
scorecards are used in practice, and it makes the app's "second-stage screen" scope real.

The limits are in src/config.py (POLICY_*). Each failed rule gives a message for the analyst and
a reason phrase from src/explain.py's fixed list, so the Letter Agent can name it like any other
reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import POLICY_MAX_INSTALMENTS, POLICY_MAX_LOAN_TO_INCOME, POLICY_MAX_REPAYMENT_TO_INCOME
from src.explain import REASON_PHRASES
from src.formatting import money

# Rule name -> reason phrase (must be one of the fixed phrases the letter may use)
POLICY_REASONS = {
    "instalments": REASON_PHRASES["credit_term"],  # "Length of the loan"
    "repayment_to_income": REASON_PHRASES["annuity_income_ratio"],  # "Repayments relative to income"
    "loan_to_income": REASON_PHRASES["credit_income_ratio"],  # "Loan size relative to income"
}


def check_policy(df: pd.DataFrame) -> pd.Series:
    """The policy rules each applicant breaks, as a list per row of {"rule", "reason", "message"}.
    An empty list means the applicant passes stage 1 and the model's decision stands."""
    credit = pd.to_numeric(df["AMT_CREDIT"], errors="coerce").to_numpy(dtype=float)
    annuity = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce").to_numpy(dtype=float)
    income = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        instalments = credit / annuity
        repayment_share = annuity / income
        loan_multiple = credit / income

    failures = []
    for i in range(len(df)):
        row = []
        if instalments[i] > POLICY_MAX_INSTALMENTS:
            row.append(
                {
                    "rule": "instalments",
                    "message": (
                        f"At {money(annuity[i])} per instalment the loan of {money(credit[i])} would take "
                        f"{instalments[i]:,.0f} instalments to repay; the policy limit is "
                        f"{POLICY_MAX_INSTALMENTS}."
                    ),
                }
            )
        if repayment_share[i] > POLICY_MAX_REPAYMENT_TO_INCOME:
            row.append(
                {
                    "rule": "repayment_to_income",
                    "message": (
                        f"The repayment ({money(annuity[i])}) is larger than the applicant's income "
                        f"({money(income[i])})."
                    ),
                }
            )
        if loan_multiple[i] > POLICY_MAX_LOAN_TO_INCOME:
            row.append(
                {
                    "rule": "loan_to_income",
                    "message": (
                        f"The loan is {loan_multiple[i]:,.1f} times the applicant's income; the policy limit is "
                        f"{POLICY_MAX_LOAN_TO_INCOME:g} times."
                    ),
                }
            )
        for failure in row:
            failure["reason"] = POLICY_REASONS[failure["rule"]]
        failures.append(row)
    return pd.Series(failures, index=df.index, dtype=object)


def policy_reasons(failures: list[dict]) -> list[str]:
    """The reason phrases of the rules an applicant broke, without repeats, in rule order."""
    return list(dict.fromkeys(f["reason"] for f in failures))
