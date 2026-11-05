"""Stage 1 lending policy rules (src/policy.py)."""

import pandas as pd

from src.config import POLICY_MAX_INSTALMENTS, POLICY_MAX_LOAN_TO_INCOME
from src.explain import REASON_PHRASES
from src.policy import POLICY_REASONS, check_policy, policy_reasons


def frame(credit, annuity, income):
    return pd.DataFrame({"AMT_CREDIT": [credit], "AMT_ANNUITY": [annuity], "AMT_INCOME_TOTAL": [income]})


def rules(credit, annuity, income):
    return [f["rule"] for f in check_policy(frame(credit, annuity, income)).iloc[0]]


def test_an_ordinary_application_passes():
    assert rules(450_000, 22_500, 157_500) == []  # 20 instalments, repayment 14% of income, loan 2.9x income


def test_the_tested_application_breaks_two_rules():
    failures = check_policy(frame(1_000_000, 500, 30_000)).iloc[0]
    assert [f["rule"] for f in failures] == ["instalments", "loan_to_income"]
    assert "2,000 instalments" in failures[0]["message"] and "33.3 times" in failures[1]["message"]


def test_each_limit_is_inclusive():
    assert rules(POLICY_MAX_INSTALMENTS * 1_000, 1_000, 1_000_000) == []
    assert rules(POLICY_MAX_INSTALMENTS * 1_000 + 1, 1_000, 1_000_000) == ["instalments"]
    assert rules(300_000, 10_000, 10_000) == []  # repayment = income and loan = 30x income: at the limits
    assert rules(300_000, 10_001, 10_000) == ["repayment_to_income"]
    assert rules(POLICY_MAX_LOAN_TO_INCOME * 10_000 + 10_000, 9_000, 10_000) == ["loan_to_income"]


def test_reasons_come_from_the_fixed_phrase_list():
    assert set(POLICY_REASONS.values()) <= set(REASON_PHRASES.values())  # the letter accepts them
    failures = check_policy(frame(1_000_000, 500, 30_000)).iloc[0]
    assert policy_reasons(failures) == ["Length of the loan", "Loan size relative to income"]


def test_missing_values_do_not_break_a_rule():
    assert rules(None, 500, 30_000) == []  # validation reports the missing loan; policy stays silent


def test_second_user_test_is_declined_by_policy():
    """Mathilde's case (5 Nov): $120,000 at $2,500 per instalment = 48 instalments, beyond the 45 any
    past approved loan needed. It was approved at a PD of 2.8% before the limit was lowered from 48."""
    assert POLICY_MAX_INSTALMENTS == 45
    assert rules(120_000, 2_500, 26_000) == ["instalments"]
