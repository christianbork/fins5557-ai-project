"""Tests for src/formatting.py."""

import numpy as np
import pytest

from src.formatting import count, missing, money, money_short, number, pct, pct_points, ratio, score

EMPTY = [None, float("nan"), np.nan, np.float64("nan")]


@pytest.mark.parametrize("fmt", [money, money_short, pct, pct_points, ratio, score, number])
@pytest.mark.parametrize("value", EMPTY)
def test_empty_values_show_a_dash(fmt, value):
    assert fmt(value) == missing() == "–"


def test_count_and_blank_text_show_a_dash():
    assert count(None, "applicant") == missing()
    assert pct("") == missing()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (450000, "$450,000"),
        (0, "$0"),
        (-92951, "-$92,951"),
        (-0.4, "$0"),  # rounds to zero: no "-$0"
        (1407.4, "$1,407"),
        (np.int64(65741), "$65,741"),
        (3.621925180e9, "$3,621,925,180"),
        (1e15, "$1,000,000,000,000,000"),
    ],
)
def test_money(value, expected):
    assert money(value) == expected


def test_money_decimals():
    assert money(12.345, decimals=2) == "$12.35"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.62e9, "$3.62bn"),
        (3_178_209_835, "$3.18bn"),
        (128_400, "$128.4k"),
        (4_500_000, "$4.50m"),
        (950, "$950"),
        (0, "$0"),
        (-92_951, "-$93.0k"),
        (999_999, "$1.00m"),  # rounds up into the next unit, not "$1000.0k"
        (2.5e13, "$25,000.00bn"),
    ],
)
def test_money_short(value, expected):
    assert money_short(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.005, "0.5%"), (0.475, "47.5%"), (0, "0.0%"), (1, "100.0%"), (-0.031, "-3.1%"), (12.5, "1250.0%")],
)
def test_pct(value, expected):
    assert pct(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(0.043, "+4.3 pts"), (-0.05, "-5.0 pts"), (0, "+0.0 pts")])
def test_pct_points(value, expected):
    assert pct_points(value) == expected


def test_ratio_and_score():
    assert ratio(0.8534) == "0.85"
    assert ratio(0) == "0.00"
    assert score(0.75712) == "0.757"
    assert score(6.1289) == "6.129"


def test_number():
    assert number(1000) == "1,000"
    assert number(-2190.5, decimals=1) == "-2,190.5"


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, "1 applicant"), (6, "6 applicants"), (0, "0 applicants"), (1200, "1,200 applicants")],
)
def test_count(n, expected):
    assert count(n, "applicant") == expected


def test_count_irregular_plural():
    assert count(1, "row has", "rows have") == "1 row has"
    assert count(3, "row has", "rows have") == "3 rows have"
