"""How numbers are written in the app: one function per kind of number, used by every page,
table, chart label and CSV export, so the same value always looks the same.

    money(450000)        -> "$450,000"      money_short(3.62e9) -> "$3.62bn"
    pct(0.475)           -> "47.5%"         pct_points(0.043)   -> "+4.3 pts"
    ratio(0.853)         -> "0.85"          score(0.7571)       -> "0.757"
    count(6, "applicant") -> "6 applicants"

Every function returns missing() (an en dash) for None or NaN, so tables never show None or nan.
Amounts are in the dataset's own units, shown with a $ as a stated assumption (config.CURRENCY_NOTE).
"""

from __future__ import annotations

import math

from src.config import CURRENCY_SYMBOL

MISSING = "–"  # en dash
SHORT_UNITS = [(1e9, "bn", 2), (1e6, "m", 2), (1e3, "k", 1)]  # (size, suffix, decimals)


def missing() -> str:
    """What an empty value looks like."""
    return MISSING


def _is_missing(x) -> bool:
    try:
        return x is None or math.isnan(float(x))
    except (TypeError, ValueError):
        return x == ""


def money(x, decimals: int = 0) -> str:
    """450000 -> "$450,000"; -92951 -> "-$92,951"."""
    if _is_missing(x):
        return MISSING
    sign = "-" if round(float(x), decimals) < 0 else ""
    return f"{sign}{CURRENCY_SYMBOL}{abs(float(x)):,.{decimals}f}"


def money_short(x) -> str:
    """3.62e9 -> "$3.62bn", 128_400 -> "$128.4k", 950 -> "$950", for headline figures and chart axes."""
    if _is_missing(x):
        return MISSING
    value = abs(float(x))
    sign = "-" if float(x) < 0 else ""
    for size, suffix, decimals in SHORT_UNITS:
        if round(value / size, decimals) >= 1:
            return f"{sign}{CURRENCY_SYMBOL}{value / size:,.{decimals}f}{suffix}"
    return money(x)


def pct(x) -> str:
    """A share as a percentage with one decimal: 0.005 -> "0.5%", 0.475 -> "47.5%"."""
    return MISSING if _is_missing(x) else f"{float(x):.1%}"


def pct_points(x) -> str:
    """A gap between two shares, in percentage points with a sign: 0.043 -> "+4.3 pts"."""
    return MISSING if _is_missing(x) else f"{float(x) * 100:+.1f} pts"


def ratio(x) -> str:
    """A ratio with two decimals, e.g. the disparate-impact ratio: 0.8534 -> "0.85"."""
    return MISSING if _is_missing(x) else f"{float(x):.2f}"


def score(x) -> str:
    """A model metric (AUC, Gini, KS, Brier, PSI) with three decimals: 0.75712 -> "0.757"."""
    return MISSING if _is_missing(x) else f"{float(x):.3f}"


def number(x, decimals: int = 0) -> str:
    """A plain number with thousands separators: 1000 -> "1,000"."""
    return MISSING if _is_missing(x) else f"{float(x):,.{decimals}f}"


def count(n, noun: str, plural: str | None = None) -> str:
    """1 -> "1 applicant", 6 -> "6 applicants", 1200 -> "1,200 applicants" (no "(s)")."""
    if _is_missing(n):
        return MISSING
    n = int(n)
    return f"{n:,} {noun if n == 1 else plural or noun + 's'}"
