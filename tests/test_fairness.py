"""Tests for src/fairness.py and src/reporting.py on small hand-made examples."""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET
from src.fairness import audit_groups, fairness_table
from src.reporting import reliability_points, roc_points, write_app_reports


@pytest.fixture
def ten_applicants():
    """5 women, 5 men, all aged about 33, married, secondary education. Cut-off 0.5.

    Women: 4 good (all approved) + 1 defaulter (declined) -> approval 0.8, good approved 1.0, defaulters 0.0
    Men:   3 good (2 approved) + 2 defaulters (1 approved) -> approval 0.6, good approved 2/3, defaulters 1/2
    """
    df = pd.DataFrame(
        {
            "CODE_GENDER": ["F"] * 5 + ["M"] * 5,
            "DAYS_BIRTH": -12_000,
            "NAME_FAMILY_STATUS": "Married",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            TARGET: [0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
        }
    )
    pd_ = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.6, 0.1, 0.9])
    return df, pd_


def row(table, attribute, group):
    return table[(table["attribute"] == attribute) & (table["group"] == group)].iloc[0]


def test_rates_by_hand(ten_applicants):
    df, pd_ = ten_applicants
    table = fairness_table(df, pd_, threshold=0.5, min_size=1)
    women, men = row(table, "Sex", "Female"), row(table, "Sex", "Male")
    assert women["approval_rate"] == pytest.approx(0.8) and men["approval_rate"] == pytest.approx(0.6)
    assert women["good_customers_approved"] == 1.0 and men["good_customers_approved"] == pytest.approx(2 / 3)
    assert women["defaulters_approved"] == 0.0 and men["defaulters_approved"] == pytest.approx(0.5)
    assert men["default_rate"] == pytest.approx(0.4) and men["mean_pd"] == pytest.approx(0.36)


def test_gaps_and_flags_against_the_most_approved_group(ten_applicants):
    df, pd_ = ten_applicants
    table = fairness_table(df, pd_, threshold=0.5, min_size=1)
    women, men = row(table, "Sex", "Female"), row(table, "Sex", "Male")
    assert women["reference"] and women["di_ratio"] == 1.0 and women["flags"] == ""
    assert men["di_ratio"] == pytest.approx(0.75)  # 0.6 / 0.8
    assert men["dp_gap"] == pytest.approx(0.2)
    assert men["eo_gap"] == pytest.approx(0.5)  # max(|2/3 - 1|, |1/2 - 0|)
    assert (
        "disparate impact" in men["flags"]
        and "parity gap" in men["flags"]
        and "equalised odds" in men["flags"]
    )


def test_single_group_is_its_own_reference(ten_applicants):
    df, pd_ = ten_applicants
    age = row(fairness_table(df, pd_, threshold=0.5, min_size=1), "Age band", "30-39")
    assert age["reference"] and age["n"] == 10 and age["flags"] == ""


def test_small_groups_are_not_judged(ten_applicants):
    df, pd_ = ten_applicants
    men = row(fairness_table(df, pd_, threshold=0.5, min_size=6), "Sex", "Male")
    assert not men["judged"] and np.isnan(men["di_ratio"]) and "too few to judge" in men["flags"]


def test_groups_cover_everyone_and_missing_becomes_unknown(ten_applicants):
    df, _ = ten_applicants
    df.loc[0, "CODE_GENDER"] = np.nan
    groups = audit_groups(df)
    assert set(groups) == {"Sex", "Age band", "Family status", "Education"}
    assert groups["Sex"].iloc[0] == "Unknown"
    assert all(len(labels) == len(df) for labels in groups.values())


def test_age_bands():
    df = pd.DataFrame(
        {
            "CODE_GENDER": "F",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_EDUCATION_TYPE": "x",
            "DAYS_BIRTH": [-int(a * 365.25 + 10) for a in (25, 30, 49, 60, 68)],
        }
    )
    assert list(audit_groups(df)["Age band"]) == ["Under 30", "30-39", "40-49", "60 and over", "60 and over"]


def test_published_report_hides_small_groups(ten_applicants, tmp_path):
    # groups of 5 are below MIN_GROUP_SIZE (100), so only their counts may be published
    df, pd_ = ten_applicants
    val = pd.DataFrame(
        {TARGET: [0, 1] * 50, "xgboost": np.linspace(0, 1, 100), "calibrated_pd": np.linspace(0, 1, 100)}
    )
    write_app_reports(df, pd_, 0.5, val, "xgboost", out_dir=tmp_path)
    published = pd.read_csv(tmp_path / "fairness.csv")
    small = published[published["group"].isin(["Female", "Male"])]
    assert small["n"].tolist() == [5, 5]
    assert small[["approval_rate", "default_rate", "mean_pd", "good_customers_approved"]].isna().all().all()
    assert {"roc_points.csv", "reliability.csv"} <= {p.name for p in tmp_path.iterdir()}


def test_roc_points_run_from_corner_to_corner():
    val = pd.DataFrame({TARGET: [0, 0, 1, 1], "m": [0.1, 0.4, 0.35, 0.8]})
    points = roc_points(val, ["m"])
    assert len(points) == 101
    assert points["tpr"].iloc[-1] == 1.0 and points["tpr"].is_monotonic_increasing


def test_reliability_bins_are_equal_sized():
    rng = np.random.default_rng(0)
    score = rng.random(1000)
    y = rng.random(1000) < score
    points = reliability_points(y, {"pd": score}, bins=10)
    assert len(points) == 10 and (points["n"] == 100).all()
    assert points["mean_predicted"].is_monotonic_increasing
    assert np.allclose(points["observed"], points["mean_predicted"], atol=0.12)  # calibrated by construction
