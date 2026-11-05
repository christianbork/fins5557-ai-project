"""Tests for src/threshold.py: decisions at the band edges, the profit maths and the profit curve."""

import numpy as np
import pytest

from src.threshold import (
    break_even_pd,
    choose_cutoffs,
    choose_threshold,
    compare_thresholds,
    cost_ratio_threshold,
    decide,
    expected_profit,
    misclassification_cost,
    profit_curve,
    realised_profit,
)


@pytest.mark.parametrize(
    "pd_, expected",
    [
        (0.0, "approve"),
        (0.1999, "approve"),
        (0.20, "refer"),  # lower edge (threshold - band) is referred
        (0.25, "refer"),
        (0.30, "refer"),  # upper edge (threshold + band) is referred
        (0.3001, "decline"),
        (1.0, "decline"),
    ],
)
def test_decide_band_edges(pd_, expected):
    assert decide(pd_, threshold=0.25, band=0.05) == expected


def test_decide_vectorised_and_rejects_nan():
    out = decide(np.array([0.1, 0.25, 0.9]), threshold=0.25, band=0.05)
    assert list(out) == ["approve", "refer", "decline"]
    with pytest.raises(ValueError, match="missing"):
        decide([0.1, np.nan], threshold=0.25)


def test_break_even_pd():
    # At PD = margin / (margin + LGD), expected profit per customer is zero.
    pd_ = break_even_pd(lgd=0.6, margin=0.12)
    assert pd_ == pytest.approx(0.12 / 0.72)
    assert (1 - pd_) * 0.12 - pd_ * 0.6 == pytest.approx(0)


def test_profit_hand_calculation():
    pds = [0.1, 0.5]
    y = [0, 1]
    # threshold 0.3 approves only the first customer (EAD 100): repaid, earns 0.12 x 100.
    assert realised_profit(pds, y, 0.3, lgd=0.6, margin=0.12, ead=[100, 100]) == pytest.approx(12)
    # threshold 0.6 also approves the second, who defaults and loses 0.6 x 100.
    assert realised_profit(pds, y, 0.6, lgd=0.6, margin=0.12, ead=[100, 100]) == pytest.approx(-48)
    # expected profit uses the PDs instead: 100 x (0.9 x 0.12 - 0.1 x 0.6) = 4.8
    assert expected_profit(pds, 0.3, lgd=0.6, margin=0.12, ead=100) == pytest.approx(4.8)


def test_profit_curve_has_interior_maximum():
    """With informative, calibrated PDs, the best cut-off should be neither 'approve nobody' nor
    'approve everybody', and should sit near the break-even PD."""
    rng = np.random.default_rng(0)
    pds = rng.beta(2, 7, 20_000)  # mean ~22%: a test distribution, not the real data
    y = (rng.random(pds.size) < pds).astype(int)  # calibrated by construction
    curve = profit_curve(pds, y, ead=1.0)
    best = choose_threshold(curve)
    assert curve["threshold"].min() < best < curve["threshold"].max()
    assert curve["realised_profit"].max() > 0
    assert curve["realised_profit"].max() > curve["realised_profit"].iloc[-1]  # beats approving almost all
    assert abs(best - break_even_pd()) < 0.03


def test_calibrate_and_set_threshold_result(trained):
    """End to end on fake data: a method is chosen, and every test applicant gets a decision."""
    decision = trained[1]
    assert 0 < decision["threshold"] < 1
    assert decision["method"] in decision["brier"]
    assert set(decision["test"]["decision_shares"]) == {"approve", "refer", "decline"}
    assert sum(decision["test"]["decision_shares"].values()) == pytest.approx(1)


def test_threshold_comparison_reports_all_three_rules(trained):
    comparison = trained[1]["threshold_comparison"]
    assert list(comparison.index) == ["profit", "cost_ratio", "balanced_accuracy"]
    assert comparison.loc[trained[1]["decision_rule"], "cutoff"] == trained[1]["threshold"]
    assert {"approval_rate", "tpr", "tnr", "precision", "balanced_accuracy", "realised_profit"} <= set(
        comparison
    )


def test_balanced_accuracy_cut_off_declines_more_when_defaulters_are_rare():
    """At an 8% default rate balanced accuracy treats a missed defaulter as ~11x a lost good customer,
    the profit rule (LGD 60%, margin 15%) as 4x, so the balanced-accuracy cut-off is lower."""
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.08).astype(int)
    pd_ = np.clip(rng.beta(2, 20, 20_000) + 0.12 * y, 0, 1)  # PDs that rank defaulters higher
    cutoffs = choose_cutoffs(pd_, y, 1.0)
    table = compare_thresholds(cutoffs, pd_, y, 1.0)
    assert table.loc["balanced_accuracy", "cutoff"] < table.loc["profit", "cutoff"]
    assert table.loc["balanced_accuracy", "approval_rate"] < table.loc["profit", "approval_rate"]
    assert table.loc["balanced_accuracy", "tpr"] > table.loc["profit", "tpr"]
    assert table.loc["profit", "realised_profit"] >= table.loc["balanced_accuracy", "realised_profit"]


def test_cost_ratio_cut_off_follows_from_the_ratio():
    """Approve while PD x 5 < 1 - PD: below 1/6 at Hofmann's (1994) 5 : 1 cost matrix."""
    assert cost_ratio_threshold(5.0) == pytest.approx(1 / 6, abs=1e-4)
    assert cost_ratio_threshold(4.0) == pytest.approx(0.2)  # LGD 60% / margin 15%: the profit break-even


def test_misclassification_cost_counts_both_errors():
    pd_ = np.array([0.05, 0.10, 0.30, 0.40])
    y = np.array([1, 0, 0, 1])  # cut-off 0.2: one approved defaulter (5), one declined good customer (1)
    assert misclassification_cost(pd_, y, 0.2, ratio=5.0) == 6.0


def test_comparison_reports_every_rule_with_shares():
    rng = np.random.default_rng(2)
    y = (rng.random(20_000) < 0.08).astype(int)
    pd_ = np.clip(rng.beta(2, 20, 20_000) + 0.12 * y, 0, 1)
    table = compare_thresholds(choose_cutoffs(pd_, y, 1.0), pd_, y, 1.0)
    assert table.loc["cost_ratio", "cutoff"] == pytest.approx(1 / 6, abs=1e-4)
    for code in ("profit", "cost_ratio", "balanced_accuracy"):  # a lower cut-off approves fewer people
        assert table.loc[code, "approval_rate"] == pytest.approx(np.mean(pd_ < table.loc[code, "cutoff"]))
    shares = table[["approve_share", "refer_share", "decline_share"]].sum(axis=1)
    assert np.allclose(shares.astype(float), 1.0)


def test_display_margin_follows_the_decision_rule():
    """Expected profit on the screening page turns negative exactly at the rule's cut-off."""
    from src.threshold import display_margin

    meta = {"assumptions": {"lgd": 0.6, "margin": 0.15}, "decision_rule": "cost_ratio", "cost_ratio": 5.0}
    m = display_margin(meta)
    assert m == pytest.approx(0.12)
    assert m / (m + 0.6) == pytest.approx(1 / 6)  # break-even PD = the 16.7% cut-off
    assert display_margin({**meta, "decision_rule": "profit"}) == 0.15
