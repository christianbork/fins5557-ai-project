"""Tests for src/threshold.py: decisions at the band edges, the profit maths and the profit curve."""
import numpy as np
import pytest

from src.threshold import (
    break_even_pd,
    choose_threshold,
    decide,
    expected_profit,
    profit_curve,
    realised_profit,
)


@pytest.mark.parametrize("pd_, expected", [
    (0.0, "approve"),
    (0.1999, "approve"),
    (0.20, "refer"),  # lower edge (threshold - band) is referred
    (0.25, "refer"),
    (0.30, "refer"),  # upper edge (threshold + band) is referred
    (0.3001, "decline"),
    (1.0, "decline"),
])
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
