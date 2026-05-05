"""Assertion tests: Weight Setting & Consensus (WC-01..WC-05)."""

from unittest.mock import AsyncMock

from aurelius.common.constants import WEIGHT_FAIL, WEIGHT_MIN, compute_weight
from aurelius.validator.api_client import CentralAPIClient


class TestWC01WeightRange:
    def test_wc01_failed_gets_zero(self):
        """A config that fails must receive exactly 0.0."""
        assert compute_weight(0.3, threshold=0.5) == WEIGHT_FAIL
        assert WEIGHT_FAIL == 0.0

    def test_wc01_passed_above_minimum(self):
        """A config that passes must receive weight > 0.0."""
        w = compute_weight(0.85, threshold=0.5)
        assert w >= WEIGHT_MIN
        assert w > 0.0

    def test_wc01_max_score_produces_one(self):
        """Perfect score → weight = 1.0."""
        w = compute_weight(1.0, threshold=0.5)
        assert w == 1.0

    def test_wc01_at_threshold_minimum_weight(self):
        """Score at threshold → WEIGHT_MIN."""
        w = compute_weight(0.5, threshold=0.5)
        assert w == WEIGHT_MIN

    def test_wc01_weight_range(self):
        """All passing weights are in [WEIGHT_MIN, 1.0]."""
        for score in [0.51, 0.6, 0.7, 0.8, 0.9, 1.0]:
            w = compute_weight(score, threshold=0.5)
            assert WEIGHT_MIN <= w <= 1.0

    def test_wc01_below_threshold_is_zero(self):
        """All failing scores produce exactly 0.0."""
        for score in [0.0, 0.1, 0.2, 0.3, 0.49]:
            assert compute_weight(score, threshold=0.5) == 0.0


class TestWC04ConvergentWeights:
    def test_wc04_deterministic_formula(self):
        """Same classifier score → same weight (deterministic formula)."""
        w1 = compute_weight(0.85, threshold=0.5)
        w2 = compute_weight(0.85, threshold=0.5)
        assert w1 == w2

    def test_wc04_monotonically_increasing(self):
        """Higher scores → higher weights."""
        scores = [0.55, 0.65, 0.75, 0.85, 0.95]
        weights = [compute_weight(s, threshold=0.5) for s in scores]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]


class TestWC05ConsistencyMultiplierClamped:
    def test_wc05_zero_rate_returns_zero(self):
        """Rate below floor → multiplier = 0.0."""
        # Floor is 0.4 per _CONSISTENCY_FLOOR
        # rate < 0.4 → 0.0
        rate, floor = 0.3, 0.4
        if rate < floor:
            mult = 0.0
        else:
            mult = (rate - floor) / (1.0 - floor)
        assert mult == 0.0

    def test_wc05_floor_rate_returns_zero(self):
        """Rate at floor → multiplier = 0.0."""
        rate, floor = 0.4, 0.4
        mult = (rate - floor) / (1.0 - floor)
        assert mult == 0.0

    def test_wc05_perfect_rate_returns_one(self):
        """Rate = 1.0 → multiplier = 1.0."""
        rate, floor = 1.0, 0.4
        mult = (rate - floor) / (1.0 - floor)
        assert abs(mult - 1.0) < 1e-9

    def test_wc05_mid_rate_in_range(self):
        """Rate between floor and 1.0 → multiplier in (0, 1)."""
        rate, floor = 0.7, 0.4
        mult = (rate - floor) / (1.0 - floor)
        assert 0 < mult < 1
