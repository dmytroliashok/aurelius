from aurelius.common.constants import WEIGHT_FAIL, WEIGHT_MIN, compute_weight

# Default classifier threshold used in tests
THRESHOLD = 0.5


class TestComputeWeight:
    def test_compute_weight_no_score(self):
        """None score (no classifier) returns WEIGHT_FAIL (fail closed)."""
        assert compute_weight(None, THRESHOLD) == WEIGHT_FAIL

    def test_compute_weight_below_threshold(self):
        """Score below threshold returns WEIGHT_FAIL (0.0)."""
        assert compute_weight(0.3, THRESHOLD) == WEIGHT_FAIL
        assert compute_weight(0.49, THRESHOLD) == WEIGHT_FAIL

    def test_compute_weight_at_threshold(self):
        """Score exactly at threshold returns WEIGHT_MIN (0.1)."""
        result = compute_weight(THRESHOLD, THRESHOLD)
        assert result == WEIGHT_MIN

    def test_compute_weight_max(self):
        """Score of 1.0 returns 1.0 (full weight)."""
        assert compute_weight(1.0, THRESHOLD) == 1.0

    def test_compute_weight_mid_range(self):
        """Score halfway between threshold and 1.0 returns ~0.5."""
        midpoint = THRESHOLD + (1.0 - THRESHOLD) / 2  # 0.75
        result = compute_weight(midpoint, THRESHOLD)
        assert abs(result - 0.5) < 0.01

    def test_compute_weight_just_above_threshold(self):
        """Score slightly above threshold returns close to WEIGHT_MIN."""
        result = compute_weight(THRESHOLD + 0.01, THRESHOLD)
        assert result >= WEIGHT_MIN
        assert result < WEIGHT_MIN + 0.05
