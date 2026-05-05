import time

from aurelius.validator.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(max_submissions=3, window_seconds=60)
        assert limiter.check("miner_a") is True
        limiter.record("miner_a")
        assert limiter.check("miner_a") is True
        limiter.record("miner_a")
        assert limiter.check("miner_a") is True
        limiter.record("miner_a")
        # Now at limit
        assert limiter.check("miner_a") is False

    def test_independent_uids(self):
        limiter = RateLimiter(max_submissions=1, window_seconds=60)
        limiter.record("miner_a")
        assert limiter.check("miner_a") is False
        assert limiter.check("miner_b") is True  # Different UID

    def test_window_expiry(self):
        limiter = RateLimiter(max_submissions=1, window_seconds=0.1)
        limiter.record("miner_a")
        assert limiter.check("miner_a") is False
        time.sleep(0.15)
        assert limiter.check("miner_a") is True  # Window expired

    def test_update_config(self):
        limiter = RateLimiter(max_submissions=1, window_seconds=60)
        limiter.record("miner_a")
        assert limiter.check("miner_a") is False

        limiter.update_config(max_submissions=5, window_seconds=60)
        assert limiter.check("miner_a") is True  # Limit raised


class TestT8ClockJumpGuard:
    """T-8: if the system clock jumps backwards (NTP correction) between
    saves, persisted entries end up with wall timestamps in the future
    relative to the new wall clock. Those entries must be discarded and
    the event surfaced to operators — silently forgetting them resets
    rate-limit budgets in an invisible way.
    """

    def _persist_path(self, tmp_path):
        return str(tmp_path / "rate_limiter_state.json")

    def test_future_entries_discarded_and_warned(self, tmp_path, caplog):
        import json
        import logging

        persist = self._persist_path(tmp_path)
        # Hand-craft a persisted file with entries 30 seconds in the future —
        # what you'd see if the clock jumped backwards by 30s since last save.
        future_ts = time.time() + 30
        with open(persist, "w") as f:
            json.dump({"miner_a": [future_ts, future_ts + 1], "miner_b": [future_ts]}, f)

        with caplog.at_level(logging.INFO, logger="aurelius.validator.rate_limiter"):
            limiter = RateLimiter(max_submissions=3, window_seconds=60, persist_path=persist)

        # No entries should survive — they were all future-dated.
        assert "miner_a" not in limiter._timestamps
        assert "miner_b" not in limiter._timestamps
        # The budgets effectively reset, so both miners are under the limit again.
        assert limiter.check("miner_a") is True
        assert limiter.check("miner_b") is True
        # And the operator sees the warning.
        assert any("clock moved backwards" in r.message for r in caplog.records), [
            r.message for r in caplog.records
        ]
        # The INFO summary should count the discards.
        assert any("future-dated" in r.message for r in caplog.records)

    def test_stale_entries_dropped_without_warning(self, tmp_path, caplog):
        import json
        import logging

        persist = self._persist_path(tmp_path)
        # Entry older than the window — legitimate expiry, not a clock anomaly.
        stale_ts = time.time() - 9999
        with open(persist, "w") as f:
            json.dump({"miner_a": [stale_ts]}, f)

        with caplog.at_level(logging.WARNING, logger="aurelius.validator.rate_limiter"):
            RateLimiter(max_submissions=3, window_seconds=60, persist_path=persist)

        # No "clock moved backwards" warning for plain expiry.
        assert not any("clock moved backwards" in r.message for r in caplog.records)

    def test_fresh_entries_retained(self, tmp_path):
        import json

        persist = self._persist_path(tmp_path)
        fresh_ts = time.time() - 5  # 5 seconds ago
        with open(persist, "w") as f:
            json.dump({"miner_a": [fresh_ts]}, f)

        limiter = RateLimiter(max_submissions=1, window_seconds=60, persist_path=persist)
        # One entry survived; miner_a is at the limit.
        assert limiter.check("miner_a") is False
