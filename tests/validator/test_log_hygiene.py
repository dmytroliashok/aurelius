"""Tests for validator log output hygiene (T-5, T-10, httpx)."""

import logging
import re

import pytest

from aurelius.validator.validator import (
    _configure_logging,
    _fingerprint_secret,
    _is_weights_rate_limit,
)


class TestT5IsWeightsRateLimit:
    """T-5: subtensor rate-limit rejections must be identifiable so the
    validator can demote them from WARNING to DEBUG."""

    @pytest.mark.parametrize(
        "message",
        [
            None,
            "",
            "None",
            "none",
            " None ",
            "RateLimit exceeded",
            "subtensor rate limit",
            "weights rate-limit hit",
        ],
    )
    def test_rate_limit_messages_detected(self, message):
        assert _is_weights_rate_limit(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Permission denied",
            "Custom error: something broke",
            "Insufficient stake",
            "Invalid signature",
            "Success",
        ],
    )
    def test_real_failures_not_treated_as_rate_limit(self, message):
        assert _is_weights_rate_limit(message) is False


class TestT10FingerprintSecret:
    """T-10: the startup banner must not contain any substring of an API
    key. A short sha256 fingerprint lets operators confirm two boots use
    the same secret without leaking material that maps back to a breach
    dump."""

    def test_empty_secret_renders_as_not_set(self):
        assert _fingerprint_secret("") == "(not set)"
        assert _fingerprint_secret(None) == "(not set)"  # type: ignore[arg-type]

    def test_fingerprint_shape(self):
        fp = _fingerprint_secret("sk-fake1234567890abcdefhelloworld-4a97")
        assert fp.startswith("sha256:")
        assert re.fullmatch(r"sha256:[0-9a-f]{8}", fp)

    def test_fingerprint_does_not_contain_secret_material(self):
        """Core invariant: no substring of the secret appears in the output."""
        secret = "sk-abcdef1234567890-real-looking-api-key-value"
        fp = _fingerprint_secret(secret)
        # Every 4-char window of the secret must be absent from the fingerprint.
        for i in range(len(secret) - 3):
            assert secret[i : i + 4] not in fp, (
                f"fingerprint {fp!r} leaks substring {secret[i : i + 4]!r}"
            )

    def test_fingerprint_is_deterministic(self):
        assert _fingerprint_secret("abc") == _fingerprint_secret("abc")

    def test_fingerprint_distinguishes_different_secrets(self):
        assert _fingerprint_secret("one") != _fingerprint_secret("two")


class TestH5ClockDriftPeriodic:
    """H-5: the startup clock-drift check runs once; mid-run NTP failures
    must also surface. `_tick_clock_drift_check` throttles calls to at most
    once per CLOCK_DRIFT_CHECK_INTERVAL and delegates to the same
    `_check_clock_drift` method used at startup.
    """

    def _make(self):
        from unittest.mock import AsyncMock, MagicMock

        from aurelius.validator.validator import Validator

        v = Validator.__new__(Validator)
        # Avoid real Bittensor / API side-effects in the test.
        v._check_clock_drift = AsyncMock()
        # Far enough in the past that the first tick always fires — monotonic
        # can be small on fresh CI boots, so 0.0 isn't guaranteed-stale.
        v._last_clock_drift_check = -1e9
        return v

    async def test_first_tick_runs_the_check(self):
        v = self._make()
        await v._tick_clock_drift_check()
        v._check_clock_drift.assert_awaited_once()

    async def test_second_tick_inside_interval_is_throttled(self):
        v = self._make()
        await v._tick_clock_drift_check()
        await v._tick_clock_drift_check()
        # Only the first tick actually ran the check.
        assert v._check_clock_drift.await_count == 1

    async def test_fatal_drift_propagates_systemexit(self):
        """If the underlying check SystemExits on catastrophic drift, the
        tick must not swallow it — operators rely on the container restart
        + fail-fast behaviour."""
        import pytest

        v = self._make()
        v._check_clock_drift.side_effect = SystemExit("clock too far")
        with pytest.raises(SystemExit):
            await v._tick_clock_drift_check()

    async def test_other_exceptions_are_logged_not_raised(self, caplog):
        """API unavailable should not crash the validator — the next tick
        tries again. Logged at WARNING."""
        import logging

        v = self._make()
        v._check_clock_drift.side_effect = RuntimeError("API down")
        with caplog.at_level(logging.WARNING, logger="aurelius.validator.validator"):
            await v._tick_clock_drift_check()
        assert any(
            "Periodic clock-drift check failed" in r.message for r in caplog.records
        )


class TestHttpxLoggerQuieted:
    """Operator logs get one INFO line per HTTP call from httpx. Quiet the
    libraries that emit per-request chatter so only non-2xx surfaces."""

    def test_configure_logging_sets_httpx_to_warning(self, monkeypatch):
        # Reset prior state: httpx may have been set to any level by a
        # previous test run or import side-effect.
        logging.getLogger("httpx").setLevel(logging.NOTSET)
        logging.getLogger("httpcore").setLevel(logging.NOTSET)

        # Point LOG_FORMAT at text so we take the non-json branch
        # deterministically.
        from aurelius.validator import validator as v

        monkeypatch.setattr(v.Config, "LOG_FORMAT", "text")
        _configure_logging()

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
