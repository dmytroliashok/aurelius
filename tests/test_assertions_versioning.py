"""Assertion tests: Versioning & Upgrades (VU-01..VU-05)."""

import pytest

from aurelius.common.version import PROTOCOL_VERSION, VersionResult, check_compatibility


class TestVU04ValidatorBelowMinRefusesWeights:
    def test_vu04_major_mismatch_rejects(self):
        """A validator below min_validator_version must refuse to set weights."""
        # If min_validator_version is "2.0.0" and our PROTOCOL_VERSION is "1.0.0" → REJECT
        result = check_compatibility(PROTOCOL_VERSION, "2.0.0")
        assert result == VersionResult.REJECT

    def test_vu04_compatible_version_accepts(self):
        """Validator at or above min version can set weights."""
        result = check_compatibility(PROTOCOL_VERSION, PROTOCOL_VERSION)
        assert result == VersionResult.ACCEPT


class TestVU05MaxAgentsIncreaseLogged:
    def test_vu05_max_agents_bounds_enforced(self):
        """max_agents is bounded by remote config safety (RC-01) to [2, 10]."""
        from aurelius.validator.remote_config import RemoteConfig

        rc = RemoteConfig()
        clamped = rc._clamp({"max_agents": 999})
        assert clamped["max_agents"] <= 10

        clamped = rc._clamp({"max_agents": 0})
        assert clamped["max_agents"] >= 2
