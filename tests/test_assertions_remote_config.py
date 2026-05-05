"""Assertion tests: Remote Config Safety (RC-01, RC-02)."""

from aurelius.common.version import PROTOCOL_VERSION, SemanticVersion
from aurelius.validator.remote_config import RemoteConfig


class TestRC01ValuesClamped:
    def test_rc01_extreme_values_clamped(self):
        """Remote config numeric values must be clamped to hardcoded bounds."""
        rc = RemoteConfig()
        # Simulate receiving extreme values from API
        raw = {
            "classifier_threshold": 0.0,  # min bound is 0.1
            "novelty_threshold": 0.0,     # min bound is 0.5
            "rate_limit_per_uid_per_tempo": 9999,  # max bound is 100
            "work_token_cost_per_unit": 0.0,       # min bound is 0.001
            "concordia_timeout_seconds": 1,         # min bound is 60
            "max_agents": 999,                      # max bound is 10
            "polling_interval_seconds": 1,           # min bound is 30
        }
        clamped = rc._clamp(raw)
        assert clamped["classifier_threshold"] >= 0.1
        assert clamped["novelty_threshold"] >= 0.5
        assert clamped["rate_limit_per_uid_per_tempo"] <= 100
        assert clamped["work_token_cost_per_unit"] >= 0.001
        assert clamped["concordia_timeout_seconds"] >= 60
        assert clamped["max_agents"] <= 10
        assert clamped["polling_interval_seconds"] >= 30

    def test_rc01_normal_values_unchanged(self):
        """Values within bounds pass through unchanged."""
        rc = RemoteConfig()
        raw = {
            "classifier_threshold": 0.5,
            "novelty_threshold": 0.92,
            "rate_limit_per_uid_per_tempo": 3,
        }
        clamped = rc._clamp(raw)
        assert clamped["classifier_threshold"] == 0.5
        assert clamped["novelty_threshold"] == 0.92
        assert clamped["rate_limit_per_uid_per_tempo"] == 3

    def test_rc01_bounds_dict_covers_all_numeric_fields(self):
        """_BOUNDS must cover all configurable numeric parameters."""
        expected_keys = {
            # Remote-only operational
            "classifier_threshold", "novelty_threshold", "rate_limit_per_uid_per_tempo",
            "work_token_cost_per_unit", "concordia_timeout_seconds", "max_agents",
            "custom_archetype_threshold_bump", "polling_interval_seconds",
            # Overridable-local numeric
            "burn_percentage", "weight_interval", "query_timeout", "container_pool_size",
            "sim_base_timeout", "sim_base_ram_mb", "sim_cpu_count",
            "max_config_size", "work_id_freshness_seconds", "min_consistency_reports",
            "consistency_floor", "queue_max_size", "queue_max_file_size_mb", "queue_max_age_seconds",
        }
        assert set(RemoteConfig._BOUNDS.keys()) == expected_keys


class TestRC02VersionGateBounded:
    """Remote version gates must not let a compromised API disqualify
    every node by pushing min_*_version to an unreachable future."""

    def test_rc02_valid_version_passes_through(self):
        raw = {"min_miner_version": "1.2.3", "min_validator_version": "1.5.0"}
        clamped = RemoteConfig._clamp(raw)
        assert clamped["min_miner_version"] == "1.2.3"
        assert clamped["min_validator_version"] == "1.5.0"

    def test_rc02_single_major_bump_allowed(self):
        local_major = SemanticVersion.parse(PROTOCOL_VERSION).major
        allowed = f"{local_major + RemoteConfig._MAX_MAJOR_DELTA}.0.0"
        clamped = RemoteConfig._clamp({"min_miner_version": allowed})
        assert clamped["min_miner_version"] == allowed

    def test_rc02_excessive_major_rejected(self):
        local_major = SemanticVersion.parse(PROTOCOL_VERSION).major
        attacker = f"{local_major + RemoteConfig._MAX_MAJOR_DELTA + 1}.0.0"
        clamped = RemoteConfig._clamp({"min_miner_version": attacker})
        assert clamped["min_miner_version"] == RemoteConfig._VERSION_FALLBACK

    def test_rc02_malformed_version_rejected(self):
        for bogus in ["", "not-a-version", "1.0", "1.0.0-beta", "99.x.0"]:
            clamped = RemoteConfig._clamp({"min_validator_version": bogus})
            assert clamped["min_validator_version"] == RemoteConfig._VERSION_FALLBACK, bogus

    def test_rc02_non_string_type_rejected(self):
        for bogus in [123, 1.5, None, ["1", "0", "0"]]:
            clamped = RemoteConfig._clamp({"min_miner_version": bogus})
            assert clamped["min_miner_version"] == RemoteConfig._VERSION_FALLBACK

    def test_rc02_applies_to_both_version_keys(self):
        assert RemoteConfig._VERSION_KEYS == frozenset(
            {"min_miner_version", "min_validator_version"}
        )
