"""Tests for RemoteConfig local/remote merge semantics.

Verifies:
- In ENVIRONMENT == "local", local Config values always win.
- In non-local environments, remote wins when present, with local fallback.
- Fallback applies when the remote key is missing or empty.
"""

from types import SimpleNamespace

import pytest

from aurelius.validator.remote_config import RemoteConfig


def _fake_local() -> SimpleNamespace:
    """Minimal double of the LocalConfig class exposing the fields we need."""
    return SimpleNamespace(
        BURN_MODE=True,
        BURN_PERCENTAGE=0.5,
        WEIGHT_INTERVAL=300,
        QUERY_TIMEOUT=30.0,
        CONTAINER_POOL_SIZE=0,
        LLM_MODEL="local-model",
        LLM_BASE_URL="https://local.example.com",
        CONCORDIA_IMAGE_NAME="local/image",
        CONCORDIA_IMAGE_DIGEST="",
        REQUIRE_IMAGE_DIGEST=False,
        SIM_NETWORK_NAME="local-net",
        SIM_BASE_TIMEOUT=600,
        SIM_BASE_RAM_MB=4096,
        SIM_CPU_COUNT=2,
        SIM_ALLOWED_LLM_HOSTS=["local.example.com"],
        SIM_DATA_DIR="",
        SIM_DATA_HOST_DIR="",
        MAX_CONFIG_SIZE=65536,
        WORK_ID_FRESHNESS_SECONDS=300,
        MIN_CONSISTENCY_REPORTS=10,
        CONSISTENCY_FLOOR=0.4,
        QUEUE_MAX_SIZE=500,
        QUEUE_MAX_FILE_SIZE_MB=50,
        QUEUE_MAX_AGE_SECONDS=12960,
    )


class TestLocalEnvironmentOverrides:
    def test_local_env_ignores_remote_values(self):
        """In ENVIRONMENT='local', local Config wins even when remote has a value."""
        rc = RemoteConfig(local=_fake_local(), environment="local")
        # Inject contradictory remote values — these must be ignored.
        rc._config = {
            "burn_mode": False,
            "burn_percentage": 0.9,
            "weight_interval": 999,
            "query_timeout": 99.0,
            "container_pool_size": 8,
            "llm_model": "remote-model",
            "max_config_size": 10_000,
        }
        assert rc.burn_mode is True
        assert rc.burn_percentage == 0.5
        assert rc.weight_interval == 300
        assert rc.query_timeout == 30.0
        assert rc.container_pool_size == 0
        assert rc.llm_model == "local-model"
        assert rc.max_config_size == 65536

    def test_local_env_sim_allowed_hosts_uses_local(self):
        rc = RemoteConfig(local=_fake_local(), environment="local")
        rc._config = {"sim_allowed_llm_hosts": "remote.example.com,other.example.com"}
        assert rc.sim_allowed_llm_hosts == ["local.example.com"]


class TestNonLocalEnvironmentOverrides:
    @pytest.mark.parametrize("env", ["testnet", "mainnet", "staging"])
    def test_remote_wins_when_set(self, env):
        rc = RemoteConfig(local=_fake_local(), environment=env)
        rc._config = {
            "burn_mode": False,
            "burn_percentage": 0.9,
            "weight_interval": 600,
            "query_timeout": 12.0,
            "container_pool_size": 4,
            "llm_model": "remote-model",
            "max_config_size": 131_072,
            "consistency_floor": 0.6,
        }
        assert rc.burn_mode is False
        assert rc.burn_percentage == 0.9
        assert rc.weight_interval == 600
        assert rc.query_timeout == 12.0
        assert rc.container_pool_size == 4
        assert rc.llm_model == "remote-model"
        assert rc.max_config_size == 131_072
        assert rc.consistency_floor == 0.6

    def test_fallback_to_local_when_remote_absent(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {}  # No remote overrides at all
        assert rc.burn_mode is True
        assert rc.burn_percentage == 0.5
        assert rc.weight_interval == 300
        assert rc.llm_model == "local-model"
        assert rc.max_config_size == 65536
        assert rc.sim_allowed_llm_hosts == ["local.example.com"]

    def test_fallback_to_local_when_remote_is_empty_string(self):
        """Empty-string remote values should fall back to local (string fields)."""
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"llm_model": "", "concordia_image_name": ""}
        assert rc.llm_model == "local-model"
        assert rc.concordia_image_name == "local/image"

    def test_sim_allowed_hosts_parses_comma_string(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"sim_allowed_llm_hosts": "a.example.com, b.example.com , c.example.com"}
        assert rc.sim_allowed_llm_hosts == ["a.example.com", "b.example.com", "c.example.com"]

    def test_sim_allowed_hosts_accepts_list(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"sim_allowed_llm_hosts": ["a.example.com", "b.example.com"]}
        assert rc.sim_allowed_llm_hosts == ["a.example.com", "b.example.com"]


class TestTypeCoercion:
    def test_int_coercion(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"weight_interval": "600"}  # String value from DB
        assert rc.weight_interval == 600
        assert isinstance(rc.weight_interval, int)

    def test_float_coercion(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"burn_percentage": "0.42"}
        assert rc.burn_percentage == 0.42
        assert isinstance(rc.burn_percentage, float)

    def test_bool_coercion_from_string(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"burn_mode": "false"}
        assert rc.burn_mode is False
        rc._config = {"burn_mode": "true"}
        assert rc.burn_mode is True
        rc._config = {"burn_mode": "1"}
        assert rc.burn_mode is True
        rc._config = {"burn_mode": "0"}
        assert rc.burn_mode is False

    def test_coercion_failure_falls_back_to_local(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"weight_interval": "not-a-number"}
        # Invalid remote value → fall back to local
        assert rc.weight_interval == 300


class TestRemoteOnlyFieldsUnchanged:
    """Remote-only fields (no local counterpart) keep their original behavior."""

    def test_classifier_threshold_from_remote(self):
        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        rc._config = {"classifier_threshold": 0.7}
        assert rc.classifier_threshold == 0.7

    def test_novelty_threshold_default_when_absent(self):
        from aurelius.common.constants import DEFAULT_NOVELTY_THRESHOLD

        rc = RemoteConfig(local=_fake_local(), environment="testnet")
        # Remote-only fields fall back to hardcoded defaults, NOT local Config.
        # The default is seeded by _defaults().
        assert rc.novelty_threshold == DEFAULT_NOVELTY_THRESHOLD

    def test_classifier_threshold_ignores_local_env(self):
        """Remote-only fields follow remote even in local env (they have no local equivalent)."""
        rc = RemoteConfig(local=_fake_local(), environment="local")
        rc._config["classifier_threshold"] = 0.65
        assert rc.classifier_threshold == 0.65
