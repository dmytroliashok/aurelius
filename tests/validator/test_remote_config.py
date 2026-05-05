from aurelius.common.constants import DEFAULT_MAX_AGENTS, DEFAULT_NOVELTY_THRESHOLD, DEFAULT_POLLING_INTERVAL_SECONDS
from aurelius.validator.remote_config import RemoteConfig


class TestRemoteConfig:
    def test_defaults(self):
        rc = RemoteConfig()
        assert rc.polling_interval == DEFAULT_POLLING_INTERVAL_SECONDS
        assert rc.novelty_threshold == DEFAULT_NOVELTY_THRESHOLD
        assert rc.max_agents == DEFAULT_MAX_AGENTS
        assert rc.work_token_cost == 1.0

    def test_get_with_default(self):
        rc = RemoteConfig()
        assert rc.get("nonexistent_key", "fallback") == "fallback"

    def test_min_versions(self):
        rc = RemoteConfig()
        assert rc.min_miner_version == "1.0.0"
        assert rc.min_validator_version == "1.0.0"

    def test_gatekeeper_prompt_defaults_empty(self):
        rc = RemoteConfig()
        # No remote config loaded → empty string disables the stage.
        assert rc.gatekeeper_prompt == ""

    def test_gatekeeper_prompt_from_remote(self):
        rc = RemoteConfig()
        rc._config = {"gatekeeper_prompt": "Evaluate this transcript rigorously."}
        assert rc.gatekeeper_prompt == "Evaluate this transcript rigorously."

    def test_gatekeeper_prompt_empty_remote_stays_empty(self):
        rc = RemoteConfig()
        rc._config = {"gatekeeper_prompt": ""}
        assert rc.gatekeeper_prompt == ""
