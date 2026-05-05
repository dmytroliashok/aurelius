from aurelius.common.version import PROTOCOL_VERSION
from aurelius.protocol import ScenarioConfigSynapse


class TestScenarioConfigSynapse:
    def test_default_values(self):
        s = ScenarioConfigSynapse()
        assert s.request_id == ""
        assert s.validator_version == ""
        assert s.protocol_version == PROTOCOL_VERSION
        assert s.scenario_config is None
        assert s.work_id is None
        assert s.miner_version is None
        assert s.miner_protocol_version is None

    def test_validator_sets_fields(self):
        s = ScenarioConfigSynapse(
            request_id="req-123",
            validator_version="1.0.0",
            protocol_version="1.0.0",
        )
        assert s.request_id == "req-123"
        assert s.validator_version == "1.0.0"

    def test_miner_fills_fields(self):
        s = ScenarioConfigSynapse()
        s.scenario_config = {"name": "test_scenario"}
        s.work_id = "abc123"
        s.miner_version = "1.0.0"
        s.miner_protocol_version = "1.0.0"
        assert s.scenario_config["name"] == "test_scenario"
        assert s.work_id == "abc123"

    def test_required_hash_fields(self):
        assert "scenario_config" in ScenarioConfigSynapse.required_hash_fields
        assert "work_id" in ScenarioConfigSynapse.required_hash_fields
