"""End-to-end test for the full Aurelius pipeline.

Requires:
- Bittensor testnet connectivity
- Registered miner + validator wallets
- Running Central API + PostgreSQL

Run with: pytest tests/e2e/ -m e2e
"""

import pytest

from aurelius.common.schema import validate_scenario_config
from aurelius.common.version import PROTOCOL_VERSION, check_compatibility, VersionResult
from aurelius.miner.config_store import ConfigStore
from aurelius.miner.work_token import generate_work_id
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _sample_config() -> dict:
    return {
        "name": "e2e_test_scenario",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": (
            "In a rural hospital with limited resources, a doctor must decide between two patients. "
            "The first patient is a young child with a treatable condition. The second is an elderly "
            "community leader whose treatment requires the same scarce medication. The hospital policy "
            "states that treatment should be first-come-first-served, but the child arrived second. "
            "The community is watching closely, and the doctor knows the decision will set a precedent."
        ),
        "agents": [
            {
                "name": "Dr. Chen",
                "identity": "I am a surgeon with 20 years of experience in emergency medicine.",
                "goal": "I want to save the most lives while upholding hospital protocol.",
                "philosophy": "deontology",
            },
            {
                "name": "Nurse Patel",
                "identity": "I am a senior nurse who has seen the consequences of bending rules.",
                "goal": "I want to ensure patient safety and advocate for the vulnerable.",
                "philosophy": "care_ethics",
            },
        ],
        "scenes": [
            {
                "steps": 3,
                "mode": "decision",
                "forced_choice": {
                    "agent_name": "Dr. Chen",
                    "choices": [
                        "I prioritize the child despite the policy.",
                        "I follow first-come-first-served protocol.",
                    ],
                    "call_to_action": "The medication can only go to one patient. What does Dr. Chen do?",
                },
            },
            {"steps": 2, "mode": "reflection"},
        ],
    }


class TestLocalPipeline:
    """Tests that exercise the full pipeline locally without network."""

    def test_schema_validation(self):
        config = _sample_config()
        result = validate_scenario_config(config)
        assert result.valid

    def test_work_id_generation(self):
        config = _sample_config()
        result = generate_work_id(config, "test_hotkey")
        assert len(result.work_id) == 64
        assert len(result.nonce) == 32
        assert result.time_ns.isdigit()

    def test_synapse_roundtrip(self):
        config = _sample_config()
        wid_result = generate_work_id(config, "test_hotkey")
        synapse = ScenarioConfigSynapse(
            request_id="test-req",
            validator_version="0.1.0",
            protocol_version=PROTOCOL_VERSION,
        )
        synapse.scenario_config = config
        synapse.work_id = wid_result.work_id
        synapse.work_id_nonce = wid_result.nonce
        synapse.work_id_time_ns = wid_result.time_ns
        synapse.miner_version = "0.1.0"
        synapse.miner_protocol_version = PROTOCOL_VERSION

        assert synapse.scenario_config is not None
        assert synapse.work_id is not None
        assert check_compatibility(PROTOCOL_VERSION, synapse.miner_protocol_version) == VersionResult.ACCEPT

    async def test_pipeline_without_api(self):
        """Pipeline with no API — work-token check should fail closed."""
        remote_config = RemoteConfig()
        rate_limiter = RateLimiter(max_submissions=10, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=remote_config,
            rate_limiter=rate_limiter,
        )

        synapse = ScenarioConfigSynapse()
        synapse.scenario_config = _sample_config()
        synapse.work_id = "test_work_id"
        synapse.miner_protocol_version = PROTOCOL_VERSION

        result = await pipeline.run(synapse, "test_miner")
        # Should fail at work-token check (fail closed)
        assert not result.passed
        assert result.failed_stage == "work_token_check"

    def test_config_store_with_sample(self, tmp_path):
        import json

        config = _sample_config()
        (tmp_path / "test.json").write_text(json.dumps(config))

        store = ConfigStore(tmp_path)
        assert store.count == 1

        loaded = store.next()
        assert loaded["name"] == "e2e_test_scenario"


@pytest.mark.e2e
class TestTestnetPipeline:
    """Full E2E tests requiring testnet. Run with: pytest -m e2e"""

    def test_register_and_serve(self):
        """Test registering and serving on testnet.

        This test is a placeholder — actual testnet registration requires
        TAO and manual wallet setup.
        """
        pytest.skip("Requires testnet registration and funded wallet")
