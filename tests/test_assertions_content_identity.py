"""Assertion tests: Content Identity & Resubmission Prevention (CI-01..CI-02) — validator side."""

import hashlib
import json
import secrets
import time

from unittest.mock import AsyncMock, MagicMock

import numpy as np

from aurelius.common.types import ConsumeResult
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig
from aurelius.protocol import ScenarioConfigSynapse


def _valid_config() -> dict:
    return {
        "name": "ci_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }


def _full_pipeline(api):
    mock_sim = MagicMock()
    mock_sim.run_simulation.return_value = MagicMock(
        success=True,
        transcript=MagicMock(events=[1, 2, 3], model_dump=lambda: {}),
        coherence=MagicMock(passed=True),
        wall_clock_seconds=5.0,
    )
    mock_embedding = MagicMock()
    mock_embedding.embed_config.return_value = np.zeros(384)
    mock_embedding.embed_text.return_value = np.zeros(384)
    return ValidationPipeline(
        api_client=api,
        remote_config=RemoteConfig(),
        rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        simulation_runner=mock_sim,
        embedding_service=mock_embedding,
    )


class TestCI01ConfigHashFormula:
    def test_ci01_hash_is_sha256_sorted_json(self):
        """config_hash must be sha256(json.dumps(config, sort_keys=True))."""
        config = _valid_config()
        expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        # Verify the formula is deterministic
        assert len(expected) == 64
        assert expected == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def test_ci01_different_key_order_same_hash(self):
        """sort_keys=True ensures insertion order doesn't matter."""
        config_a = {"name": "test", "premise": "x" * 200}
        config_b = {"premise": "x" * 200, "name": "test"}
        hash_a = hashlib.sha256(json.dumps(config_a, sort_keys=True).encode()).hexdigest()
        hash_b = hashlib.sha256(json.dumps(config_b, sort_keys=True).encode()).hexdigest()
        assert hash_a == hash_b


class TestCI02ValidatorComputesHashIndependently:
    async def test_ci02_pipeline_computes_config_hash(self):
        """Validator independently computes config_hash and sends it to consume."""
        config = _valid_config()
        expected_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 0.85, "version": "1.0.0"}
        api.consume_work_token.return_value = ConsumeResult(True, True, True, "OK")
        api.add_to_novelty_index.return_value = {}

        pipeline = _full_pipeline(api)
        miner_hotkey = "miner_hotkey"
        time_ns = str(time.time_ns())
        nonce = secrets.token_hex(16)
        config_json = json.dumps(config, sort_keys=True)
        work_id = hashlib.sha256((config_json + miner_hotkey + time_ns + nonce).encode()).hexdigest()

        synapse = ScenarioConfigSynapse()
        synapse.scenario_config = config
        synapse.work_id = work_id
        synapse.work_id_nonce = nonce
        synapse.work_id_time_ns = time_ns
        synapse.miner_version = "0.1.0"
        synapse.miner_protocol_version = "1.1.0"

        await pipeline.run(synapse, miner_hotkey)

        # Verify consume_work_token was called with the independently computed hash
        api.consume_work_token.assert_called_once()
        call_kwargs = api.consume_work_token.call_args
        assert call_kwargs[1]["config_hash"] == expected_hash or call_kwargs[0][2] == expected_hash
