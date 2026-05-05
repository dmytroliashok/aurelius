"""Assertion tests: Rate Limiting (RL-01..RL-04)."""

import hashlib
import json
import secrets
import time

from unittest.mock import AsyncMock, MagicMock

import numpy as np

from aurelius.common.types import ConsumeResult
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _valid_config() -> dict:
    return {
        "name": "rl_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }


def _make_synapse(config=None, miner_hotkey="miner_hotkey"):
    s = ScenarioConfigSynapse()
    s.scenario_config = config
    s.work_id_nonce = secrets.token_hex(16)
    s.work_id_time_ns = str(time.time_ns())
    config_json = json.dumps(config, sort_keys=True) if config else ""
    s.work_id = hashlib.sha256(
        (config_json + miner_hotkey + s.work_id_time_ns + s.work_id_nonce).encode()
    ).hexdigest()
    s.miner_version = "0.1.0"
    s.miner_protocol_version = "1.1.0"
    return s


class TestRL01MaxSubmissionsPerTempo:
    def test_rl01_within_limit_allows(self):
        """Miner must not have more than rate_limit_per_uid_per_tempo accepted submissions."""
        limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        assert limiter.check("miner_a")

    def test_rl01_at_limit_denies(self):
        limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        for _ in range(3):
            limiter.record("miner_a")
        assert not limiter.check("miner_a")

    def test_rl01_per_miner_independent(self):
        limiter = RateLimiter(max_submissions=1, window_seconds=4320)
        limiter.record("miner_a")
        assert not limiter.check("miner_a")
        assert limiter.check("miner_b")  # Different miner is unaffected


class TestRL02RateLimitBeforeExpensiveOps:
    async def test_rl02_rate_limit_before_classifier_and_simulation(self):
        """Rate limit checks must occur before classifier and Concordia."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        mock_sim = MagicMock()
        mock_embedding = MagicMock()

        limiter = RateLimiter(max_submissions=0, window_seconds=4320)  # Always deny
        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=limiter,
            simulation_runner=mock_sim,
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert result.failed_stage == "rate_limit_check"
        api.classify_config.assert_not_called()
        mock_sim.run_simulation.assert_not_called()


class TestRL03SlotsConsumedOnlyOnFullPass:
    async def test_rl03_failure_does_not_consume_slot(self):
        """Rate limit slots consumed only for submissions that pass the full pipeline."""
        limiter = RateLimiter(max_submissions=10, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=None,  # Will fail at work_token_check
            remote_config=RemoteConfig(),
            rate_limiter=limiter,
        )
        synapse = _make_synapse(_valid_config())
        await pipeline.run(synapse, "miner_hotkey")

        # Rate limiter should NOT have recorded usage for this failed submission
        assert limiter.check("miner_hotkey")  # Still has capacity

    async def test_rl03_success_consumes_slot(self):
        """Full pipeline pass → rate limit slot consumed."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 0.85, "version": "1.0.0"}
        api.consume_work_token.return_value = ConsumeResult(True, True, True, "OK")
        api.add_to_novelty_index.return_value = {}

        limiter = RateLimiter(max_submissions=1, window_seconds=4320)
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

        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=limiter,
            simulation_runner=mock_sim,
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert result.passed
        assert not limiter.check("miner_hotkey")  # Slot consumed
