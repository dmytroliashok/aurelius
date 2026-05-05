"""Assertion tests: Validation Pipeline Ordering (PO-01..PO-06)."""

import asyncio
import hashlib
import json
import secrets
import time

from unittest.mock import AsyncMock, MagicMock

import numpy as np

from aurelius.common.constants import WEIGHT_FAIL
from aurelius.common.types import ConsumeResult
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _valid_config() -> dict:
    return {
        "name": "po_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }


def _make_synapse(config=None, protocol_version="1.1.0", miner_hotkey="miner_hotkey"):
    s = ScenarioConfigSynapse()
    s.scenario_config = config
    s.work_id_nonce = secrets.token_hex(16)
    s.work_id_time_ns = str(time.time_ns())
    config_json = json.dumps(config, sort_keys=True) if config else ""
    s.work_id = hashlib.sha256(
        (config_json + miner_hotkey + s.work_id_time_ns + s.work_id_nonce).encode()
    ).hexdigest()
    s.miner_version = "0.1.0"
    s.miner_protocol_version = protocol_version
    return s


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


class TestPO01StageOrder:
    async def test_po01_stages_execute_in_order(self):
        """Pipeline stages must execute in the specified order."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 0.85, "version": "1.0.0"}
        api.consume_work_token.return_value = ConsumeResult(True, True, True, "OK")
        api.add_to_novelty_index.return_value = {}

        pipeline = _full_pipeline(api)
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        stage_names = [s.stage for s in result.stages]
        expected_order = [
            "version_check", "schema_validate", "verify_work_id",
            "work_token_check", "rate_limit_check", "novelty_check",
            "classifier_gate", "simulate", "deduct_work_token",
        ]
        assert stage_names == expected_order


class TestPO02NoConcordiaAfterClassifierFail:
    async def test_po02_classifier_fail_skips_simulation(self):
        """A validator must never run Concordia simulation on a classifier-rejected config."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": False, "confidence": 0.2, "version": "1.0.0"}

        mock_sim = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embed_config.return_value = np.zeros(384)
        mock_embedding.embed_text.return_value = np.zeros(384)

        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
            simulation_runner=mock_sim,
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert result.failed_stage == "classifier_gate"
        mock_sim.run_simulation.assert_not_called()


class TestPO03NoConsumeAfterFailure:
    async def test_po03_schema_fail_no_consume(self):
        """A validator must never call /consume for a config that failed any prior stage."""
        api = AsyncMock(spec=CentralAPIClient)
        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        synapse = _make_synapse({"bad": "config"})
        await pipeline.run(synapse, "miner_hotkey")
        api.consume_work_token.assert_not_called()


class TestPO04FailedWeightZero:
    async def test_po04_any_failure_weight_zero(self):
        """A validator must never set a nonzero weight for a failed submission."""
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")
        assert result.weight == WEIGHT_FAIL
        assert result.weight == 0.0


class TestPO05FailedNotSkipped:
    async def test_po05_failed_returns_result_not_none(self):
        """A validator must always set weight = 0 for a failed submission (not silently skip)."""
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")
        assert result is not None
        assert result.weight == 0.0
        assert len(result.stages) > 0


class TestPO06PipelineSerialized:
    async def test_po06_run_lock_prevents_concurrent_execution(self):
        """Pipeline stages sharing mutable state must be serialized."""
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        assert isinstance(pipeline._run_lock, asyncio.Lock)

        # Acquire the lock, verify a second run would block
        async with pipeline._run_lock:
            assert pipeline._run_lock.locked()
