"""Assertion tests: Novelty / Deduplication (ND-01..ND-04)."""

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
        "name": "nd_assertion_test",
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


class TestND01SimilarityAboveThresholdRejected:
    async def test_nd01_not_novel_rejected(self):
        """A config with cosine similarity >= novelty_threshold must be rejected."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": False, "similarity": 0.95, "message": "Too similar"}

        mock_embedding = MagicMock()
        mock_embedding.embed_config.return_value = np.zeros(384)
        mock_embedding.embed_text.return_value = np.zeros(384)

        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert not result.passed
        assert result.failed_stage == "novelty_check"


class TestND02AgentsSortedAlphabetically:
    def test_nd02_embed_config_sorts_agents(self):
        """Embedding for novelty must sort agent identities alphabetically before mean-pooling."""
        # Verify the sorting happens in the code path
        from aurelius.common.embeddings import EmbeddingService

        svc = EmbeddingService()
        config_ab = {
            "premise": "A doctor faces a moral dilemma about patient care.",
            "agents": [
                {"name": "Alice", "identity": "A surgeon", "goal": "Save lives"},
                {"name": "Bob", "identity": "A nurse", "goal": "Patient safety"},
            ],
        }
        config_ba = {
            "premise": "A doctor faces a moral dilemma about patient care.",
            "agents": [
                {"name": "Bob", "identity": "A nurse", "goal": "Patient safety"},
                {"name": "Alice", "identity": "A surgeon", "goal": "Save lives"},
            ],
        }
        emb_ab = svc.embed_config(config_ab)
        emb_ba = svc.embed_config(config_ba)
        # Sorting ensures identical embeddings regardless of agent order
        np.testing.assert_array_almost_equal(emb_ab, emb_ba, decimal=5)


class TestND03PassingConfigAddedToIndex:
    async def test_nd03_novelty_add_called_on_pass(self):
        """A novel config that passes the full pipeline must be added to the dedup index."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 0.85, "version": "1.0.0"}
        api.consume_work_token.return_value = ConsumeResult(True, True, True, "OK")
        api.add_to_novelty_index.return_value = {"status": "added"}

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
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
            simulation_runner=mock_sim,
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert result.passed
        api.add_to_novelty_index.assert_called_once()
