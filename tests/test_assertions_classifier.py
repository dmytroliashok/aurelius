"""Assertion tests: Classifier (CL-01..CL-06) — validator side."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np

from aurelius.common.constants import compute_weight
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _valid_config() -> dict:
    return {
        "name": "cl_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }


class TestCL06ResponseIncludesVersion:
    async def test_cl06_classifier_version_logged(self):
        """Classifier API response must include model version tag."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 0.85, "version": "1.2.3"}

        mock_embedding = MagicMock()
        mock_embedding.embed_config.return_value = np.zeros(384)
        mock_embedding.embed_text.return_value = np.zeros(384)

        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
            embedding_service=mock_embedding,
        )
        result = await pipeline._classifier_gate(_valid_config())
        assert result.passed
        # The pipeline stores the confidence from the response
        assert pipeline._last_classifier_score == 0.85


class TestCL10TimeoutFailClosed:
    async def test_cl10_connection_error_fails_closed(self):
        """If classifier doesn't respond, validator must treat as rejection."""
        import httpx

        api = AsyncMock(spec=CentralAPIClient)
        api.classify_config.side_effect = httpx.ConnectError("Connection refused")

        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        result = await pipeline._classifier_gate(_valid_config())
        assert not result.passed
        assert "fail closed" in result.reason.lower()

    async def test_cl10_timeout_fails_closed(self):
        """Timeout from classifier must be treated as rejection."""
        import httpx

        api = AsyncMock(spec=CentralAPIClient)
        api.classify_config.side_effect = httpx.ReadTimeout("Read timed out")

        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        result = await pipeline._classifier_gate(_valid_config())
        assert not result.passed
        assert "fail closed" in result.reason.lower()
