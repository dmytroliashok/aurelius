"""Assertion tests: Central API Degraded Mode (DM-01..DM-06)."""

from unittest.mock import AsyncMock

import httpx
import pytest

from aurelius.validator.api_client import CentralAPIClient, _BalanceResponse, _ClassifierResponse, _ConsistencyResponse
from aurelius.validator.local_queue import LocalSubmissionQueue, QueuedSubmission
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _valid_config() -> dict:
    return {
        "name": "dm_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }


class TestDM01ClassificationFailsClosed:
    async def test_dm01_no_api_classifier_rejects(self):
        """When API is unreachable, classification must fail closed."""
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        result = await pipeline._classifier_gate(_valid_config())
        assert not result.passed
        assert "fail closed" in result.reason.lower()


class TestDM02WorkTokenFailsClosed:
    async def test_dm02_no_api_work_token_rejects(self):
        """In degraded mode, work-token checks must fail closed."""
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        result = await pipeline._work_token_check("miner_hotkey")
        assert not result.passed
        assert "fail closed" in result.reason.lower()

    async def test_dm02_no_api_deduction_rejects(self):
        """In degraded mode, deduction must also fail closed."""
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        result = await pipeline._deduct_work_token("miner", "work_123")
        assert not result.passed
        assert "fail closed" in result.reason.lower()


class TestDM03QueueLocallyAndDrain:
    def test_dm03_enqueue_and_drain(self):
        """Accepted configs must be queued locally and reported when API returns."""
        queue = LocalSubmissionQueue()
        sub = QueuedSubmission(
            work_id="test_wid", miner_hotkey="miner_1",
            scenario_config={"test": True}, classifier_score=0.9,
            simulation_transcript={"events": []},
        )
        queue.enqueue(sub)
        assert queue.size == 1

        drained = queue.drain(max_count=10)
        assert len(drained) == 1
        assert drained[0].work_id == "test_wid"
        assert queue.is_empty

    def test_dm03_persistence(self, tmp_path):
        """Queue persists to disk for crash recovery."""
        path = tmp_path / "queue.jsonl"
        queue1 = LocalSubmissionQueue(persist_path=str(path))
        queue1.enqueue(QueuedSubmission("wid1", "miner", {"c": 1}, None, None))
        assert path.exists()

        queue2 = LocalSubmissionQueue(persist_path=str(path))
        assert queue2.size == 1


class TestDM04DistinguishUnreachableVsError:
    async def test_dm04_http_error_vs_connection_error(self):
        """Validator must distinguish between API-unreachable and classifier-returned-error."""
        # HTTP error (API returned 500)
        api_500 = AsyncMock(spec=CentralAPIClient)
        api_500.check_balance.side_effect = httpx.HTTPStatusError(
            "Server Error", request=httpx.Request("GET", "http://test"), response=httpx.Response(500)
        )
        pipeline = ValidationPipeline(
            api_client=api_500,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        result_500 = await pipeline._work_token_check("miner")
        assert not result_500.passed
        assert "500" in result_500.reason

        # Connection error (API unreachable)
        api_conn = AsyncMock(spec=CentralAPIClient)
        api_conn.check_balance.side_effect = httpx.ConnectError("Connection refused")
        pipeline2 = ValidationPipeline(
            api_client=api_conn,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
        )
        result_conn = await pipeline2._work_token_check("miner")
        assert not result_conn.passed
        # Both fail closed but with different messages
        assert result_500.reason != result_conn.reason


class TestDM05AgeBasedExpiry:
    def test_dm05_old_submissions_discarded(self):
        """Locally queued submissions older than configurable age must be discarded."""
        queue = LocalSubmissionQueue(max_age_seconds=0.01)  # 10ms max age
        queue.enqueue(QueuedSubmission("old_wid", "miner", {}, None, None))
        import time
        time.sleep(0.02)  # Let it expire
        drained = queue.drain(max_count=10)
        assert len(drained) == 0  # Should be discarded as stale

    def test_dm05_fresh_submissions_kept(self):
        """Fresh submissions are returned by drain."""
        queue = LocalSubmissionQueue(max_age_seconds=60)
        queue.enqueue(QueuedSubmission("fresh_wid", "miner", {}, None, None))
        drained = queue.drain(max_count=10)
        assert len(drained) == 1
        assert drained[0].work_id == "fresh_wid"

    def test_dm05_mixed_age_only_fresh_returned(self):
        """Only fresh submissions are returned; stale ones are silently discarded."""
        queue = LocalSubmissionQueue(max_age_seconds=0.05)
        queue.enqueue(QueuedSubmission("old_wid", "miner", {}, None, None))
        import time
        time.sleep(0.06)
        queue.enqueue(QueuedSubmission("new_wid", "miner", {}, None, None))
        drained = queue.drain(max_count=10)
        assert len(drained) == 1
        assert drained[0].work_id == "new_wid"


class TestDM06ResponseSchemaValidated:
    def test_dm06_balance_response_model_exists(self):
        """API client validates response schemas via Pydantic models."""
        data = _BalanceResponse.model_validate({"has_balance": True})
        assert data.has_balance is True

    def test_dm06_balance_response_rejects_malformed(self):
        """Malformed responses are caught by Pydantic validation."""
        with pytest.raises(Exception):
            _BalanceResponse.model_validate({"wrong_field": 123})

    def test_dm06_classifier_response_model_exists(self):
        data = _ClassifierResponse.model_validate({"passed": True, "confidence": 0.85, "version": "1.0"})
        assert data.passed is True
        assert data.confidence == 0.85

    def test_dm06_consistency_response_model_exists(self):
        data = _ConsistencyResponse.model_validate({"agreement_rate": 0.95, "total_reports": 50})
        assert data.agreement_rate == 0.95
