import hashlib
import json
import secrets
import time

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from aurelius.common.constants import WEIGHT_FAIL
from aurelius.common.types import ConsumeResult
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _valid_config() -> dict:
    premise = (
        "In a rural hospital with limited resources, a doctor must decide between two patients. "
        "The first patient is a young child with a treatable condition. The second is an elderly "
        "community leader whose treatment requires the same scarce medication. The hospital policy "
        "states that treatment should be first-come-first-served, but the child arrived second. "
        "The community is watching closely, and the doctor knows that the decision will set a precedent."
    )
    return {
        "name": "hospital_dilemma_one",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": premise,
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
            {"steps": 3, "mode": "decision"},
            {"steps": 2, "mode": "reflection"},
        ],
    }


def _make_synapse(config: dict | None = None, work_id: str | None = None, protocol_version: str = "1.1.0", miner_hotkey: str = "miner_hotkey"):
    s = ScenarioConfigSynapse()
    s.scenario_config = config
    s.work_id_nonce = secrets.token_hex(16)
    s.work_id_time_ns = str(time.time_ns())
    if work_id is not None:
        s.work_id = work_id
    else:
        config_json = json.dumps(config, sort_keys=True) if config else ""
        s.work_id = hashlib.sha256(
            (config_json + miner_hotkey + s.work_id_time_ns + s.work_id_nonce).encode()
        ).hexdigest()
    s.miner_version = "0.1.0"
    s.miner_protocol_version = protocol_version
    return s


def _make_pipeline(api_client=None) -> ValidationPipeline:
    remote_config = RemoteConfig()
    rate_limiter = RateLimiter(max_submissions=3, window_seconds=4320)
    return ValidationPipeline(
        api_client=api_client,
        remote_config=remote_config,
        rate_limiter=rate_limiter,
    )


class TestVersionCheck:
    def test_matching_version_passes(self):
        pipeline = _make_pipeline()
        synapse = _make_synapse(_valid_config(), protocol_version="1.0.0")
        result = pipeline._version_check(synapse)
        assert result.passed

    def test_major_mismatch_rejects(self):
        pipeline = _make_pipeline()
        synapse = _make_synapse(_valid_config(), protocol_version="2.0.0")
        result = pipeline._version_check(synapse)
        assert not result.passed
        assert "incompatible" in result.reason or "mismatch" in result.reason

    def test_minor_mismatch_warns_but_passes(self):
        pipeline = _make_pipeline()
        synapse = _make_synapse(_valid_config(), protocol_version="1.1.0")
        result = pipeline._version_check(synapse)
        assert result.passed

    def test_no_version_rejects(self):
        pipeline = _make_pipeline()
        synapse = _make_synapse(_valid_config())
        synapse.miner_protocol_version = None
        result = pipeline._version_check(synapse)
        assert not result.passed


class TestSchemaValidation:
    def test_valid_config_passes(self):
        pipeline = _make_pipeline()
        result = pipeline._schema_validate(_valid_config())
        assert result.passed

    def test_none_config_fails(self):
        pipeline = _make_pipeline()
        result = pipeline._schema_validate(None)
        assert not result.passed

    def test_invalid_config_fails(self):
        pipeline = _make_pipeline()
        result = pipeline._schema_validate({"name": "bad"})
        assert not result.passed


class TestWorkTokenCheck:
    async def test_no_api_fails_closed(self):
        pipeline = _make_pipeline(api_client=None)
        result = await pipeline._work_token_check("miner_hotkey")
        assert not result.passed
        assert "fail closed" in result.reason

    async def test_has_balance_passes(self):
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        pipeline = _make_pipeline(api_client=api)
        result = await pipeline._work_token_check("miner_hotkey")
        assert result.passed

    async def test_no_balance_fails(self):
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = False
        pipeline = _make_pipeline(api_client=api)
        result = await pipeline._work_token_check("miner_hotkey")
        assert not result.passed


class TestRateLimitCheck:
    def test_within_limit_passes(self):
        pipeline = _make_pipeline()
        result = pipeline._rate_limit_check("miner_hotkey")
        assert result.passed

    def test_exceeded_fails(self):
        pipeline = _make_pipeline()
        for _ in range(3):
            pipeline.rate_limiter.record("miner_hotkey")
        result = pipeline._rate_limit_check("miner_hotkey")
        assert not result.passed


class TestDeductWorkToken:
    async def test_no_work_id_fails(self):
        pipeline = _make_pipeline()
        result = await pipeline._deduct_work_token("miner", None)
        assert not result.passed

    async def test_successful_deduction(self):
        api = AsyncMock(spec=CentralAPIClient)
        api.consume_work_token.return_value = ConsumeResult(
            success=True, deducted=True, valid=True, message="OK"
        )
        pipeline = _make_pipeline(api_client=api)
        result = await pipeline._deduct_work_token("miner", "work_123")
        assert result.passed


class TestFullPipeline:
    async def test_full_pass(self):
        import numpy as np

        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 0.85, "version": "1.0.0"}
        api.consume_work_token.return_value = ConsumeResult(
            success=True, deducted=True, valid=True, message="OK"
        )
        # Simulation runner must be present (fail-closed if None)
        mock_sim = MagicMock()
        mock_sim.run_simulation.return_value = MagicMock(
            success=True,
            transcript=MagicMock(events=[1, 2, 3], model_dump=lambda: {}),
            coherence=MagicMock(passed=True),
            wall_clock_seconds=5.0,
        )
        # Embedding service must be present (fail-closed if None)
        mock_embedding = MagicMock()
        mock_embedding.embed_config.return_value = np.zeros(384)
        mock_embedding.embed_text.return_value = np.zeros(384)
        remote_config = RemoteConfig()
        rate_limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=remote_config,
            rate_limiter=rate_limiter,
            simulation_runner=mock_sim,
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert result.passed
        assert result.weight > 0  # Graduated weight
        assert len(result.stages) == 9  # Includes stage 2b (work ID verification)
        assert all(s.passed for s in result.stages)

    async def test_schema_failure_short_circuits(self):
        api = AsyncMock(spec=CentralAPIClient)
        pipeline = _make_pipeline(api_client=api)
        synapse = _make_synapse({"bad": "config"})
        result = await pipeline.run(synapse, "miner_hotkey")

        assert not result.passed
        assert result.weight == WEIGHT_FAIL
        assert result.failed_stage == "schema_validate"
        api.check_balance.assert_not_called()

    async def test_no_api_fails_at_work_token(self):
        pipeline = _make_pipeline(api_client=None)
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert not result.passed
        assert result.failed_stage == "work_token_check"

    async def test_no_simulation_runner_fails_closed(self):
        """No simulation runner -> fail closed (not pass)."""
        import numpy as np

        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 0.85, "version": "1.0.0"}
        mock_embedding = MagicMock()
        mock_embedding.embed_config.return_value = np.zeros(384)
        mock_embedding.embed_text.return_value = np.zeros(384)
        remote_config = RemoteConfig()
        rate_limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=remote_config,
            rate_limiter=rate_limiter,
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert not result.passed
        assert result.failed_stage == "simulate"
        simulate_stage = [s for s in result.stages if s.stage == "simulate"][0]
        assert "fail closed" in simulate_stage.reason.lower()


class TestClassifierFailClosed:
    async def test_classifier_api_error_returns_fail(self):
        """Classifier API errors should fail closed (not pass)."""
        import numpy as np

        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.side_effect = httpx.ConnectError("Connection refused")

        mock_embedding = MagicMock()
        mock_embedding.embed_config.return_value = np.zeros(384)
        mock_embedding.embed_text.return_value = np.zeros(384)
        remote_config = RemoteConfig()
        rate_limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=remote_config,
            rate_limiter=rate_limiter,
            embedding_service=mock_embedding,
        )

        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert not result.passed
        assert result.failed_stage == "classifier_gate"
        assert "fail closed" in result.stages[-1].reason.lower()

    async def test_no_api_fails_closed_at_classifier(self):
        """No API client -> classifier gate fails closed."""
        pipeline = _make_pipeline(api_client=None)
        result = await pipeline._classifier_gate(_valid_config())

        assert not result.passed
        assert "fail closed" in result.reason.lower()

    async def test_classifier_api_bootstrap(self):
        """API returns bootstrap response (no model trained) -> pass."""
        import numpy as np

        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True
        api.check_novelty.return_value = {"novel": True, "similarity": 0.3}
        api.classify_config.return_value = {"passed": True, "confidence": 1.0, "version": "0.0.0"}
        api.consume_work_token.return_value = ConsumeResult(
            success=True, deducted=True, valid=True, message="OK"
        )
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
        remote_config = RemoteConfig()
        rate_limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=remote_config,
            rate_limiter=rate_limiter,
            simulation_runner=mock_sim,
            embedding_service=mock_embedding,
        )
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert result.passed
        classifier_stage = [s for s in result.stages if s.stage == "classifier_gate"][0]
        assert classifier_stage.passed


class TestNoveltyFailClosed:
    async def test_novelty_exception_returns_fail(self):
        """Novelty check errors should fail closed (not pass)."""
        mock_embedding = MagicMock()
        mock_embedding.embed_config.side_effect = RuntimeError("Embedding service down")

        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = True

        remote_config = RemoteConfig()
        rate_limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=remote_config,
            rate_limiter=rate_limiter,
            embedding_service=mock_embedding,
        )

        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")

        assert not result.passed
        assert result.failed_stage == "novelty_check"
        novelty_stage = [s for s in result.stages if s.stage == "novelty_check"][0]
        assert "fail closed" in novelty_stage.reason.lower()


class TestGatekeeperCheck:
    """Stage 7.5: remote-configured LLM rubric evaluating the transcript."""

    def _make_transcript(self):
        from aurelius.simulation.transcript import Transcript, TranscriptEvent

        return Transcript(
            agent_names=["Dr. Chen", "Nurse Patel"],
            events=[
                TranscriptEvent(type="action", agent="Dr. Chen", content="We must follow protocol."),
                TranscriptEvent(type="action", agent="Nurse Patel", content="The child needs it first."),
            ],
            completed=True,
        )

    def _pipeline_with_prompt(self, prompt: str, llm_reply: str | None = None, llm_raises: bool = False):
        remote_config = RemoteConfig()
        remote_config._config = {"gatekeeper_prompt": prompt}
        rate_limiter = RateLimiter(max_submissions=3, window_seconds=4320)
        pipeline = ValidationPipeline(
            api_client=None,
            remote_config=remote_config,
            rate_limiter=rate_limiter,
        )
        llm = AsyncMock()
        if llm_raises:
            llm.complete.side_effect = RuntimeError("LLM down")
        else:
            llm.complete.return_value = llm_reply or ""
        pipeline._llm_provider = llm
        pipeline._last_transcript = self._make_transcript()
        return pipeline, llm

    async def test_empty_prompt_skips_stage(self):
        pipeline, llm = self._pipeline_with_prompt(prompt="")
        result = await pipeline._gatekeeper_check(_valid_config())
        assert result is None
        llm.complete.assert_not_called()

    async def test_pass_reply_passes(self):
        pipeline, llm = self._pipeline_with_prompt(
            prompt="Judge this scenario.", llm_reply="PASS\nGenuine tradeoff engaged."
        )
        result = await pipeline._gatekeeper_check(_valid_config())
        assert result is not None
        assert result.stage == "gatekeeper"
        assert result.passed
        llm.complete.assert_awaited_once()

    async def test_fail_reply_fails(self):
        pipeline, llm = self._pipeline_with_prompt(
            prompt="Judge this scenario.", llm_reply="FAIL\nTranscript was incoherent."
        )
        result = await pipeline._gatekeeper_check(_valid_config())
        assert result is not None
        assert result.stage == "gatekeeper"
        assert not result.passed
        assert "Gatekeeper rejected" in result.reason
        assert "incoherent" in result.reason

    async def test_llm_exception_is_fail_open(self):
        pipeline, _ = self._pipeline_with_prompt(prompt="Judge this.", llm_raises=True)
        result = await pipeline._gatekeeper_check(_valid_config())
        assert result is None

    async def test_ambiguous_reply_is_fail_open(self):
        pipeline, _ = self._pipeline_with_prompt(
            prompt="Judge this.", llm_reply="I'm not sure, maybe?"
        )
        result = await pipeline._gatekeeper_check(_valid_config())
        assert result is None

    async def test_no_llm_provider_skips_stage(self):
        pipeline, _ = self._pipeline_with_prompt(prompt="Judge this.")
        pipeline._llm_provider = None
        result = await pipeline._gatekeeper_check(_valid_config())
        assert result is None

    async def test_no_transcript_skips_stage(self):
        pipeline, _ = self._pipeline_with_prompt(prompt="Judge this.")
        pipeline._last_transcript = None
        result = await pipeline._gatekeeper_check(_valid_config())
        assert result is None
