"""Assertion tests: Work-Token System (WT-01..WT-14) — validator side."""

import hashlib
import json
import re
import secrets
import time

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from aurelius.common.constants import WEIGHT_FAIL
from aurelius.common.types import ConsumeResult
from aurelius.miner.work_token import generate_work_id, recompute_work_id
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _valid_config() -> dict:
    return {
        "name": "wt_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }


def _make_synapse(config=None, work_id=None, nonce=None, time_ns=None, protocol_version="1.1.0", miner_hotkey="miner_hotkey"):
    s = ScenarioConfigSynapse()
    s.scenario_config = config
    s.work_id_nonce = nonce or secrets.token_hex(16)
    s.work_id_time_ns = time_ns or str(time.time_ns())
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


def _make_pipeline(api_client=None):
    return ValidationPipeline(
        api_client=api_client,
        remote_config=RemoteConfig(),
        rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
    )


class TestWT03PipelineFailsOnNoBalance:
    async def test_wt03_pipeline_fails_on_no_balance(self):
        """A submission must not pass the full pipeline if the miner's balance check returns false."""
        api = AsyncMock(spec=CentralAPIClient)
        api.check_balance.return_value = False
        pipeline = _make_pipeline(api_client=api)
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")
        assert not result.passed
        assert result.weight == WEIGHT_FAIL
        assert result.failed_stage == "work_token_check"


class TestWT04APIUnreachableFailsClosed:
    async def test_wt04_api_unreachable_fails_closed(self):
        """When the Central API is unreachable, work-token checks must fail closed."""
        pipeline = _make_pipeline(api_client=None)
        synapse = _make_synapse(_valid_config())
        result = await pipeline.run(synapse, "miner_hotkey")
        assert not result.passed
        assert result.failed_stage == "work_token_check"
        stage = [s for s in result.stages if s.stage == "work_token_check"][0]
        assert "fail closed" in stage.reason.lower()


class TestWT11WorkIdRecomputation:
    async def test_wt11_mismatch_rejected(self):
        """Validator must recompute work ID and reject if it doesn't match."""
        config = _valid_config()
        hotkey = "5FakeHotkey" * 4
        result = generate_work_id(config, hotkey)
        synapse = _make_synapse(config, work_id="b" * 64, nonce=result.nonce, time_ns=result.time_ns)
        pipeline = _make_pipeline()
        stage_result = pipeline._verify_work_id(synapse, hotkey)
        assert not stage_result.passed
        assert "mismatch" in stage_result.reason.lower()

    async def test_wt11_correct_work_id_passes(self):
        """Validator recomputes and accepts matching work ID."""
        config = _valid_config()
        hotkey = "5FakeHotkey" * 4
        result = generate_work_id(config, hotkey)
        synapse = _make_synapse(config, work_id=result.work_id, nonce=result.nonce, time_ns=result.time_ns)
        pipeline = _make_pipeline()
        stage_result = pipeline._verify_work_id(synapse, hotkey)
        assert stage_result.passed


class TestWT12FreshnessWindow:
    async def test_wt12_stale_time_ns_rejected(self):
        """A work ID's time_ns must be within a configurable freshness window."""
        config = _valid_config()
        hotkey = "5FakeHotkey" * 4
        stale_time = str(1)  # epoch nanoseconds = way in the past
        nonce = "a" * 32
        config_json = json.dumps(config, sort_keys=True)
        payload = config_json + hotkey + stale_time + nonce
        work_id = hashlib.sha256(payload.encode()).hexdigest()
        synapse = _make_synapse(config, work_id=work_id, nonce=nonce, time_ns=stale_time)
        pipeline = _make_pipeline()
        stage_result = pipeline._verify_work_id(synapse, hotkey)
        assert not stage_result.passed
        assert "stale" in stage_result.reason.lower()

    async def test_wt12_fresh_time_ns_accepted(self):
        """A work ID with current time_ns passes the freshness check."""
        import time
        config = _valid_config()
        hotkey = "5FakeHotkey" * 4
        fresh_time = str(time.time_ns())
        nonce = "a" * 32
        config_json = json.dumps(config, sort_keys=True)
        payload = config_json + hotkey + fresh_time + nonce
        work_id = hashlib.sha256(payload.encode()).hexdigest()
        synapse = _make_synapse(config, work_id=work_id, nonce=nonce, time_ns=fresh_time)
        pipeline = _make_pipeline()
        stage_result = pipeline._verify_work_id(synapse, hotkey)
        assert stage_result.passed


class TestWT13BackwardCompatRemoved:
    async def test_wt13_no_nonce_rejected(self):
        """Miners without nonce/time_ns must be rejected (backward compat removed in v1.1.0)."""
        synapse = ScenarioConfigSynapse()
        synapse.scenario_config = _valid_config()
        synapse.work_id = "a" * 64
        synapse.work_id_nonce = None
        synapse.work_id_time_ns = None
        synapse.miner_version = "0.1.0"
        synapse.miner_protocol_version = "1.1.0"
        pipeline = _make_pipeline()
        stage_result = pipeline._verify_work_id(synapse, "miner_hotkey")
        assert not stage_result.passed
        assert "required" in stage_result.reason.lower()


class TestWT14CryptographicNonce:
    def test_wt14_nonce_is_hex(self):
        """Work ID generation must include a cryptographically random nonce."""
        result = generate_work_id({"test": True}, "hotkey")
        assert re.match(r"^[a-f0-9]{32}$", result.nonce)

    def test_wt14_nonce_unique_across_calls(self):
        """Nonce ensures uniqueness even for the same config + hotkey."""
        nonces = {generate_work_id({"test": True}, "hotkey").nonce for _ in range(100)}
        assert len(nonces) == 100

    def test_wt14_work_id_unique_per_call(self):
        """Same config → different work_ids due to nonce."""
        ids = {generate_work_id({"test": True}, "hotkey").work_id for _ in range(50)}
        assert len(ids) == 50

    def test_wt14_recompute_matches(self):
        """Recomputation with same inputs produces same work_id."""
        config = _valid_config()
        result = generate_work_id(config, "hotkey")
        recomputed = recompute_work_id(config, "hotkey", result.time_ns, result.nonce)
        assert recomputed == result.work_id
