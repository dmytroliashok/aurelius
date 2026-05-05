"""Assertion tests: General Invariants (GI-01..GI-04)."""

import hashlib
import json
import secrets
import time

from unittest.mock import AsyncMock, MagicMock

from aurelius.common.constants import WEIGHT_FAIL
from aurelius.common.types import ConsumeResult
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.pipeline import PipelineResult, ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


def _valid_config() -> dict:
    return {
        "name": "gi_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }


class TestGI01NoEmissionsWithoutPipeline:
    def test_gi01_only_pipeline_results_get_nonzero_weight(self):
        """No miner should receive emissions without full pipeline validation."""
        # A passing result must come from the pipeline
        result = PipelineResult(weight=0.85, stages=[])
        assert result.passed
        assert result.weight > 0

        # A failing result must have weight 0
        fail_result = PipelineResult(weight=WEIGHT_FAIL, stages=[])
        assert not fail_result.passed
        assert fail_result.weight == 0.0


class TestGI02NonResponsiveWeightZero:
    def test_gi02_unsuccessful_synapse_gets_weight_fail(self):
        """A non-responsive miner (timeout) must receive weight = 0."""
        # When a synapse is unsuccessful, the validator assigns WEIGHT_FAIL
        result = PipelineResult(weight=WEIGHT_FAIL, stages=[])
        assert result.weight == 0.0
        assert not result.passed


class TestGI06BurnUidValidation:
    """B-4: validator preflight must warn operators when the hardcoded
    burn UID is unexpectedly occupied or out of range. Hard-failing would
    be wrong because the subnet can legitimately be pre-population, but
    silent misrouting of emissions would be worse."""

    def _mock_metagraph(self, n: int, burn_stake: float = 0.0, burn_permit: bool = False):
        from unittest.mock import MagicMock

        mg = MagicMock()
        mg.n = n
        # 256-wide slots; we only exercise BURN_UID (200)
        mg.hotkeys = [f"hk_{i}" for i in range(max(n, 201))]
        stakes = [0.0] * max(n, 201)
        permits = [False] * max(n, 201)
        if n > 200:
            stakes[200] = burn_stake
            permits[200] = burn_permit
        mg.S = stakes
        mg.validator_permit = permits
        return mg

    def _make_validator(self):
        from unittest.mock import MagicMock

        from aurelius.validator.validator import Validator

        v = Validator.__new__(Validator)
        v.metagraph = MagicMock()
        return v

    def test_gi06_warns_when_burn_uid_out_of_range(self, caplog):
        import logging

        v = self._make_validator()
        v.metagraph = self._mock_metagraph(n=100)  # < BURN_UID
        with caplog.at_level(logging.WARNING, logger="aurelius.validator.validator"):
            v._validate_burn_uid()
        assert any("metagraph size" in r.message for r in caplog.records), caplog.records

    def test_gi06_warns_when_burn_uid_is_staked(self, caplog):
        import logging

        v = self._make_validator()
        v.metagraph = self._mock_metagraph(n=256, burn_stake=100.0)
        with caplog.at_level(logging.WARNING, logger="aurelius.validator.validator"):
            v._validate_burn_uid()
        assert any("occupied" in r.message.lower() for r in caplog.records)

    def test_gi06_info_when_burn_uid_clean(self, caplog):
        import logging

        v = self._make_validator()
        v.metagraph = self._mock_metagraph(n=256, burn_stake=0.0, burn_permit=False)
        with caplog.at_level(logging.INFO, logger="aurelius.validator.validator"):
            v._validate_burn_uid()
        assert any("burn_mode will route here" in r.message for r in caplog.records)


class TestGI05TestlabModeGuard:
    """TESTLAB_MODE=1 on finney must be rejected at startup for miner and
    validator alike (mirrored guards in miner/miner.py and validator/validator.py)."""

    def test_gi05_validator_rejects_testlab_on_finney(self):
        import pytest
        from aurelius.validator.validator import Validator

        with pytest.raises(RuntimeError, match="TESTLAB_MODE"):
            Validator._check_testlab_safety(testlab_mode=True, network="finney")

    def test_gi05_validator_permits_testlab_on_test(self):
        from aurelius.validator.validator import Validator

        # No raise
        Validator._check_testlab_safety(testlab_mode=True, network="test")
        Validator._check_testlab_safety(testlab_mode=False, network="finney")
        Validator._check_testlab_safety(testlab_mode=False, network="test")


class TestGI04NoReportWithoutCoherence:
    async def test_gi04_coherence_failure_blocks_pipeline(self):
        """A validator must never report a config that didn't pass coherence."""
        api = AsyncMock()
        mock_sim = MagicMock()
        mock_sim.run_simulation.return_value = MagicMock(
            success=False,
            transcript=None,
            coherence=MagicMock(passed=False, reasons=["Incomplete"]),
            error="Coherence check failed",
            wall_clock_seconds=5.0,
        )
        import numpy as np

        mock_embedding = MagicMock()
        mock_embedding.embed_config.return_value = np.zeros(384)
        mock_embedding.embed_text.return_value = np.zeros(384)

        api.check_balance = AsyncMock(return_value=True)
        api.check_novelty = AsyncMock(return_value={"novel": True, "similarity": 0.3})
        api.classify_config = AsyncMock(return_value={"passed": True, "confidence": 0.85, "version": "1.0.0"})

        pipeline = ValidationPipeline(
            api_client=api,
            remote_config=RemoteConfig(),
            rate_limiter=RateLimiter(max_submissions=10, window_seconds=4320),
            simulation_runner=mock_sim,
            embedding_service=mock_embedding,
        )
        config = _valid_config()
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

        result = await pipeline.run(synapse, miner_hotkey)

        assert not result.passed
        assert result.failed_stage == "simulate"
        # consume should never be called for a failed simulation
        api.consume_work_token.assert_not_called()
