"""Multi-stage validation pipeline for miner submissions.

Each stage returns a StageResult. The pipeline short-circuits on failure.
Async stages are used for network I/O (API calls); CPU-bound work
(simulation) is run in a thread executor.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass

import httpx

from aurelius.common.constants import WEIGHT_FAIL, compute_weight
from aurelius.common.schema import validate_scenario_config
from aurelius.common.types import ScenarioConfig
from aurelius.common.version import PROTOCOL_VERSION, VersionResult, check_compatibility
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.simulation.transcript import Transcript
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    passed: bool
    reason: str
    stage: str


@dataclass
class PipelineResult:
    weight: float
    stages: list[StageResult]
    work_id: str | None = None
    miner_hotkey: str | None = None
    scenario_config: dict | None = None
    classifier_score: float | None = None
    simulation_transcript: dict | None = None
    # Block at which the validator recorded this result into self.results.
    # Used for the result-retention TTL so a passing miner's vote survives
    # subsequent burn-only cycles long enough to reach the next chain tempo
    # boundary. Stamped by Validator._record_result, not by the pipeline.
    recorded_at_block: int | None = None

    @property
    def passed(self) -> bool:
        return self.weight > 0

    @property
    def failed_stage(self) -> str | None:
        for s in self.stages:
            if not s.passed:
                return s.stage
        return None


class ValidationPipeline:
    def __init__(
        self,
        api_client: CentralAPIClient | None,
        remote_config: RemoteConfig,
        rate_limiter: RateLimiter,
        validator_version: str = "0.1.0",
        embedding_service=None,
        simulation_runner=None,
        llm_provider=None,
    ):
        self.api_client = api_client
        self.remote_config = remote_config
        self.rate_limiter = rate_limiter
        self.validator_version = validator_version
        self._embedding_service = embedding_service
        self._simulation_runner = simulation_runner
        self._llm_provider = llm_provider  # For semantic coherence checks
        self._run_lock = asyncio.Lock()
        self._last_pooled_embedding: list[float] | None = None

    async def run(
        self,
        synapse: ScenarioConfigSynapse,
        miner_hotkey: str,
        anchor_ns: int | None = None,
    ) -> PipelineResult:
        """Run the full validation pipeline on a miner's response.

        VP6: Not safe for concurrent use on the same instance — uses a lock
        to protect per-run mutable state (_last_transcript, _last_classifier_score,
        _parsed_config, _last_pooled_embedding) from interleaving.

        Cross-validator serialization is handled by the Central API: rate limits,
        novelty index, and work-token consume all use database-level atomicity.
        Each validator runs a single pipeline instance per hotkey by design.

        ``anchor_ns`` is the freshness reference point for verify_work_id; the
        validator passes the same value for every miner in a cycle so drift is
        measured against cycle start, not per-stage execution time. When None,
        falls back to time.time_ns() at stage execution (legacy behaviour).
        """
        async with self._run_lock:
            return await self._run_locked(synapse, miner_hotkey, anchor_ns)

    async def _run_locked(
        self,
        synapse: ScenarioConfigSynapse,
        miner_hotkey: str,
        anchor_ns: int | None = None,
    ) -> PipelineResult:
        """Pipeline implementation (must be called under _run_lock)."""
        stages: list[StageResult] = []
        config = synapse.scenario_config
        work_id = synapse.work_id
        self._last_transcript: Transcript | None = None
        self._last_classifier_score: float | None = None
        self._parsed_config: ScenarioConfig | None = None
        self._last_pooled_embedding: list[float] | None = None

        # Stage 1: Version check
        result = self._version_check(synapse)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 2: Schema validation
        result = self._schema_validate(config)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 2b: Work ID verification (recompute from config + hotkey + nonce)
        result = self._verify_work_id(synapse, miner_hotkey, anchor_ns)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 3: Work-token balance check
        result = await self._work_token_check(miner_hotkey)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 4: Rate limit check
        result = self._rate_limit_check(miner_hotkey)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 5: Novelty check
        result = await self._novelty_check(config)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 6: Classifier gate
        result = await self._classifier_gate(config)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 7: Concordia simulation (blocking — run in thread)
        result = await self._simulate(config)
        stages.append(result)
        if not result.passed:
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 7.5: Gatekeeper — LLM rubric served by the Central API decides
        # whether the transcript represents a valid moral-reasoning outcome.
        # Rubric is remote-controlled so operators tune it without re-deploying
        # validators and so miners can't optimize against a leaked prompt.
        gatekeeper_result = await self._gatekeeper_check(config)
        if gatekeeper_result is not None:
            stages.append(gatekeeper_result)
            if not gatekeeper_result.passed:
                return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 7b: Semantic coherence check (controlled by remote config)
        if self._llm_provider and self._last_transcript and config:
            if self.remote_config.semantic_coherence_enabled:
                semantic_result = await self._semantic_coherence_check(config)
                if semantic_result:
                    stages.append(semantic_result)
                    if not semantic_result.passed:
                        return PipelineResult(weight=WEIGHT_FAIL, stages=stages)
            else:
                logger.info("Semantic coherence check disabled via remote config")

        # Stage 8 needs the config hash both for the tamper check in the
        # consume call and for the PO-07 rollback path if that call fails.
        cfg_hash = ""
        if config:
            cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

        # Stage 7c: Add to novelty index BEFORE token deduction to prevent
        # silent embedding loss that would allow duplicate resubmissions (CS-C2).
        # The config_hash is passed so the add can be rolled back via
        # /novelty/remove if stage 8 fails (PO-07).
        if self.api_client and self._last_pooled_embedding is not None:
            novelty_indexed = False
            for attempt in range(3):
                try:
                    await self.api_client.add_to_novelty_index(
                        self._last_pooled_embedding,
                        config_hash=cfg_hash or None,
                    )
                    novelty_indexed = True
                    break
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(1.0 * (2**attempt))
                    else:
                        logger.error(
                            "Failed to add to novelty index after 3 attempts — rejecting to prevent duplicate bypass"
                        )
            if not novelty_indexed:
                stages.append(
                    StageResult(
                        passed=False,
                        reason="Failed to index embedding in novelty store (fail closed to prevent duplicates)",
                        stage="novelty_index_add",
                    )
                )
                return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Stage 8: Work-token deduction (with config hash for tamper detection)
        work_id_signature = synapse.work_id_signature or ""
        result = await self._deduct_work_token(
            miner_hotkey,
            work_id,
            config_hash=cfg_hash,
            work_id_signature=work_id_signature,
        )
        stages.append(result)
        if not result.passed:
            if self._last_pooled_embedding is not None:
                logger.warning(
                    "novelty_burn | consume failed after novelty add; rolling back "
                    "(miner=%s work_id=%s config_hash=%s reason=%s)",
                    miner_hotkey,
                    work_id,
                    cfg_hash or "(none)",
                    result.reason,
                )
                if cfg_hash and self.api_client:
                    try:
                        await self.api_client.remove_from_novelty_index(cfg_hash)
                    except Exception:
                        logger.exception(
                            "novelty rollback failed — embedding remains in the index; "
                            "miner may be penalized on resubmission"
                        )
            return PipelineResult(weight=WEIGHT_FAIL, stages=stages)

        # Record rate limit usage
        self.rate_limiter.record(miner_hotkey)

        # Capture transcript from simulation if available
        transcript_dict = None
        if self._last_transcript:
            transcript_dict = self._last_transcript.model_dump()

        # Graduated weight based on classifier confidence
        # Use the same threshold the classifier gate used (including custom archetype bump)
        threshold = self.remote_config.classifier_threshold
        archetype = config.get("tension_archetype", "") if config else ""
        if archetype == "custom":
            bump = self.remote_config.custom_archetype_threshold_bump
            threshold = min(threshold + bump, 0.99)
        weight = compute_weight(self._last_classifier_score, threshold)

        return PipelineResult(
            weight=weight,
            stages=stages,
            work_id=work_id,
            miner_hotkey=miner_hotkey,
            scenario_config=config,
            classifier_score=self._last_classifier_score,
            simulation_transcript=transcript_dict,
        )

    # --- Stage implementations ---

    def _version_check(self, synapse: ScenarioConfigSynapse) -> StageResult:
        miner_version = synapse.miner_protocol_version
        if not miner_version:
            return StageResult(passed=False, reason="No miner_protocol_version", stage="version_check")

        try:
            compat = check_compatibility(PROTOCOL_VERSION, miner_version)
        except (ValueError, TypeError):
            return StageResult(
                passed=False,
                reason=f"Invalid semver: {miner_version!r}",
                stage="version_check",
            )

        min_version = self.remote_config.min_miner_version
        try:
            min_compat = check_compatibility(min_version, miner_version)
        except (ValueError, TypeError):
            return StageResult(
                passed=False,
                reason=f"Invalid semver: {miner_version!r}",
                stage="version_check",
            )
        if min_compat == VersionResult.REJECT:
            return StageResult(
                passed=False,
                reason=f"Miner protocol {miner_version} incompatible with minimum {min_version}",
                stage="version_check",
            )

        if compat == VersionResult.REJECT:
            return StageResult(
                passed=False,
                reason=f"Major protocol mismatch: local={PROTOCOL_VERSION} miner={miner_version}",
                stage="version_check",
            )

        if compat == VersionResult.WARN:
            logger.warning("Minor protocol mismatch: local=%s miner=%s", PROTOCOL_VERSION, miner_version)

        return StageResult(passed=True, reason="OK", stage="version_check")

    # Maximum serialized config size (bytes) to prevent memory exhaustion during embedding
    # Configurable via MAX_CONFIG_SIZE env var (default: 65536)

    def _schema_validate(self, config: dict | None) -> StageResult:
        if config is None:
            return StageResult(passed=False, reason="No scenario_config", stage="schema_validate")

        config_size = len(json.dumps(config, sort_keys=True).encode())
        max_config_size = self.remote_config.max_config_size
        if config_size > max_config_size:
            return StageResult(
                passed=False,
                reason=f"Config too large ({config_size} bytes > {max_config_size} limit)",
                stage="schema_validate",
            )

        result = validate_scenario_config(config, max_agents=self.remote_config.max_agents)
        if not result.valid:
            return StageResult(
                passed=False,
                reason=f"Schema validation failed: {result.errors[:3]}",
                stage="schema_validate",
            )

        try:
            self._parsed_config = ScenarioConfig(**config)
        except Exception as e:
            return StageResult(
                passed=False,
                reason=f"Pydantic validation failed: {e}",
                stage="schema_validate",
            )

        return StageResult(passed=True, reason="OK", stage="schema_validate")

    # Work ID freshness window (configurable via WORK_ID_FRESHNESS_SECONDS env var)

    def _verify_work_id(
        self,
        synapse: ScenarioConfigSynapse,
        miner_hotkey: str,
        anchor_ns: int | None = None,
    ) -> StageResult:
        """Recompute the work ID from config + hotkey + nonce and verify it matches."""
        if not synapse.work_id:
            return StageResult(passed=False, reason="No work_id", stage="verify_work_id")
        if not synapse.work_id_nonce or not synapse.work_id_time_ns:
            # Backward compatibility removed in protocol 1.1.0: work_id_nonce and
            # work_id_time_ns are now mandatory.  Miners still on old versions must
            # upgrade.
            logger.warning("Miner did not provide work_id_nonce/time_ns — rejecting (required since v1.1.0)")
            return StageResult(
                passed=False,
                reason="work_id_nonce and work_id_time_ns are required (since protocol v1.1.0)",
                stage="verify_work_id",
            )

        # Freshness check: time_ns must be within ±5 minutes of validator wall-clock time
        try:
            import time as _time

            miner_time_ns = int(synapse.work_id_time_ns)
            now_ns = anchor_ns if anchor_ns is not None else _time.time_ns()
            drift_ns = abs(now_ns - miner_time_ns)
            freshness_s = self.remote_config.work_id_freshness_seconds
            freshness_ns = freshness_s * 1_000_000_000
            if drift_ns > freshness_ns:
                drift_s = drift_ns / 1_000_000_000
                return StageResult(
                    passed=False,
                    reason=f"Work ID too stale: time drift {drift_s:.0f}s exceeds ±{freshness_s}s window",
                    stage="verify_work_id",
                )
            # Warn when drift exceeds 50% of the window — early signal before hard rejections
            if drift_ns > freshness_ns // 2:
                drift_s = drift_ns / 1_000_000_000
                logger.warning(
                    "Clock drift warning: %.0fs drift with miner (limit: %ds). "
                    "Check NTP synchronization to avoid future rejections.",
                    drift_s,
                    freshness_s,
                )
        except (ValueError, TypeError):
            return StageResult(passed=False, reason="Invalid work_id_time_ns (not an integer)", stage="verify_work_id")

        config_json = json.dumps(synapse.scenario_config, sort_keys=True)
        payload = config_json + miner_hotkey + synapse.work_id_time_ns + synapse.work_id_nonce
        expected = hashlib.sha256(payload.encode()).hexdigest()

        if expected != synapse.work_id:
            return StageResult(
                passed=False,
                reason=f"Work ID mismatch: expected {expected[:16]}..., got {synapse.work_id[:16]}...",
                stage="verify_work_id",
            )
        return StageResult(passed=True, reason="OK", stage="verify_work_id")

    async def _work_token_check(self, miner_hotkey: str) -> StageResult:
        if self.api_client is None:
            return StageResult(passed=False, reason="API unavailable (fail closed)", stage="work_token_check")

        try:
            has_balance = await self.api_client.check_balance(miner_hotkey)
            if not has_balance:
                return StageResult(passed=False, reason="Insufficient work-token balance", stage="work_token_check")
            return StageResult(passed=True, reason="OK", stage="work_token_check")
        except httpx.HTTPStatusError as e:
            reason = f"Balance check HTTP error: {e.response.status_code}"
            return StageResult(passed=False, reason=reason, stage="work_token_check")
        except Exception as e:
            return StageResult(passed=False, reason=f"Balance check failed: {e}", stage="work_token_check")

    def _rate_limit_check(self, miner_hotkey: str) -> StageResult:
        if not self.rate_limiter.check(miner_hotkey):
            return StageResult(passed=False, reason="Rate limit exceeded", stage="rate_limit_check")
        return StageResult(passed=True, reason="OK", stage="rate_limit_check")

    async def _novelty_check(self, config: dict) -> StageResult:
        if self.api_client is None:
            return StageResult(
                passed=False,
                reason="Novelty check unavailable: no API (fail closed)",
                stage="novelty_check",
            )
        if self._embedding_service is None:
            return StageResult(
                passed=False,
                reason="Novelty check unavailable: no embedding service (fail closed)",
                stage="novelty_check",
            )

        try:
            svc = self._embedding_service
            pooled = svc.embed_config(config)

            field_embeddings = svc.extract_field_embeddings(config, parsed_config=self._parsed_config)

            self._last_pooled_embedding = pooled.tolist()

            threshold = self.remote_config.novelty_threshold
            result = await self.api_client.check_novelty(
                self._last_pooled_embedding,
                threshold=threshold,
                field_embeddings=field_embeddings if field_embeddings else None,
            )
            if not result.get("novel", True):
                msg = result.get("message", "Too similar")
                return StageResult(passed=False, reason=msg, stage="novelty_check")
            return StageResult(passed=True, reason="OK", stage="novelty_check")
        except httpx.HTTPError as e:
            logger.warning("Novelty check HTTP error (fail closed): %s", e)
            return StageResult(
                passed=False, reason=f"Novelty check HTTP error (fail closed): {e}", stage="novelty_check"
            )
        except Exception as e:
            logger.error("Novelty check unexpected error (fail closed): %s", e, exc_info=True)
            return StageResult(passed=False, reason=f"Novelty check error (fail closed): {e}", stage="novelty_check")

    async def _classifier_gate(self, config: dict) -> StageResult:
        if self.api_client is None:
            return StageResult(passed=False, reason="Classifier API unavailable (fail closed)", stage="classifier_gate")

        try:
            threshold = self.remote_config.classifier_threshold

            # Custom archetype conservative scoring: raise the threshold
            archetype = config.get("tension_archetype", "")
            if archetype == "custom":
                bump = self.remote_config.custom_archetype_threshold_bump
                threshold = min(threshold + bump, 0.99)

            result = await self.api_client.classify_config(config, threshold)
            confidence = result["confidence"]
            passed = result["passed"]
            self._last_classifier_score = confidence
            if not passed:
                # M1: keep the numeric decision boundary out of the reason
                # string. An attacker who gains access to reasons (future
                # response surfacing, telemetry, log aggregation) could
                # reverse-engineer the classifier decision boundary over
                # repeated probes. The values are still captured in
                # structured logs for operators.
                logger.info(
                    "classifier_reject | confidence=%.3f threshold=%.3f archetype=%s",
                    confidence,
                    threshold,
                    archetype or "",
                )
                return StageResult(
                    passed=False,
                    reason="Classifier rejected submission",
                    stage="classifier_gate",
                )
            logger.debug("classifier_accept | confidence=%.3f threshold=%.3f", confidence, threshold)
            return StageResult(passed=True, reason="OK", stage="classifier_gate")
        except httpx.HTTPError as e:
            logger.warning("Classifier gate HTTP error (fail closed): %s", e)
            return StageResult(
                passed=False, reason=f"Classifier API HTTP error (fail closed): {e}", stage="classifier_gate"
            )
        except Exception as e:
            logger.error("Classifier gate unexpected error (fail closed): %s", e, exc_info=True)
            return StageResult(
                passed=False, reason=f"Classifier API unavailable (fail closed): {e}", stage="classifier_gate"
            )

    async def _simulate(self, config: dict) -> StageResult:
        if self._simulation_runner is None:
            return StageResult(passed=False, reason="No simulation runner (fail closed)", stage="simulate")

        try:
            # Run blocking Docker operation in thread executor
            result = await asyncio.to_thread(self._simulation_runner.run_simulation, config)
            if not result.success:
                reason = result.error or "Simulation failed"
                if result.coherence and not result.coherence.passed:
                    reason = f"Coherence check failed: {result.coherence.reasons}"
                return StageResult(passed=False, reason=reason, stage="simulate")

            self._last_transcript = result.transcript
            return StageResult(
                passed=True,
                reason=f"OK ({result.wall_clock_seconds:.1f}s, {len(result.transcript.events)} events)",
                stage="simulate",
            )
        except Exception as e:
            logger.error("Simulation error: %s", e)
            return StageResult(passed=False, reason=f"Simulation error: {e}", stage="simulate")

    async def _gatekeeper_check(self, config: dict) -> StageResult | None:
        """Ask a remote-configured LLM rubric whether the just-run simulation
        represents a valid moral-reasoning outcome.

        Returns None when the feature is disabled (empty remote prompt, no
        transcript, or no LLM provider) so the caller skips appending a stage
        record. Fail-open on any LLM error or ambiguous reply — a quality gate
        outage should not reject every submission.
        """
        prompt = self.remote_config.gatekeeper_prompt if self.remote_config else ""
        if not prompt:
            logger.debug("gatekeeper: skipped (no remote prompt configured)")
            return None
        if self._last_transcript is None or self._llm_provider is None:
            logger.debug("gatekeeper: skipped (no transcript or LLM provider)")
            return None

        summary = _summarize_transcript_for_gatekeeper(self._last_transcript, config)
        try:
            reply = await self._llm_provider.complete(
                summary,
                system=f"{prompt}\n\nRespond with exactly PASS or FAIL on the first line, then one sentence.",
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("gatekeeper LLM call failed (fail-open): %s", e)
            return None

        stripped = reply.strip()
        first_word = stripped.split(None, 1)[0].upper() if stripped else ""
        if first_word.startswith("FAIL"):
            # Keep the full reply (verdict + explanation) up to 256 chars so
            # operators can see why the gatekeeper rejected in pipeline logs.
            logger.info("gatekeeper: FAIL %s", stripped[:200])
            return StageResult(
                passed=False,
                reason=f"Gatekeeper rejected: {stripped[:256]}",
                stage="gatekeeper",
            )
        if first_word.startswith("PASS"):
            logger.info("gatekeeper: PASS %s", stripped[:200])
            return StageResult(passed=True, reason="OK", stage="gatekeeper")
        logger.warning("gatekeeper: ambiguous LLM reply %r (fail-open)", reply[:120])
        return None

    async def _semantic_coherence_check(self, config: dict) -> StageResult | None:
        """Run LLM-based semantic coherence check on simulation output."""
        if not self._last_transcript or not self._llm_provider:
            return None

        try:
            from aurelius.simulation.coherence import validate_semantic_coherence

            result = await validate_semantic_coherence(
                transcript=self._last_transcript,
                scenario_config=config,
                llm_provider=self._llm_provider,
            )
            if not result.passed:
                return StageResult(
                    passed=False,
                    reason=f"Semantic coherence failed: {result.reasons}",
                    stage="semantic_coherence",
                )
            return StageResult(passed=True, reason="OK", stage="semantic_coherence")
        except Exception as e:
            logger.warning("Semantic coherence check error (fail closed): %s", e)
            return StageResult(
                passed=False,
                reason=f"Semantic coherence check error (fail closed): {e}",
                stage="semantic_coherence",
            )

    async def _deduct_work_token(
        self,
        miner_hotkey: str,
        work_id: str | None,
        config_hash: str = "",
        work_id_signature: str = "",
    ) -> StageResult:
        if not work_id:
            return StageResult(passed=False, reason="No work_id", stage="deduct_work_token")

        if self.api_client is None:
            return StageResult(passed=False, reason="API unavailable (fail closed)", stage="deduct_work_token")

        try:
            result = await self.api_client.consume_work_token(
                miner_hotkey,
                work_id,
                config_hash=config_hash,
                work_id_signature=work_id_signature,
            )
            if result.success:
                return StageResult(passed=True, reason="OK", stage="deduct_work_token")
            return StageResult(passed=False, reason=result.message, stage="deduct_work_token")
        except httpx.HTTPStatusError as e:
            reason = f"Deduction HTTP error: {e.response.status_code}"
            return StageResult(passed=False, reason=reason, stage="deduct_work_token")
        except Exception as e:
            return StageResult(passed=False, reason=f"Deduction failed: {e}", stage="deduct_work_token")


def _summarize_transcript_for_gatekeeper(transcript, config: dict, char_cap: int = 6000) -> str:
    """Build the user message sent to the gatekeeper LLM.

    Includes the scenario premise, agent roster, and an ordered list of
    transcript events (truncated to `char_cap` characters total) so the LLM
    can judge whether the simulation actually engaged with a moral tradeoff.
    Separate from the system prompt (which carries the remote-configured
    rubric) so the rubric owner controls the evaluation criteria.
    """
    parts: list[str] = []
    premise = (config or {}).get("premise") or ""
    if premise:
        parts.append(f"PREMISE:\n{premise}\n")
    agents = getattr(transcript, "agent_names", None) or []
    if agents:
        parts.append("AGENTS: " + ", ".join(agents) + "\n")
    events = getattr(transcript, "events", None) or []
    if events:
        parts.append("TRANSCRIPT:")
        for ev in events:
            agent = getattr(ev, "agent", None)
            content = getattr(ev, "content", "") or ""
            prefix = f"[{agent}] " if agent else ""
            parts.append(f"{prefix}{content}")
    summary = "\n".join(parts)
    if len(summary) > char_cap:
        summary = summary[:char_cap] + "\n… [truncated]"
    return summary
