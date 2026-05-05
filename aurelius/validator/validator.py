import asyncio
import json
import logging
import signal
import time
import uuid
from collections import defaultdict
from pathlib import Path

import bittensor as bt
import httpx
from bittensor.utils.weight_utils import process_weights_for_netuid

from aurelius.common.constants import (
    BURN_UID,
    DEFAULT_RATE_LIMIT_PER_UID_PER_TEMPO,
    MIN_VALIDATIONS_FOR_WEIGHT,
    RAMP_UP_TEMPOS,
    RESULT_RETENTION_BLOCKS,
    TEMPO_BLOCKS,
    TEMPO_SECONDS,
    WEIGHT_FAIL,
)
from aurelius.common.version import PROTOCOL_VERSION, VersionResult, check_compatibility
from aurelius.config import Config
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.api_client import CentralAPIClient
from aurelius.validator.local_queue import LocalSubmissionQueue, QueuedSubmission
from aurelius.validator.pipeline import PipelineResult, ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig

# Consistency enforcement thresholds (configurable via env vars)

logger = logging.getLogger(__name__)

MIN_VALIDATOR_COUNT_WARN = 2


def _fingerprint_secret(val: str) -> str:
    """T-10: render a secret safely for logs.

    Returns `"(not set)"` if empty, else a short sha256 fingerprint of the
    form `sha256:ab12cd34` (8 hex chars, ~32 bits — enough for operators
    to confirm two boots use the same secret, not enough for anyone to
    match against a breach dump). No substring of the secret appears.
    """
    if not val:
        return "(not set)"
    import hashlib as _hl

    digest = _hl.sha256(val.encode("utf-8", errors="replace")).hexdigest()
    return f"sha256:{digest[:8]}"


def _is_weights_rate_limit(message) -> bool:
    """T-5: detect a subtensor rate-limit rejection from a set_weights result.

    The bittensor SDK currently returns `(success=False, message=None)` for
    the per-subnet weights rate limit. Older builds returned strings with
    "rate limit" in them. Match both so the helper survives an SDK bump.
    """
    if message is None:
        return True
    text = str(message).strip().lower()
    if not text or text == "none":
        return True
    # Accept both "rate limit" and "rate-limit" phrasings, plus the
    # lone-word variant "ratelimit".
    normalized = text.replace("-", " ")
    return "rate limit" in normalized or "ratelimit" in text


def _render_cycle_summary(stats: dict) -> str:
    """One-line structured summary of a main-loop cycle for operators.

    Keeping the render a pure function means unit tests can cover the
    formatting without running the validator's main loop — simpler tests,
    and no surprise changes to a line operators probably grep for.

    Expected keys (missing keys are omitted, not defaulted):
      miners_queried: int
      miners_passed: int
      stage_failures: dict[str, int] (stage name -> count)
      weights_outcome: str ("success" / "rate_limit" / "failed" / "skipped")
      cycle_duration_s: float (rounded)
      in_ramp_up: bool
      degraded_mode: bool
    """
    parts = ["cycle_summary"]
    for key in (
        "miners_queried",
        "miners_passed",
        "cycle_duration_s",
        "weights_outcome",
        "in_ramp_up",
        "degraded_mode",
    ):
        if key in stats:
            parts.append(f"{key}={stats[key]}")
    fails = stats.get("stage_failures") or {}
    if fails:
        fail_parts = ",".join(f"{k}:{v}" for k, v in sorted(fails.items()))
        parts.append(f"failures={fail_parts}")
    return " | ".join(parts)


# CS-11: periodic Docker daemon health check cadence (seconds).
DOCKER_HEALTH_CHECK_INTERVAL = 300.0
# H-5: periodic clock-drift re-check cadence. Running it once per tempo is
# plenty — NTP corrections are infrequent and the Central API's HEAD call
# adds perceptible latency. First check still happens eagerly at startup.
CLOCK_DRIFT_CHECK_INTERVAL = TEMPO_SECONDS


class Validator:
    @staticmethod
    def _check_testlab_safety(testlab_mode: bool, network: str) -> None:
        """TESTLAB_MODE on finney (mainnet) is never safe — it disables
        ramp-up and other guard behaviors. Mirrors Miner's guard at
        `aurelius/miner/miner.py:33-40`. Raises RuntimeError when violated.
        """
        if testlab_mode and network == "finney":
            raise RuntimeError(
                "TESTLAB_MODE=1 is not allowed on mainnet (finney). "
                "This disables validator safety gates (ramp-up, permit checks) "
                "and exposes the network to unaudited consensus. "
                "Remove TESTLAB_MODE or set ENVIRONMENT=testnet."
            )

    def __init__(self):
        self.config = Config
        self.should_exit = False

        # Fail-fast: TESTLAB_MODE on mainnet. Must run before any bittensor
        # I/O so a misconfigured mainnet validator refuses to touch the chain.
        self._check_testlab_safety(self.config.TESTLAB_MODE, self.config.NETWORK)

        # Pipeline results: hotkey -> latest PipelineResult
        self.results: dict[str, PipelineResult] = {}
        # Validation counts per hotkey (for ramp-up)
        self.validation_counts: dict[str, int] = defaultdict(int)
        self.last_weight_block = 0
        self.start_time = time.monotonic()
        self._last_seen: dict[str, float] = {}  # hotkey -> last validation time
        self._in_flight: int = 0  # count of in-flight pipeline runs

        self.wallet = bt.Wallet(name=self.config.WALLET_NAME, hotkey=self.config.WALLET_HOTKEY)
        self.subtensor = bt.Subtensor(network=self.config.NETWORK)
        self.dendrite = bt.Dendrite(wallet=self.wallet)
        self.metagraph = bt.Metagraph(
            netuid=self.config.NETUID,
            network=self.config.NETWORK,
            subtensor=self.subtensor,
        )

        # Central API client (async — auth deferred to run())
        api_url = self.config.CENTRAL_API_URL
        self.api_client: CentralAPIClient | None = CentralAPIClient(base_url=api_url, timeout=self.config.API_TIMEOUT)

        # Remote config
        self.remote_config = RemoteConfig(api_client=self.api_client)

        # Rate limiter (persisted across restarts)
        self.rate_limiter = RateLimiter(
            max_submissions=DEFAULT_RATE_LIMIT_PER_UID_PER_TEMPO,
            window_seconds=TEMPO_SECONDS,
            persist_path=self.config.RATE_LIMITER_STATE_PATH,
        )

        # Validation counts (persisted across restarts for ramp-up survival)
        self._validation_counts_path = self.config.VALIDATION_COUNTS_PATH
        self._load_validation_counts()

        # H-9: ramp-up anchor is a block number (not a wall-clock timestamp)
        # persisted to disk. This way a restart mid-ramp-up resumes where it
        # left off instead of rewinding the window to zero. Lazily initialized
        # on first metagraph sync (see _ensure_ramp_up_anchor).
        self._ramp_up_anchor_path = self.config.RAMP_UP_ANCHOR_PATH
        self._ramp_up_start_block: int | None = self._load_ramp_up_anchor()

        # Local submission queue (degraded mode)
        self.local_queue = LocalSubmissionQueue(
            persist_path=self.config.SUBMISSION_QUEUE_PATH,
            remote_config=self.remote_config,
        )
        self._drain_consecutive_failures: int = 0
        self._drain_next_attempt: float = 0.0  # monotonic time
        # Cycle-summary outcome of the most recent _set_weights call.
        # "skipped" means the loop did not attempt to set weights this
        # cycle (block interval not yet reached).
        self._last_weights_outcome: str = "skipped"

        # Preflight validation (fail-fast on misconfiguration)
        self._preflight_checks()

        self._log_validator_initialized()

    def _log_validator_initialized(self):
        """Log static identity info at construction time."""
        logger.info(
            "Validator initialized | wallet=%s hotkey=%s netuid=%d neurons=%d",
            self.wallet.name,
            self.wallet.hotkey_str,
            self.config.NETUID,
            self.metagraph.n,
        )

    def _log_config_summary(self):
        """Log runtime config summary. Call after first remote_config refresh so
        operators see the effective values (remote overrides applied) rather
        than local fallbacks."""
        from aurelius.config import ENVIRONMENT

        gatekeeper_prompt_len = len(self.remote_config.gatekeeper_prompt or "")
        gatekeeper_status = f"configured ({gatekeeper_prompt_len} chars)" if gatekeeper_prompt_len else "OFF"
        logger.info(
            "Config summary | env=%s network=%s api_url=%s llm_model=%s llm_base=%s llm_key=%s testlab=%s burn_mode=%s burn_pct=%.0f%% pool_size=%d gatekeeper=%s",
            ENVIRONMENT,
            self.config.NETWORK,
            self.config.CENTRAL_API_URL,
            self.remote_config.llm_model,
            self.remote_config.llm_base_url,
            _fingerprint_secret(self.config.LLM_API_KEY),
            self.config.TESTLAB_MODE,
            self.remote_config.burn_mode,
            self.remote_config.burn_percentage * 100,
            self.remote_config.container_pool_size,
            gatekeeper_status,
        )

    @property
    def in_ramp_up(self) -> bool:
        """True iff the validator is still within its ramp-up window.

        H-9: the anchor is a persisted block number so a restart doesn't
        reset the window. If the anchor hasn't been captured yet (metagraph
        hasn't synced — should only happen in tests), we fall back to the
        wall-clock elapsed check, which is strictly no-more-permissive than
        the block-based one on a steadily-progressing chain.
        """
        if self.config.TESTLAB_MODE:
            return False
        ramp_blocks = RAMP_UP_TEMPOS * TEMPO_BLOCKS
        if self._ramp_up_start_block is not None:
            current_block = int(getattr(self.metagraph, "block", 0) or 0)
            if current_block <= 0:
                # Metagraph not yet synced — treat as still in ramp-up so
                # we don't release weights before the first sync lands.
                return True
            return (current_block - self._ramp_up_start_block) < ramp_blocks
        elapsed = time.monotonic() - self.start_time
        return elapsed < (RAMP_UP_TEMPOS * TEMPO_SECONDS)

    def _preflight_checks(self):
        """Validate critical configuration before starting. Raises ValueError on issues."""
        from aurelius.config import ENVIRONMENT

        is_local = ENVIRONMENT == "local"

        # 1. CENTRAL_API_URL must not be localhost in non-local environments
        api_url = self.config.CENTRAL_API_URL
        if not is_local and ("localhost" in api_url or "127.0.0.1" in api_url):
            raise ValueError(
                f"CENTRAL_API_URL={api_url} contains localhost but ENVIRONMENT={ENVIRONMENT}. "
                "Set CENTRAL_API_URL to the actual Central API address for distributed deployment."
            )

        # 2. LLM_API_KEY must be set when using an external LLM endpoint
        if not is_local and not self.config.LLM_API_KEY:
            from aurelius.simulation.docker_runner import DockerSimulationRunner

            if not DockerSimulationRunner._check_local_base_url(self.remote_config.llm_base_url):
                raise ValueError(
                    "LLM_API_KEY is empty but LLM_BASE_URL points to an external API. "
                    "Set LLM_API_KEY or set LLM_BASE_URL to a local endpoint."
                )

        # 3. Warn if wallet is still on defaults (easy identity collision)
        if self.config.WALLET_NAME == "default" and self.config.WALLET_HOTKEY == "default":
            logger.warning(
                "WALLET_NAME and WALLET_HOTKEY are both 'default'. "
                "Set explicit wallet names to avoid identity collisions between operators."
            )

        # 4. Check validator permit on metagraph (warn, don't block — permit may come after staking)
        if not is_local:
            hotkey = self.wallet.hotkey.ss58_address
            if hotkey in self.metagraph.hotkeys:
                uid = self.metagraph.hotkeys.index(hotkey)
                stake = float(self.metagraph.S[uid])
                has_permit = bool(self.metagraph.validator_permit[uid])
                if not has_permit:
                    logger.warning(
                        "Validator permit NOT granted (UID %d, stake: %.4f TAO). "
                        "The Central API will reject authentication until you have a permit. "
                        "Stake TAO to this hotkey: btcli stake add --netuid %d --network %s",
                        uid,
                        stake,
                        self.config.NETUID,
                        self.config.NETWORK,
                    )
                else:
                    logger.info("Validator permit confirmed (UID %d, stake: %.4f TAO)", uid, stake)
            else:
                logger.error(
                    "Hotkey %s is NOT registered on subnet %d. "
                    "Register first: btcli subnet register --netuid %d --network %s",
                    hotkey[:16],
                    self.config.NETUID,
                    self.config.NETUID,
                    self.config.NETWORK,
                )

        # 5. Data directories are writable
        self.config.ensure_data_dirs()

        # 6. Burn UID validation (B-4). Burn mode routes all emissions to
        # BURN_UID; if that UID doesn't exist on the subnet or is occupied
        # by a staked miner, emissions go to the wrong place.
        if not is_local:
            self._validate_burn_uid()

        logger.info("Preflight checks passed")

    def _validate_burn_uid(self) -> None:
        """B-4: warn (don't block) if the hardcoded burn UID is unexpectedly
        occupied or out of range. We don't hard-fail because the subnet could
        legitimately be pre-population, but the operator must know emissions
        may be routed to a real miner if the expected reservation slips.
        """
        n = self.metagraph.n
        if BURN_UID >= n:
            logger.warning(
                "Burn UID %d >= metagraph size %d. Weight-setting will "
                "likely drop the burn slice (process_weights_for_netuid "
                "clamps out-of-range UIDs). Either wait for the subnet to "
                "populate or coordinate with the subnet owner to reserve "
                "UID %d.",
                BURN_UID,
                n,
                BURN_UID,
            )
            return
        burn_hotkey = self.metagraph.hotkeys[BURN_UID]
        burn_stake = float(self.metagraph.S[BURN_UID])
        burn_has_permit = bool(self.metagraph.validator_permit[BURN_UID])
        if burn_stake > 1.0 or burn_has_permit:
            logger.warning(
                "Burn UID %d is occupied (hotkey=%s stake=%.2f TAO permit=%s). "
                "Emissions from burn_mode will accrue to this hotkey, not the "
                "subnet owner. Verify with the subnet owner that this is the "
                "intended burn address.",
                BURN_UID,
                burn_hotkey[:16],
                burn_stake,
                burn_has_permit,
            )
        else:
            logger.info(
                "Burn UID %d verified (hotkey=%s, unstaked) — emissions from burn_mode will route here as intended.",
                BURN_UID,
                burn_hotkey[:16],
            )

    async def _check_clock_drift(self):
        """GI-5: Verify system clock accuracy. Excessive drift causes silent 100% rejection."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.head(self.config.CENTRAL_API_URL + "/health")
            server_date = resp.headers.get("date")
            if not server_date:
                logger.warning("Clock drift check: server did not return Date header, skipping")
                return
            from email.utils import parsedate_to_datetime

            server_time = parsedate_to_datetime(server_date).timestamp()
            local_time = time.time()
            drift = abs(local_time - server_time)
            freshness = self.remote_config.work_id_freshness_seconds
            if drift > freshness:
                raise SystemExit(
                    f"FATAL: Clock drift {drift:.0f}s exceeds freshness window {freshness}s. "
                    "Fix system clock before starting validator."
                )
            elif drift > freshness * 0.5:
                logger.warning("Clock drift %.0fs is >50%% of freshness window %ds", drift, freshness)
            else:
                logger.info("Clock drift check passed (drift=%.1fs)", drift)
        except SystemExit:
            raise
        except Exception as e:
            logger.warning("Could not verify clock drift: %s (continuing anyway)", e)

    async def _initialize_async(self):
        """Async initialization: authenticate and load quality gate dependencies."""

        # Authenticate with Central API (retry with backoff)
        for attempt in range(3):
            try:
                await self.api_client.authenticate(self.wallet)
                logger.info("Central API client authenticated: %s", self.config.CENTRAL_API_URL)
                break
            except httpx.HTTPStatusError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                is_auth_rejection = e.response.status_code == 403
                if is_auth_rejection:
                    # 403 = permanent rejection (not registered, no permit). Don't retry.
                    logger.error(
                        "Central API rejected authentication: %s",
                        detail or str(e),
                    )
                    if "not registered" in detail:
                        logger.error(
                            "ACTION REQUIRED: Register this hotkey on subnet %d. "
                            "Run: btcli subnet register --netuid %d --network %s",
                            self.config.NETUID,
                            self.config.NETUID,
                            self.config.NETWORK,
                        )
                    elif "validator permit" in detail or "no_permit" in detail:
                        logger.error(
                            "ACTION REQUIRED: Stake TAO on this hotkey to obtain a validator permit. "
                            "Run: btcli stake add --netuid %d --network %s",
                            self.config.NETUID,
                            self.config.NETWORK,
                        )
                    logger.warning("Running in degraded mode (local queue only)")
                    self.api_client = None
                    break
                elif attempt < 2:
                    delay = 2.0 * (2**attempt)
                    logger.warning(
                        "Failed to authenticate with Central API (attempt %d/3), retrying in %.0fs: %s",
                        attempt + 1,
                        delay,
                        detail or str(e),
                    )
                    await asyncio.sleep(delay)
                else:
                    # Transient failure — keep api_client so the main loop can
                    # re-authenticate once the API recovers. Only 403 nulls it.
                    logger.warning(
                        "Failed to authenticate with Central API after 3 attempts, "
                        "running in degraded mode; will retry in main loop: %s",
                        detail or str(e),
                    )
            except Exception as e:
                if attempt < 2:
                    delay = 2.0 * (2**attempt)
                    logger.warning(
                        "Failed to authenticate with Central API (attempt %d/3), retrying in %.0fs: %s",
                        attempt + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "Failed to authenticate with Central API after 3 attempts, "
                        "running in degraded mode; will retry in main loop: %s",
                        e,
                    )

        # GI-5: Verify system clock accuracy before starting
        await self._check_clock_drift()

        # First remote-config refresh — do this before constructing the
        # simulation runner / LLM provider so they pick up remote values
        # on their initial construction. If the API is unreachable, refresh()
        # is a no-op and the dependents fall back to local Config.
        if self.api_client is not None:
            await self.remote_config.refresh()

        # Quality gate dependencies
        embedding_service = None
        simulation_runner = None

        try:
            from aurelius.common.embeddings import EmbeddingService

            embedding_service = EmbeddingService()
            logger.info("Embedding service initialized")
        except ImportError:
            logger.warning("sentence-transformers not installed — novelty check will be skipped")
        except Exception as e:
            logger.warning("Failed to initialize embedding service: %s", e)

        if self.config.LLM_API_KEY:
            try:
                from aurelius.simulation.docker_runner import DockerSimulationRunner

                simulation_runner = DockerSimulationRunner(
                    remote_config=self.remote_config,
                    llm_api_key=self.config.LLM_API_KEY,
                )
                # CS-11: surface Docker daemon unreachability at startup,
                # not only on the first simulation attempt.
                try:
                    simulation_runner.preflight_check()
                    self._docker_healthy = True
                    logger.info("Simulation runner initialized")
                except RuntimeError as e:
                    self._docker_healthy = False
                    logger.error("Docker daemon preflight failed: %s — simulation will fail closed until recovered", e)
            except ImportError:
                logger.warning("docker not installed — simulation will be skipped")
                self._docker_healthy = False
            except Exception as e:
                logger.warning("Failed to initialize simulation runner: %s", e)
                self._docker_healthy = False
        else:
            logger.info("No LLM_API_KEY configured — simulation will be skipped")
            self._docker_healthy = False
        self._simulation_runner = simulation_runner
        self._last_docker_health_check: float = 0.0
        # H-5: startup _check_clock_drift already ran in _initialize_async, so
        # set the next periodic check one interval out from startup. Setting
        # to 0.0 would cause the loop's first pass to re-check immediately.
        self._last_clock_drift_check: float = time.monotonic()

        # LLM provider for semantic coherence checks (optional)
        llm_provider = None
        if self.config.LLM_API_KEY:
            try:
                from aurelius.common.llm import create_llm

                llm_provider = create_llm(
                    api_key=self.config.LLM_API_KEY,
                    model=self.remote_config.llm_model,
                    base_url=self.remote_config.llm_base_url or None,
                )
                logger.info("LLM provider initialized for semantic coherence checks")
            except Exception as e:
                logger.warning("Failed to initialize LLM provider for coherence: %s", e)

        # Validation pipeline
        self.pipeline = ValidationPipeline(
            api_client=self.api_client,
            remote_config=self.remote_config,
            rate_limiter=self.rate_limiter,
            embedding_service=embedding_service,
            simulation_runner=simulation_runner,
            llm_provider=llm_provider,
        )

        # Summary log — deferred until after remote_config refresh and all
        # dependencies are built so it reflects the actual effective config.
        self._log_config_summary()

    async def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        await self._initialize_async()

        logger.info("Validator running. Press Ctrl+C to exit.")
        while not self.should_exit:
            try:
                cycle_start = time.monotonic()
                # Reset per-cycle outcome so a "skipped" rolls over cleanly if
                # this cycle doesn't attempt _set_weights.
                self._last_weights_outcome = "skipped"

                # Sync bittensor calls run in thread executor
                await asyncio.to_thread(self.metagraph.sync, subtensor=self.subtensor)

                # H-9: capture the ramp-up anchor on the first sync after
                # boot; no-op every other cycle.
                self._ensure_ramp_up_anchor()

                self._check_validator_count()

                # Re-authenticate if token is expiring
                if self.api_client and self.api_client.needs_reauth:
                    try:
                        await self.api_client.authenticate(self.wallet)
                        logger.info("Re-authenticated with Central API")
                    except Exception as e:
                        logger.warning("Failed to re-authenticate with Central API: %s", e)

                # Refresh remote config (TTL-based)
                was_available = self.remote_config.api_available
                await self.remote_config.refresh()
                now_available = self.remote_config.api_available
                if was_available and not now_available:
                    logger.error(
                        "validator_api_state | available=False failures=%d — entering degraded mode",
                        self.remote_config._fetch_failures,
                    )
                elif not was_available and now_available:
                    logger.error(
                        "validator_api_state | available=True — exiting degraded mode",
                    )

                # Update rate limiter from remote config
                self.rate_limiter.update_config(
                    max_submissions=self.remote_config.rate_limit_per_uid_per_tempo,
                    window_seconds=TEMPO_SECONDS,
                )

                # CS-11: periodic Docker daemon health check. Surface
                # transitions in logs so outages aren't silent until the
                # next simulation attempt.
                self._tick_docker_health_check()

                # H-5: periodic clock-drift re-check. Startup catches drift
                # once; this catches mid-run NTP failures that would
                # otherwise silently reject every miner submission.
                await self._tick_clock_drift_check()

                # Drain local queue if API is back
                await self._drain_local_queue()

                axons = self._get_miner_axons()
                if not axons:
                    logger.info("No serving miners found, sleeping...")
                    await asyncio.sleep(30)
                    continue

                responses = await asyncio.to_thread(self._query_miners, axons)

                # Block fetch moved before validation so we can stamp
                # recorded_at_block on each result and run the retention
                # prune. Replaces the old self.results.clear() — see
                # _prune_stale_results / _record_result.
                current_block = await asyncio.to_thread(lambda: self.subtensor.block)
                self._prune_stale_results(current_block)
                await self._validate_responses(responses, current_block)

                blocks_since = current_block - self.last_weight_block
                block_interval = self.remote_config.weight_interval // 12
                if blocks_since >= block_interval:
                    await self._set_weights()
                    self.last_weight_block = current_block
                    self._cleanup_stale_results()

                # One structured summary line per cycle so operators don't
                # have to grep across 10 different per-stage INFO lines.
                logger.info(
                    _render_cycle_summary(
                        self._build_cycle_stats(
                            responses=responses,
                            cycle_start=cycle_start,
                            current_block=current_block,
                        )
                    )
                )

                polling_interval = self.remote_config.polling_interval
                await asyncio.sleep(polling_interval)

            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception("Error in validator loop")
                await asyncio.sleep(30)

        await self.stop()

    def _tick_docker_health_check(self) -> None:
        """CS-11: run a Docker health check at most once per interval.

        Logs ERROR on transition to unhealthy and INFO on recovery.
        When there is no simulation runner configured (no LLM key, or
        docker package missing), this is a no-op.
        """
        if self._simulation_runner is None:
            return
        now = time.monotonic()
        if now - self._last_docker_health_check < DOCKER_HEALTH_CHECK_INTERVAL:
            return
        self._last_docker_health_check = now
        was_healthy = self._docker_healthy
        self._docker_healthy = self._simulation_runner.health_check()
        if was_healthy and not self._docker_healthy:
            logger.error("docker_health | state=unhealthy — simulations will fail closed until recovered")
        elif not was_healthy and self._docker_healthy:
            logger.info("docker_health | state=recovered")

    def _build_cycle_stats(self, responses, cycle_start: float, current_block: int | None = None) -> dict:
        """Collect per-cycle stats for _render_cycle_summary.

        Pure-ish helper: side-effect-free, pulls from already-computed
        instance state (self.results, self._last_weights_outcome, etc).
        Kept as a method so tests can exercise it with a constructed
        Validator + canned state without running the main loop.

        ``current_block`` filters self.results to entries recorded this
        cycle so retained passes from prior cycles don't inflate
        miners_passed. Optional for back-compat with older test fixtures
        that don't stamp recorded_at_block — when None, every entry is
        counted (legacy behavior).
        """
        from collections import Counter

        if current_block is None:
            cycle_results = list(self.results.values())
        else:
            cycle_results = [r for r in self.results.values() if r.recorded_at_block == current_block]

        stage_failures: Counter = Counter()
        passed = 0
        for result in cycle_results:
            if result.passed:
                passed += 1
            elif result.failed_stage:
                stage_failures[result.failed_stage] += 1
        return {
            "miners_queried": len(responses) if responses is not None else 0,
            "miners_passed": passed,
            "stage_failures": dict(stage_failures),
            "weights_outcome": self._last_weights_outcome,
            "cycle_duration_s": round(time.monotonic() - cycle_start, 2),
            "in_ramp_up": self.in_ramp_up,
            "degraded_mode": not self.remote_config.api_available,
        }

    async def _tick_clock_drift_check(self) -> None:
        """H-5: re-check clock drift once per tempo.

        Startup check at _check_clock_drift already SystemExits on drift
        exceeding the freshness window. Mid-run NTP failure would otherwise
        cause every miner submission to be silently rejected (stale work
        id), with the operator seeing only pipeline failures. Running the
        same check periodically surfaces the problem at ERROR level and
        lets SystemExit crash the validator cleanly if drift exceeds the
        hard window — the container will restart and re-run the same
        check, failing fast until NTP is fixed.
        """
        now = time.monotonic()
        if now - self._last_clock_drift_check < CLOCK_DRIFT_CHECK_INTERVAL:
            return
        self._last_clock_drift_check = now
        try:
            await self._check_clock_drift()
        except SystemExit:
            raise
        except Exception as e:
            logger.warning("Periodic clock-drift check failed: %s", e)

    def _check_validator_count(self):
        """Alert if validator count drops to warning threshold."""
        validator_count = sum(1 for uid in range(self.metagraph.n) if self.metagraph.validator_permit[uid])
        if validator_count <= MIN_VALIDATOR_COUNT_WARN:
            logger.warning(
                "Low validator count: %d (minimum recommended: %d). Consensus may be fragile.",
                validator_count,
                MIN_VALIDATOR_COUNT_WARN + 1,
            )

    async def _drain_local_queue(self):
        """Report queued submissions if API is available.

        Uses exponential backoff on consecutive failures to avoid
        O(n) disk writes per loop iteration from re-queuing.
        """
        if self.local_queue.is_empty or self.api_client is None:
            return

        if not self.remote_config.api_available:
            return

        # Exponential backoff: skip if backing off from prior failures
        now = time.monotonic()
        if now < self._drain_next_attempt:
            return

        submissions = self.local_queue.drain(max_count=20)
        reported = 0
        for sub in submissions:
            try:
                await self.api_client.report_submission(
                    work_id=sub.work_id,
                    miner_hotkey=sub.miner_hotkey,
                    scenario_config=sub.scenario_config,
                    classifier_score=sub.classifier_score,
                    simulation_transcript=sub.simulation_transcript,
                )
                reported += 1
            except Exception:
                # Re-queue on failure and apply exponential backoff
                self.local_queue.enqueue(sub)
                self._drain_consecutive_failures += 1
                # Backoff: 60s, 120s, 240s, ... capped at 1 tempo
                delay = min(60 * (2 ** (self._drain_consecutive_failures - 1)), TEMPO_SECONDS)
                self._drain_next_attempt = time.monotonic() + delay
                logger.warning(
                    "Queue drain failed (%d consecutive), backing off %.0fs",
                    self._drain_consecutive_failures,
                    delay,
                )
                break

        if reported:
            self._drain_consecutive_failures = 0
            self._drain_next_attempt = 0.0
            logger.info("Drained %d queued submissions to API (%d remaining)", reported, self.local_queue.size)

    def _get_miner_axons(self) -> list[bt.AxonInfo]:
        """Get axons of serving miners to query.

        Filters: serving, not our own UID, and (no validator permit OR small subnet).
        On small subnets (n <= max_validators), all neurons get permits, so we
        fall back to querying any serving neuron that isn't us.
        """
        my_hotkey = self.wallet.hotkey.ss58_address
        axons = []
        for uid in range(self.metagraph.n):
            axon = self.metagraph.axons[uid]
            if not axon.is_serving:
                continue
            if self.metagraph.hotkeys[uid] == my_hotkey:
                continue
            axons.append(axon)
        return axons

    def _query_miners(self, axons: list[bt.AxonInfo]) -> list[ScenarioConfigSynapse]:
        import aurelius

        synapse = ScenarioConfigSynapse(
            request_id=str(uuid.uuid4()),
            validator_version=aurelius.__version__,
            protocol_version=PROTOCOL_VERSION,
        )
        logger.info("Querying %d miners...", len(axons))
        responses = self.dendrite.query(axons, synapse, timeout=self.remote_config.query_timeout)
        if not isinstance(responses, list):
            responses = [responses]
        return responses

    async def _validate_responses(self, responses: list[ScenarioConfigSynapse], current_block: int):
        """Run each response through the validation pipeline."""
        # Anchor freshness against the start of validation rather than the
        # moment each stage runs. With serial pipeline execution and ~30s/miner
        # in Concordia, per-stage time.time_ns() drifts by ~cycle_duration on
        # the last miner and trips verify_work_id even when the miner timestamps
        # are correct.
        cycle_anchor_ns = time.time_ns()
        for response in responses:
            hotkey = response.axon.hotkey
            if not response.is_success:
                self._record_result(hotkey, PipelineResult(weight=WEIGHT_FAIL, stages=[]), current_block)
                logger.debug("Miner %s: no response", hotkey[:8])
                continue

            # Pre-pipeline blacklist: reject miners below min_miner_version
            miner_ver = response.miner_protocol_version
            if miner_ver:
                min_ver = self.remote_config.min_miner_version
                compat = check_compatibility(min_ver, miner_ver)
                if compat == VersionResult.REJECT:
                    logger.info("Miner %s: REJECTED at blacklist (version %s < min %s)", hotkey[:8], miner_ver, min_ver)
                    self._record_result(hotkey, PipelineResult(weight=WEIGHT_FAIL, stages=[]), current_block)
                    continue

            self._in_flight += 1
            try:
                result = await self.pipeline.run(response, hotkey, anchor_ns=cycle_anchor_ns)
            finally:
                self._in_flight -= 1
            self._record_result(hotkey, result, current_block)
            self.validation_counts[hotkey] += 1
            self._save_validation_counts()
            self._last_seen[hotkey] = time.monotonic()

            if result.passed:
                logger.info("Miner %s: PASS (weight=%.3f)", hotkey[:8], result.weight)
                await self._report_submission(result, hotkey)
            else:
                failed = result.failed_stage
                reason = result.stages[-1].reason
                logger.info("Miner %s: FAIL at stage '%s': %s", hotkey[:8], failed, reason)

    async def _report_submission(self, result: PipelineResult, hotkey: str):
        """Report a passing submission to the API, or queue locally if unavailable."""
        if not result.passed:
            logger.error("BUG: _report_submission called with failed result (miner=%s)", hotkey[:8])
            return
        if not result.work_id or not result.scenario_config:
            return

        if self.api_client and self.remote_config.api_available:
            try:
                await self.api_client.report_submission(
                    work_id=result.work_id,
                    miner_hotkey=hotkey,
                    scenario_config=result.scenario_config,
                    classifier_score=result.classifier_score,
                    simulation_transcript=result.simulation_transcript,
                )
                return
            except Exception:
                logger.warning("Failed to report submission to API, queuing locally")

        # Queue locally for later reporting
        self.local_queue.enqueue(
            QueuedSubmission(
                work_id=result.work_id,
                miner_hotkey=hotkey,
                scenario_config=result.scenario_config,
                classifier_score=result.classifier_score,
                simulation_transcript=result.simulation_transcript,
            )
        )

    async def _get_consistency_multiplier(self) -> float:
        """Fetch this validator's consistency score and return a weight multiplier.

        WS4: Validators are not required to produce identical weights — Bittensor's
        Yuma consensus algorithm handles weight aggregation across validators.
        Divergence is expected due to LLM non-determinism in Concordia simulations.
        This multiplier attenuates influence of validators with consistently low
        agreement rates, encouraging convergence without enforcing it.

        Returns 1.0 if consistency data is unavailable.
        """
        if not self.api_client:
            return 1.0
        try:
            data = await self.api_client.get_consistency(self.wallet.hotkey.ss58_address)
            rate = data.get("agreement_rate", 1.0)
            total = data.get("total_reports", 0)
            min_reports = self.remote_config.min_consistency_reports
            floor = self.remote_config.consistency_floor
            if total < min_reports:
                return 1.0
            if rate < floor:
                logger.warning(
                    "Consistency too low (%.2f < %.2f) — zeroing weight influence",
                    rate,
                    floor,
                )
                return 0.0
            return (rate - floor) / (1.0 - floor)
        except Exception:
            logger.debug("Could not fetch consistency score, using default multiplier")
            return 1.0

    async def _set_weights(self):
        # Refuse to set weights if this validator is below min_validator_version
        min_ver = self.remote_config.min_validator_version
        compat = check_compatibility(PROTOCOL_VERSION, min_ver)
        if compat == VersionResult.REJECT:
            logger.warning(
                "Validator version %s below min_validator_version %s — refusing to set weights",
                PROTOCOL_VERSION,
                min_ver,
            )
            return

        import numpy as np

        # Burn mode: direct all emissions to UID 200 (burn address)
        if self.remote_config.burn_mode:
            uid_array = np.array([BURN_UID], dtype=np.int64)
            weight_array = np.array([1.0], dtype=np.float32)
        else:
            if not self.results:
                logger.info("No results to set weights for")
                return

            own_hotkey = self.wallet.hotkey.ss58_address
            consistency_mult = await self._get_consistency_multiplier()

            if consistency_mult == 0.0:
                logger.warning("Skipping weight setting due to low consistency score")
                return

            uids = []
            weights = []
            for hotkey, result in self.results.items():
                if hotkey not in self.metagraph.hotkeys:
                    continue

                # Self-validation prevention: never set weight for own hotkey
                if hotkey == own_hotkey:
                    logger.debug("Skipping self-validation weight for own hotkey %s", hotkey[:8])
                    continue

                try:
                    uid = self.metagraph.hotkeys.index(hotkey)
                except ValueError:
                    logger.debug("Hotkey %s no longer in metagraph, skipping", hotkey[:8])
                    continue

                if self.in_ramp_up and self.validation_counts[hotkey] < MIN_VALIDATIONS_FOR_WEIGHT:
                    logger.debug(
                        "Ramp-up: skipping %s (validations=%d < %d)",
                        hotkey[:8],
                        self.validation_counts[hotkey],
                        MIN_VALIDATIONS_FOR_WEIGHT,
                    )
                    continue

                uids.append(uid)
                weights.append(result.weight * consistency_mult)

            if not uids:
                logger.info("No valid UIDs to set weights for")
                return

            # Apply burn percentage: normalize miner weights then split with burn address.
            # Miner weights are normalized to sum to 1.0 first, then scaled by (1 - burn_pct)
            # so the final ratio is exactly burn_pct to UID 200 and (1 - burn_pct) to miners.
            burn_pct = self.remote_config.burn_percentage
            miner_share = 1.0 - burn_pct
            total_weight = sum(weights)
            if total_weight > 0:
                scaled_weights = [w / total_weight * miner_share for w in weights]
            else:
                scaled_weights = [0.0] * len(weights)

            uids.append(BURN_UID)
            scaled_weights.append(burn_pct)

            uid_array = np.array(uids, dtype=np.int64)
            weight_array = np.array(scaled_weights, dtype=np.float32)

        processed_uids, processed_weights = process_weights_for_netuid(
            uids=uid_array,
            weights=weight_array,
            netuid=self.config.NETUID,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )

        logger.info(
            "Setting weights for %d UIDs (burn_mode=%s, ramp_up=%s)...",
            len(processed_uids),
            self.remote_config.burn_mode,
            self.in_ramp_up,
        )

        def _do_set_weights():
            return self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.NETUID,
                uids=processed_uids,
                weights=processed_weights,
                mechid=0,
            )

        result = await asyncio.to_thread(_do_set_weights)

        if result.success:
            logger.info("Weights set successfully: %s", result.message)
            self._last_weights_outcome = "success"
        elif _is_weights_rate_limit(result.message):
            # T-5: subtensor rejects calls more frequent than its per-subnet
            # weights rate limit. Our loop attempts every `weight_interval`
            # (5 min default) while the chain rate limit is usually longer
            # (observed ~25 min on SN455), so most calls fail this way.
            # Surface at DEBUG so real weight-set failures remain visible.
            logger.debug("Weight set deferred by subtensor rate limit: %s", result.message)
            self._last_weights_outcome = "rate_limit"
        else:
            logger.warning("Failed to set weights: %s", result.message)
            self._last_weights_outcome = "failed"

    def _load_validation_counts(self) -> None:
        path = Path(self._validation_counts_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for hotkey, count in data.items():
                self.validation_counts[hotkey] = count
            logger.info("Loaded validation counts: %d hotkeys", len(data))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load validation counts: %s", e)

    def _save_validation_counts(self) -> None:
        try:
            path = Path(self._validation_counts_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(self.validation_counts)))
        except OSError as e:
            logger.warning("Failed to persist validation counts: %s", e)

    def _load_ramp_up_anchor(self) -> int | None:
        """H-9: read a previously persisted ramp-up start block, if any."""
        path = Path(self._ramp_up_anchor_path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            block = int(data.get("block", 0))
            if block <= 0:
                return None
            logger.info("Loaded ramp-up anchor block: %d", block)
            return block
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Failed to load ramp-up anchor: %s", e)
            return None

    def _ensure_ramp_up_anchor(self) -> None:
        """H-9: on first run, capture the current block as the ramp-up start
        so subsequent restarts resume the same window. No-op on later runs.

        Called from the main loop after a metagraph sync so we have a
        reliable current-block reading; doing it at startup is risky
        because the metagraph may not have synced yet.
        """
        if self._ramp_up_start_block is not None:
            return
        current_block = int(getattr(self.metagraph, "block", 0) or 0)
        if current_block <= 0:
            return  # try again next cycle
        self._ramp_up_start_block = current_block
        try:
            path = Path(self._ramp_up_anchor_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"block": current_block}))
            logger.info("Ramp-up anchor set to block %d (persisted)", current_block)
        except OSError as e:
            logger.warning(
                "Failed to persist ramp-up anchor (block=%d, anchor stays in-memory only): %s",
                current_block,
                e,
            )

    def _cleanup_stale_results(self):
        """Evict results for hotkeys not seen in 3 tempos to prevent memory leaks."""
        stale_threshold = 3 * TEMPO_SECONDS
        now = time.monotonic()
        stale = [hk for hk, ts in list(self._last_seen.items()) if now - ts > stale_threshold]
        for hk in stale:
            self.results.pop(hk, None)
            self.validation_counts.pop(hk, None)
            self._last_seen.pop(hk, None)
        if stale:
            self._save_validation_counts()
            logger.info("Cleaned up %d stale hotkey results", len(stale))

    def _prune_stale_results(self, current_block: int) -> None:
        """Drop entries older than RESULT_RETENTION_BLOCKS so a passing
        miner's vote survives subsequent burn-only cycles long enough to
        reach the next chain tempo boundary, but doesn't accumulate
        forever. Replaces the old per-cycle ``self.results.clear()``
        which erased pass votes ~5 minutes after they were recorded —
        well before SN37's 72-minute tempo, leaving incentive=0 even
        when the pipeline accepted the miner. Entries with
        ``recorded_at_block is None`` (legacy / pre-stamp) are pruned
        too: they predate this code path and would otherwise stick
        around indefinitely."""
        cutoff = current_block - RESULT_RETENTION_BLOCKS
        stale = [hk for hk, r in self.results.items() if r.recorded_at_block is None or r.recorded_at_block < cutoff]
        for hk in stale:
            self.results.pop(hk, None)

    def _record_result(self, hotkey: str, result: PipelineResult, current_block: int) -> None:
        """Write a fresh result to self.results, with two invariants:

        1. A fresh fail must never displace a still-fresh prior pass —
           that would re-introduce the same overwrite bug for miners
           that pass once and then fail (or stop responding) for a few
           cycles before the next tempo boundary.
        2. A fresh fail *should* refresh a prior fail's
           ``recorded_at_block`` so the per-cycle summary log still
           counts the miner as failing this cycle. Without this, the
           same miner failing every cycle (e.g. no work-token balance)
           only shows up in cycle_summary on the cycle they first
           failed, and operators see misleading ``failures={}`` lines.
        """
        if result.weight > 0:
            # Pass — always overwrite, stamp.
            result.recorded_at_block = current_block
            self.results[hotkey] = result
            return
        # Fail (weight=0) — refresh unless we'd be displacing a still-
        # fresh pass. Fail-over-fail refreshes the timestamp so the
        # cycle-summary stage_failures counter for this hotkey reflects
        # the current cycle.
        existing = self.results.get(hotkey)
        if existing is None or existing.weight == 0:
            result.recorded_at_block = current_block
            self.results[hotkey] = result

    async def stop(self):
        logger.info("Stopping validator... (in-flight=%d, queued=%d)", self._in_flight, self.local_queue.size)

        # Wait for in-flight validations to complete (max 10s)
        deadline = time.monotonic() + 10
        while self._in_flight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if self._in_flight > 0:
            logger.warning("Shutdown timeout: %d validations still in-flight", self._in_flight)

        # Clean up simulation runner (Docker containers, networks, iptables rules)
        if hasattr(self, "pipeline") and self.pipeline._simulation_runner is not None:
            try:
                self.pipeline._simulation_runner.close()
                logger.info("Simulation runner closed")
            except Exception as e:
                logger.warning("Error closing simulation runner: %s", e)

        if self.api_client:
            try:
                await self.api_client.close()
            except Exception as e:
                logger.warning("Error closing API client: %s", e)
        self.dendrite.close_session()

    def _signal_handler(self, signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        self.should_exit = True


def _configure_logging():
    log_format = Config.LOG_FORMAT
    if log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger

            handler = logging.StreamHandler()
            handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            logging.root.handlers = [handler]
            logging.root.setLevel(logging.INFO)
        except ImportError:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Quiet third-party loggers that emit one INFO line per HTTP call.
    # Operators get a flood of per-request logs during a cycle; WARN here
    # means we only see them when something goes wrong.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    import sys

    # T-9: `aurelius-validator doctor` runs the preflight checks and
    # exits, without constructing a full Validator (which hits the
    # network and the wallet files). Short-circuit before _configure_logging
    # so the doctor's stdout isn't drowned in info-level boot noise.
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from aurelius.validator.doctor import main as _doctor_main

        sys.exit(_doctor_main())

    _configure_logging()
    validator = Validator()
    asyncio.run(validator.run())


if __name__ == "__main__":
    main()
