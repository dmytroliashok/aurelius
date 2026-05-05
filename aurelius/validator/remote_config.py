"""Fetches and caches remote configuration from the Central API.

The remote config controls all operational parameters that validators
and the API agree on. It's refreshed on a TTL basis and falls back
to local defaults if the API is unreachable.

Merge semantics:
  - In ENVIRONMENT == "local", local Config always wins (remote is ignored).
  - Elsewhere, remote wins when the key is present in the API response;
    otherwise falls back to local Config (or hardcoded defaults for
    remote-only fields like classifier_threshold).
"""

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from aurelius.common.constants import (
    DEFAULT_CLASSIFIER_THRESHOLD,
    DEFAULT_CLASSIFIER_UPDATE_INTERVAL_HOURS,
    DEFAULT_CONCORDIA_IMAGE_TAG,
    DEFAULT_CONCORDIA_LLM_MODEL,
    DEFAULT_CONCORDIA_TIMEOUT_SECONDS,
    DEFAULT_CUSTOM_ARCHETYPE_THRESHOLD_BUMP,
    DEFAULT_MAX_AGENTS,
    DEFAULT_MAX_PREMISE_LENGTH,
    DEFAULT_MIN_PREMISE_LENGTH,
    DEFAULT_NOVELTY_THRESHOLD,
    DEFAULT_POLLING_INTERVAL_SECONDS,
    DEFAULT_RATE_LIMIT_PER_UID_PER_TEMPO,
    DEFAULT_WORK_TOKEN_COST,
)
from aurelius.common.version import PROTOCOL_VERSION, SemanticVersion
from aurelius.config import ENVIRONMENT, Config

logger = logging.getLogger(__name__)


class RemoteConfig:
    """Cached remote configuration with TTL-based refresh and local-fallback merge.

    Falls back to local Config values if the API is unreachable or the key
    is not present in the remote response. In ENVIRONMENT == "local",
    local values always win so development works without a live API.
    """

    def __init__(
        self,
        api_client=None,
        refresh_interval: float = 300.0,
        local: Any = None,
        environment: str | None = None,
    ):
        self._api_client = api_client
        self._refresh_interval = refresh_interval
        self._last_fetch: float = 0
        self._last_success: float = 0
        self._config = self._defaults()
        self._fetch_failures: int = 0
        self._local = local if local is not None else Config
        self._environment = environment if environment is not None else ENVIRONMENT

    @staticmethod
    def _defaults() -> dict:
        return {
            "polling_interval_seconds": DEFAULT_POLLING_INTERVAL_SECONDS,
            "classifier_threshold": DEFAULT_CLASSIFIER_THRESHOLD,
            "novelty_threshold": DEFAULT_NOVELTY_THRESHOLD,
            "rate_limit_per_uid_per_tempo": DEFAULT_RATE_LIMIT_PER_UID_PER_TEMPO,
            "work_token_cost_per_unit": DEFAULT_WORK_TOKEN_COST,
            "concordia_timeout_seconds": DEFAULT_CONCORDIA_TIMEOUT_SECONDS,
            "concordia_llm_model": DEFAULT_CONCORDIA_LLM_MODEL,
            "concordia_image_tag": DEFAULT_CONCORDIA_IMAGE_TAG,
            "classifier_update_interval_hours": DEFAULT_CLASSIFIER_UPDATE_INTERVAL_HOURS,
            "min_premise_length": DEFAULT_MIN_PREMISE_LENGTH,
            "max_premise_length": DEFAULT_MAX_PREMISE_LENGTH,
            "max_agents": DEFAULT_MAX_AGENTS,
            "custom_archetype_threshold_bump": DEFAULT_CUSTOM_ARCHETYPE_THRESHOLD_BUMP,
            "min_validator_version": "1.0.0",
            "min_miner_version": "1.0.0",
            "semantic_coherence_enabled": True,
        }

    # Version-string keys that get bounded separately from numeric _BOUNDS.
    # A compromised Central API must not be able to disqualify the entire
    # subnet by pushing min_*_version to an unreachable future value.
    _VERSION_KEYS: frozenset[str] = frozenset({"min_miner_version", "min_validator_version"})
    # Max allowed gap between local PROTOCOL_VERSION.major and a remote
    # min_*_version.major. Legitimate major bumps arrive with operator notice;
    # anything beyond this is treated as a misconfiguration or attack.
    _MAX_MAJOR_DELTA: int = 1
    _VERSION_FALLBACK: str = "1.0.0"

    # Bounds for remote config values — prevents a compromised API from
    # disabling quality gates by setting extreme thresholds.
    _BOUNDS: dict[str, tuple[float, float]] = {
        "classifier_threshold": (0.1, 0.99),
        "novelty_threshold": (0.5, 0.99),
        "rate_limit_per_uid_per_tempo": (1, 100),
        "work_token_cost_per_unit": (0.001, 100.0),
        "concordia_timeout_seconds": (60, 3600),
        "max_agents": (2, 10),
        "custom_archetype_threshold_bump": (0.0, 0.5),
        "polling_interval_seconds": (30, 3600),
        # Overridable-local bounds
        "burn_percentage": (0.0, 1.0),
        "weight_interval": (60, 3600),
        "query_timeout": (1, 120),
        "container_pool_size": (0, 16),
        "sim_base_timeout": (60, 3600),
        "sim_base_ram_mb": (256, 16384),
        "sim_cpu_count": (1, 16),
        "max_config_size": (1024, 1_048_576),
        "work_id_freshness_seconds": (30, 3600),
        "min_consistency_reports": (1, 1000),
        "consistency_floor": (0.0, 1.0),
        "queue_max_size": (1, 10_000),
        "queue_max_file_size_mb": (1, 1024),
        "queue_max_age_seconds": (60, 604_800),
    }

    @staticmethod
    def _clamp_version(key: str, value: Any) -> str:
        """Validate a remote version-string.

        Returns the value if it parses as semver and its major is within
        ``_MAX_MAJOR_DELTA`` of the local PROTOCOL_VERSION.major. Otherwise
        returns ``_VERSION_FALLBACK`` and logs a warning — a permissive
        fallback that never disqualifies an honest node.
        """
        if not isinstance(value, str):
            logger.warning(
                "Remote config %s=%r has non-string type %s, using fallback %s",
                key,
                value,
                type(value).__name__,
                RemoteConfig._VERSION_FALLBACK,
            )
            return RemoteConfig._VERSION_FALLBACK
        try:
            remote_ver = SemanticVersion.parse(value)
        except (ValueError, AttributeError):
            logger.warning(
                "Remote config %s=%r is not a valid semver, using fallback %s",
                key,
                value,
                RemoteConfig._VERSION_FALLBACK,
            )
            return RemoteConfig._VERSION_FALLBACK
        local_major = SemanticVersion.parse(PROTOCOL_VERSION).major
        if remote_ver.major > local_major + RemoteConfig._MAX_MAJOR_DELTA:
            logger.warning(
                "Remote config %s=%s exceeds allowed major bound (local=%s, max delta=%d); "
                "refusing to apply, using fallback %s",
                key,
                value,
                PROTOCOL_VERSION,
                RemoteConfig._MAX_MAJOR_DELTA,
                RemoteConfig._VERSION_FALLBACK,
            )
            return RemoteConfig._VERSION_FALLBACK
        return value

    @staticmethod
    def _clamp(raw: dict) -> dict:
        """Enforce min/max bounds on remote config values.

        Numeric keys are clamped via ``_BOUNDS``; version strings go through
        ``_clamp_version``. Values not in either set pass through unchanged.
        """
        clamped = dict(raw)
        for key, (lo, hi) in RemoteConfig._BOUNDS.items():
            if key in clamped:
                try:
                    val = float(clamped[key])
                    if val < lo or val > hi:
                        logger.warning("Remote config %s=%s out of bounds [%s, %s], clamping", key, val, lo, hi)
                        val = max(lo, min(hi, val))
                    # Always store as numeric (int if whole number, float otherwise)
                    clamped[key] = int(val) if val == int(val) else val
                except (ValueError, TypeError):
                    pass
        for key in RemoteConfig._VERSION_KEYS:
            if key in clamped:
                clamped[key] = RemoteConfig._clamp_version(key, clamped[key])
        return clamped

    @property
    def is_stale(self) -> bool:
        """True if config hasn't been refreshed in 2× the refresh interval."""
        if self._last_success == 0:
            return True
        return (time.monotonic() - self._last_success) > (self._refresh_interval * 2)

    @property
    def api_available(self) -> bool:
        return self._fetch_failures == 0

    async def refresh(self) -> bool:
        """Fetch remote config if TTL has expired.

        Returns True if successfully refreshed, False if using cached/defaults.
        """
        now = time.monotonic()
        if now - self._last_fetch < self._refresh_interval:
            return self._fetch_failures == 0

        self._last_fetch = now

        if self._api_client is None:
            return False

        try:
            raw = await self._api_client.get_remote_config()
            self._config = self._clamp(raw)
            self._last_success = now
            self._fetch_failures = 0
            logger.info("Remote config refreshed")
            return True
        except Exception as e:
            self._fetch_failures += 1
            if self.is_stale:
                logger.warning(
                    "Remote config stale (last success: %.0fs ago, failures: %d): %s. Using cached values.",
                    now - self._last_success if self._last_success else float("inf"),
                    self._fetch_failures,
                    e,
                )
            else:
                logger.debug("Remote config fetch failed (%d): %s, using cached values", self._fetch_failures, e)
            return False

    def get(self, key: str, default=None):
        """Get a cached config value. Call refresh() separately to update."""
        return self._config.get(key, default)

    def _resolve(
        self,
        remote_key: str,
        local_attr: str,
        caster: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Merge-resolve a config value.

        - In ENVIRONMENT == "local", always return the local Config value.
        - Otherwise, prefer the remote value if present and non-empty, else
          fall back to local. Empty strings and None are treated as absent.
        """
        local_value = getattr(self._local, local_attr, None)
        if self._environment == "local":
            return local_value

        remote_value = self._config.get(remote_key)
        if remote_value is None or (isinstance(remote_value, str) and remote_value == ""):
            return local_value

        if caster is not None:
            try:
                return caster(remote_value)
            except (ValueError, TypeError):
                logger.warning(
                    "Remote config %s=%r failed %s coercion, falling back to local",
                    remote_key,
                    remote_value,
                    caster.__name__,
                )
                return local_value
        return remote_value

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    @staticmethod
    def _as_host_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return []

    # --- Remote-only typed properties ---

    @property
    def polling_interval(self) -> int:
        return self.get("polling_interval_seconds", DEFAULT_POLLING_INTERVAL_SECONDS)

    @property
    def classifier_threshold(self) -> float:
        return self.get("classifier_threshold", DEFAULT_CLASSIFIER_THRESHOLD)

    @property
    def novelty_threshold(self) -> float:
        return self.get("novelty_threshold", DEFAULT_NOVELTY_THRESHOLD)

    @property
    def rate_limit_per_uid_per_tempo(self) -> int:
        return self.get("rate_limit_per_uid_per_tempo", DEFAULT_RATE_LIMIT_PER_UID_PER_TEMPO)

    @property
    def work_token_cost(self) -> float:
        return self.get("work_token_cost_per_unit", DEFAULT_WORK_TOKEN_COST)

    @property
    def max_agents(self) -> int:
        return self.get("max_agents", DEFAULT_MAX_AGENTS)

    @property
    def concordia_timeout(self) -> int:
        return self.get("concordia_timeout_seconds", DEFAULT_CONCORDIA_TIMEOUT_SECONDS)

    @property
    def concordia_llm_model(self) -> str:
        return self.get("concordia_llm_model", DEFAULT_CONCORDIA_LLM_MODEL)

    @property
    def concordia_image_tag(self) -> str:
        return self.get("concordia_image_tag", DEFAULT_CONCORDIA_IMAGE_TAG)

    @property
    def classifier_update_interval_hours(self) -> int:
        return self.get("classifier_update_interval_hours", DEFAULT_CLASSIFIER_UPDATE_INTERVAL_HOURS)

    @property
    def custom_archetype_threshold_bump(self) -> float:
        return self.get("custom_archetype_threshold_bump", DEFAULT_CUSTOM_ARCHETYPE_THRESHOLD_BUMP)

    @property
    def min_miner_version(self) -> str:
        return self.get("min_miner_version", "1.0.0")

    @property
    def min_validator_version(self) -> str:
        return self.get("min_validator_version", "1.0.0")

    @property
    def semantic_coherence_enabled(self) -> bool:
        val = self.get("semantic_coherence_enabled", True)
        return self._as_bool(val)

    # --- Overridable-local merge properties ---

    @property
    def burn_mode(self) -> bool:
        return self._resolve("burn_mode", "BURN_MODE", self._as_bool)

    @property
    def burn_percentage(self) -> float:
        return self._resolve("burn_percentage", "BURN_PERCENTAGE", float)

    @property
    def weight_interval(self) -> int:
        return self._resolve("weight_interval", "WEIGHT_INTERVAL", int)

    @property
    def query_timeout(self) -> float:
        return self._resolve("query_timeout", "QUERY_TIMEOUT", float)

    @property
    def container_pool_size(self) -> int:
        return self._resolve("container_pool_size", "CONTAINER_POOL_SIZE", int)

    @property
    def llm_model(self) -> str:
        return self._resolve("llm_model", "LLM_MODEL")

    @property
    def llm_base_url(self) -> str:
        return self._resolve("llm_base_url", "LLM_BASE_URL")

    @property
    def concordia_image_name(self) -> str:
        return self._resolve("concordia_image_name", "CONCORDIA_IMAGE_NAME")

    @property
    def concordia_image_digest(self) -> str:
        return self._resolve("concordia_image_digest", "CONCORDIA_IMAGE_DIGEST")

    @property
    def require_image_digest(self) -> bool:
        return self._resolve("require_image_digest", "REQUIRE_IMAGE_DIGEST", self._as_bool)

    @property
    def sim_network_name(self) -> str:
        return self._resolve("sim_network_name", "SIM_NETWORK_NAME")

    @property
    def sim_base_timeout(self) -> int:
        return self._resolve("sim_base_timeout", "SIM_BASE_TIMEOUT", int)

    @property
    def sim_base_ram_mb(self) -> int:
        return self._resolve("sim_base_ram_mb", "SIM_BASE_RAM_MB", int)

    @property
    def sim_cpu_count(self) -> int:
        return self._resolve("sim_cpu_count", "SIM_CPU_COUNT", int)

    @property
    def sim_allowed_llm_hosts(self) -> list[str]:
        # Explicit local opt-out: operator sets SIM_ALLOWED_LLM_HOSTS="" in
        # .env to signal "no sim egress enforcement", even on testnet/mainnet
        # where remote config is otherwise authoritative. Honor it as a hard
        # override so a stale remote-side allowlist can't resurrect rules on
        # hosts where the operator has explicitly acknowledged unrestricted
        # egress.
        if os.environ.get("SIM_ALLOWED_LLM_HOSTS") == "":
            return []
        # Handle local-env / missing-remote fallback using the already-list Config attr
        if self._environment == "local":
            return list(getattr(self._local, "SIM_ALLOWED_LLM_HOSTS", []) or [])
        raw = self._config.get("sim_allowed_llm_hosts")
        if raw is None or raw == "" or raw == []:
            return list(getattr(self._local, "SIM_ALLOWED_LLM_HOSTS", []) or [])
        return self._as_host_list(raw)

    @property
    def sim_data_dir(self) -> str | None:
        val = self._resolve("sim_data_dir", "SIM_DATA_DIR")
        return val or None

    @property
    def sim_data_host_dir(self) -> str:
        return self._resolve("sim_data_host_dir", "SIM_DATA_HOST_DIR") or ""

    @property
    def max_config_size(self) -> int:
        return self._resolve("max_config_size", "MAX_CONFIG_SIZE", int)

    @property
    def work_id_freshness_seconds(self) -> int:
        return self._resolve("work_id_freshness_seconds", "WORK_ID_FRESHNESS_SECONDS", int)

    @property
    def min_consistency_reports(self) -> int:
        return self._resolve("min_consistency_reports", "MIN_CONSISTENCY_REPORTS", int)

    @property
    def consistency_floor(self) -> float:
        return self._resolve("consistency_floor", "CONSISTENCY_FLOOR", float)

    @property
    def queue_max_size(self) -> int:
        return self._resolve("queue_max_size", "QUEUE_MAX_SIZE", int)

    @property
    def queue_max_file_size_mb(self) -> int:
        return self._resolve("queue_max_file_size_mb", "QUEUE_MAX_FILE_SIZE_MB", int)

    @property
    def queue_max_age_seconds(self) -> int:
        return self._resolve("queue_max_age_seconds", "QUEUE_MAX_AGE_SECONDS", int)

    @property
    def gatekeeper_prompt(self) -> str:
        """LLM rubric (served by Central API) the validator consults between
        the Concordia simulation and work-token deduction. Empty string
        disables the stage. Purely remote-controlled — no local env override
        so a leaked local env can't disable the gate on a misconfigured node.
        """
        raw = self._config.get("gatekeeper_prompt")
        if not raw:
            return ""
        return str(raw)
