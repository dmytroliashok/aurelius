"""Two-tier configuration system.

Local config: wallet, network, netuid, secrets (env vars / .env file).
Remote config: operational parameters fetched from Central API (cached with TTL).

Local config is loaded at startup and never changes.
Remote config is refreshed periodically by the validator.

Configuration resolution order:
  1. Explicit environment variable (always wins)
  2. Environment profile default (based on ENVIRONMENT)
  3. Hardcoded fallback

Set ENVIRONMENT=local|testnet|mainnet to switch profiles.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Environment profiles
# ---------------------------------------------------------------------------

ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")

_COMMON = {
    # Wallet
    "WALLET_NAME": "default",
    "WALLET_HOTKEY": "default",
    # Miner
    "AXON_PORT": "8091",
    "AXON_EXTERNAL_IP": "",
    "AXON_EXTERNAL_PORT": "",
    "MINER_CONFIG_DIR": "configs",
    # Validator
    "BURN_MODE": "0",
    "BURN_PERCENTAGE": "0.7",
    "WEIGHT_INTERVAL": "300",
    "CENTRAL_API_URL": "http://localhost:8000",
    # LLM (DeepSeek API — OpenAI-compatible)
    "LLM_API_KEY": "",
    "LLM_MODEL": "deepseek-chat",
    "LLM_BASE_URL": "https://api.deepseek.com/v1",
    # Work-token
    "DEPOSIT_ADDRESS": "",
    # Data paths (resolved under DATA_DIR; set absolute path to override)
    "SUBMISSION_QUEUE_PATH": "submission_queue.jsonl",
    "FAISS_INDEX_PATH": "novelty_index.faiss",
    "CLASSIFIER_MODEL_PATH": "classifier_model.joblib",
    # Logging
    "LOG_FORMAT": "text",
    # Simulation
    "CONCORDIA_IMAGE_NAME": "ghcr.io/aurelius-protocol/aurelius-concordia",
    "CONCORDIA_IMAGE_DIGEST": "sha256:da277d6c740fe68e9b922f78d111d25d730db79618c06e49729433fbe0fa5d6c",
    # CS-06: when "1", refuse to run simulations without a pinned image digest.
    # Defaults to "0" in local/testlab; set to "1" in mainnet/testnet profiles.
    "REQUIRE_IMAGE_DIGEST": "0",
    "SIM_NETWORK_NAME": "aurelius-sim-restricted",
    "SIM_DATA_DIR": "",
    "SIM_DATA_HOST_DIR": "",
    "SIM_BASE_TIMEOUT": "600",
    "SIM_BASE_RAM_MB": "4096",
    "SIM_CPU_COUNT": "2",
    # Egress allowlist for Concordia simulation containers — empty by default
    # (operators acknowledge unrestricted egress). Populate with a
    # comma-separated host list to re-enable iptables enforcement.
    "SIM_ALLOWED_LLM_HOSTS": "",
    # Validator pipeline
    "MAX_CONFIG_SIZE": "65536",
    "WORK_ID_FRESHNESS_SECONDS": "300",
    "MIN_CONSISTENCY_REPORTS": "10",
    "CONSISTENCY_FLOOR": "0.4",
    # Queue
    "QUEUE_MAX_SIZE": "500",
    "QUEUE_MAX_FILE_SIZE_MB": "50",
    "QUEUE_MAX_AGE_SECONDS": "12960",
    # Miner
    "METAGRAPH_SYNC_INTERVAL": "60",
    # HTTP client
    "API_TIMEOUT": "30",
}

PROFILES: dict[str, dict[str, str]] = {
    "local": {
        **_COMMON,
        "BT_SUBTENSOR_NETWORK": "test",
        "BT_NETUID": "455",
        "TESTLAB_MODE": "1",
        "QUERY_TIMEOUT": "30",
        "CONTAINER_POOL_SIZE": "0",
    },
    "testnet": {
        **_COMMON,
        "BT_SUBTENSOR_NETWORK": "test",
        "BT_NETUID": "455",
        "TESTLAB_MODE": "0",
        "QUERY_TIMEOUT": "12",
        "CONTAINER_POOL_SIZE": "2",
        "CENTRAL_API_URL": "https://new-collector-api-staging.up.railway.app",
        "SIM_DATA_DIR": "/sim-data",
        "REQUIRE_IMAGE_DIGEST": "1",
    },
    "mainnet": {
        **_COMMON,
        "BT_SUBTENSOR_NETWORK": "finney",
        "BT_NETUID": "37",
        "TESTLAB_MODE": "0",
        "QUERY_TIMEOUT": "12",
        "CONTAINER_POOL_SIZE": "2",
        "CENTRAL_API_URL": "https://new-collector-api-production.up.railway.app",
        "SIM_DATA_DIR": "/sim-data",
        "REQUIRE_IMAGE_DIGEST": "1",
    },
}

if ENVIRONMENT not in PROFILES:
    raise ValueError(f"Unknown ENVIRONMENT={ENVIRONMENT!r}. Must be one of: {', '.join(PROFILES)}")

_profile = PROFILES[ENVIRONMENT]


def _get(key: str, fallback: str = "") -> str:
    """Resolve config: env var (if non-empty) > profile default > fallback.

    Empty-string env vars are treated as unset so operators can't accidentally
    clobber a profile default by writing `KEY=` in their .env.
    """
    return os.getenv(key) or _profile.get(key, fallback)


# ---------------------------------------------------------------------------
# Data directory resolution
# ---------------------------------------------------------------------------

_DATA_DIR: str = os.getenv("DATA_DIR", "data")


def _resolve_data_path(key: str, fallback: str = "") -> str:
    """Resolve a data path: absolute paths pass through, relative paths resolve under DATA_DIR."""
    from pathlib import Path

    raw = _get(key, fallback)
    p = Path(raw)
    if p.is_absolute():
        return raw
    return str(Path(_DATA_DIR) / raw)


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


class LocalConfig:
    """Static configuration from environment variables with profile defaults."""

    # Bittensor network
    NETWORK: str = _get("BT_SUBTENSOR_NETWORK")
    NETUID: int = int(_get("BT_NETUID"))
    WALLET_NAME: str = _get("WALLET_NAME")
    WALLET_HOTKEY: str = _get("WALLET_HOTKEY")

    # Miner
    AXON_PORT: int = int(_get("AXON_PORT"))
    AXON_EXTERNAL_IP: str | None = _get("AXON_EXTERNAL_IP") or None
    AXON_EXTERNAL_PORT: int | None = int(_get("AXON_EXTERNAL_PORT")) if _get("AXON_EXTERNAL_PORT") else None
    MINER_CONFIG_DIR: str = _get("MINER_CONFIG_DIR")

    # Validator
    BURN_MODE: bool = _get("BURN_MODE") == "1"
    BURN_PERCENTAGE: float = max(0.0, min(1.0, float(_get("BURN_PERCENTAGE"))))
    WEIGHT_INTERVAL: int = int(_get("WEIGHT_INTERVAL"))
    QUERY_TIMEOUT: float = float(_get("QUERY_TIMEOUT"))

    # Central API
    CENTRAL_API_URL: str = _get("CENTRAL_API_URL")

    # LLM API (validator — Concordia simulation, OpenAI-compatible)
    LLM_API_KEY: str = _get("LLM_API_KEY")
    LLM_MODEL: str = _get("LLM_MODEL")
    LLM_BASE_URL: str = _get("LLM_BASE_URL")

    # Work-token (miner)
    DEPOSIT_ADDRESS: str = _get("DEPOSIT_ADDRESS")

    # Validator data paths (resolved under DATA_DIR unless absolute)
    SUBMISSION_QUEUE_PATH: str = _resolve_data_path("SUBMISSION_QUEUE_PATH", "submission_queue.jsonl")
    FAISS_INDEX_PATH: str = _resolve_data_path("FAISS_INDEX_PATH", "novelty_index.faiss")
    CLASSIFIER_MODEL_PATH: str = _resolve_data_path("CLASSIFIER_MODEL_PATH", "classifier_model.joblib")

    # Validator state persistence (resolved under DATA_DIR unless absolute)
    RATE_LIMITER_STATE_PATH: str = _resolve_data_path("RATE_LIMITER_STATE_PATH", "rate_limiter.json")
    VALIDATION_COUNTS_PATH: str = _resolve_data_path("VALIDATION_COUNTS_PATH", "validation_counts.json")
    # H-9: block number at which ramp-up started. Persisted so a restart
    # mid-ramp-up resumes where it left off instead of resetting the window.
    RAMP_UP_ANCHOR_PATH: str = _resolve_data_path("RAMP_UP_ANCHOR_PATH", "ramp_up_anchor.json")

    # Logging
    LOG_FORMAT: str = _get("LOG_FORMAT")

    # Testlab mode (disables validator-permit check on miners)
    TESTLAB_MODE: bool = _get("TESTLAB_MODE") == "1"

    # Simulation (Docker runner)
    CONCORDIA_IMAGE_NAME: str = _get("CONCORDIA_IMAGE_NAME")
    CONCORDIA_IMAGE_DIGEST: str = _get("CONCORDIA_IMAGE_DIGEST")
    REQUIRE_IMAGE_DIGEST: bool = _get("REQUIRE_IMAGE_DIGEST") == "1"
    SIM_NETWORK_NAME: str = _get("SIM_NETWORK_NAME")
    CONTAINER_POOL_SIZE: int = int(_get("CONTAINER_POOL_SIZE"))
    SIM_DATA_DIR: str | None = _get("SIM_DATA_DIR") or None
    SIM_DATA_HOST_DIR: str = _get("SIM_DATA_HOST_DIR")
    SIM_BASE_TIMEOUT: int = int(_get("SIM_BASE_TIMEOUT"))
    SIM_BASE_RAM_MB: int = int(_get("SIM_BASE_RAM_MB"))
    SIM_CPU_COUNT: int = int(_get("SIM_CPU_COUNT"))
    SIM_ALLOWED_LLM_HOSTS: list[str] = [h.strip() for h in _get("SIM_ALLOWED_LLM_HOSTS").split(",") if h.strip()]

    # Validator pipeline
    MAX_CONFIG_SIZE: int = int(_get("MAX_CONFIG_SIZE"))
    WORK_ID_FRESHNESS_SECONDS: int = int(_get("WORK_ID_FRESHNESS_SECONDS"))
    MIN_CONSISTENCY_REPORTS: int = int(_get("MIN_CONSISTENCY_REPORTS"))
    CONSISTENCY_FLOOR: float = float(_get("CONSISTENCY_FLOOR"))

    # Queue
    QUEUE_MAX_SIZE: int = int(_get("QUEUE_MAX_SIZE"))
    QUEUE_MAX_FILE_SIZE_MB: int = int(_get("QUEUE_MAX_FILE_SIZE_MB"))
    QUEUE_MAX_AGE_SECONDS: int = int(_get("QUEUE_MAX_AGE_SECONDS"))

    # Miner
    METAGRAPH_SYNC_INTERVAL: int = int(_get("METAGRAPH_SYNC_INTERVAL"))

    # HTTP client
    API_TIMEOUT: float = float(_get("API_TIMEOUT"))

    @classmethod
    def ensure_data_dirs(cls):
        """Create parent directories for data paths if they don't exist."""
        from pathlib import Path

        for path_attr in (
            "SUBMISSION_QUEUE_PATH",
            "FAISS_INDEX_PATH",
            "CLASSIFIER_MODEL_PATH",
            "RATE_LIMITER_STATE_PATH",
            "VALIDATION_COUNTS_PATH",
            "RAMP_UP_ANCHOR_PATH",
        ):
            path = getattr(cls, path_attr, "")
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)


# Backward-compatible alias
Config = LocalConfig
