# Polling & timing
DEFAULT_POLLING_INTERVAL_SECONDS = 120
DEFAULT_QUERY_TIMEOUT_SECONDS = 30
TEMPO_BLOCKS = 360
TEMPO_SECONDS = TEMPO_BLOCKS * 12  # ~4320s (~72 min)

# Validator ramp-up
RAMP_UP_TEMPOS = 3
MIN_VALIDATIONS_FOR_WEIGHT = 2

# How long a passing pipeline result stays in Validator.results after the
# cycle that recorded it. The Bittensor chain reads each validator's most
# recent set_weights submission at every tempo boundary; without retention
# a miner that passed once but didn't pass the next 5-minute cycle would
# be overwritten with burn-only weights long before the next tempo, and
# the chain would never see the pass. One full tempo (360 blocks ≈ 72 min)
# is the minimum that guarantees survival across one boundary.
RESULT_RETENTION_BLOCKS = TEMPO_BLOCKS

# Scenario constraints (defaults, overridable via remote config)
DEFAULT_MAX_AGENTS = 2
DEFAULT_MIN_PREMISE_LENGTH = 200
DEFAULT_MAX_PREMISE_LENGTH = 2000

# Classifier
DEFAULT_CLASSIFIER_THRESHOLD = 0.5
DEFAULT_CLASSIFIER_UPDATE_INTERVAL_HOURS = 6

# Novelty
DEFAULT_NOVELTY_THRESHOLD = 0.88

# Rate limiting
DEFAULT_RATE_LIMIT_PER_UID_PER_TEMPO = 1

# Work token — cost per submission in alpha (raised from 0.01 to mitigate Sybil)
DEFAULT_WORK_TOKEN_COST = 1.0

# Custom archetype conservative scoring — threshold bump for custom tension archetype
DEFAULT_CUSTOM_ARCHETYPE_THRESHOLD_BUMP = 0.1

# Concordia
DEFAULT_CONCORDIA_TIMEOUT_SECONDS = 600
DEFAULT_CONCORDIA_LLM_MODEL = "deepseek-chat"
DEFAULT_CONCORDIA_IMAGE_TAG = "v2.0.0"

# Validator operational defaults (overridable via remote config)
DEFAULT_BURN_MODE = False
DEFAULT_BURN_PERCENTAGE = 0.7

# Burn address UID — emissions routed here when burn_mode=True. Must be a UID
# that exists on both mainnet (SN37) and testnet (SN455) and is either
# unoccupied or controlled by the subnet owner. See B-4 / ASSERTIONS.md
# §Burn Mode. If either subnet's registration count ever drops below this,
# the validator will warn at startup and refuse to set weights.
BURN_UID = 200
DEFAULT_WEIGHT_INTERVAL = 300
DEFAULT_QUERY_TIMEOUT = 12.0
DEFAULT_CONTAINER_POOL_SIZE = 2
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"

# Simulation Docker defaults (overridable via remote config)
DEFAULT_CONCORDIA_IMAGE_NAME = "ghcr.io/aurelius-protocol/aurelius-concordia"
DEFAULT_CONCORDIA_IMAGE_DIGEST = ""
DEFAULT_REQUIRE_IMAGE_DIGEST = True
DEFAULT_SIM_NETWORK_NAME = "aurelius-sim-restricted"
DEFAULT_SIM_BASE_TIMEOUT = 600
DEFAULT_SIM_BASE_RAM_MB = 4096
DEFAULT_SIM_CPU_COUNT = 2
DEFAULT_SIM_ALLOWED_LLM_HOSTS = "api.deepseek.com,api.openai.com,api.anthropic.com"
DEFAULT_SIM_DATA_DIR = "/sim-data"
DEFAULT_SIM_DATA_HOST_DIR = ""

# Pipeline-level limits (overridable via remote config)
DEFAULT_MAX_CONFIG_SIZE = 65536
DEFAULT_WORK_ID_FRESHNESS_SECONDS = 300
DEFAULT_MIN_CONSISTENCY_REPORTS = 10
DEFAULT_CONSISTENCY_FLOOR = 0.4

# Local queue limits (overridable via remote config)
DEFAULT_QUEUE_MAX_SIZE = 500
DEFAULT_QUEUE_MAX_FILE_SIZE_MB = 50
DEFAULT_QUEUE_MAX_AGE_SECONDS = 12960

# Weight setting — graduated based on classifier confidence
WEIGHT_FAIL = 0.0
WEIGHT_MIN = 0.1  # Minimum weight for a passing submission


def compute_weight(classifier_score: float | None, threshold: float) -> float:
    """Compute graduated weight from classifier confidence.

    Maps classifier score above threshold linearly to [WEIGHT_MIN, 1.0].
    If no classifier score available (bootstrap), returns 1.0.
    """
    if classifier_score is None:
        return WEIGHT_FAIL
    if threshold >= 1.0:
        raise ValueError(f"Threshold must be < 1.0, got {threshold}")
    if classifier_score < threshold:
        return WEIGHT_FAIL
    return max(WEIGHT_MIN, (classifier_score - threshold) / (1.0 - threshold))
