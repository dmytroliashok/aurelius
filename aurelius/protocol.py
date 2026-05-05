from typing import ClassVar

import bittensor as bt

from aurelius.common.version import PROTOCOL_VERSION


class ScenarioConfigSynapse(bt.Synapse):
    """Synapse for exchanging moral dilemma scenario configurations.

    Validators set immutable fields before sending; miners populate mutable fields.

    VU2/VU3 VERSIONING POLICY:
    - New fields MUST be Optional with a default value (additive-only).
    - Removing a field or promoting Optional -> required is a BREAKING CHANGE
      that requires a MAJOR version bump in PROTOCOL_VERSION.
    - The version check in pipeline._version_check() enforces major mismatches.
    """

    # Immutable — set by validator
    request_id: str = ""
    validator_version: str = ""
    protocol_version: str = PROTOCOL_VERSION

    # Mutable — set by miner
    scenario_config: dict | None = None
    work_id: str | None = None
    work_id_nonce: str | None = None
    work_id_time_ns: str | None = None
    work_id_signature: str | None = None  # Miner's hotkey signature over work_id (ownership proof)
    miner_version: str | None = None
    miner_protocol_version: str | None = None

    required_hash_fields: ClassVar[tuple[str, ...]] = (
        "scenario_config",
        "work_id",
        "work_id_nonce",
        "work_id_time_ns",
        "work_id_signature",
        "miner_version",
        "miner_protocol_version",
    )
