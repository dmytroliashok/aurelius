"""Work-token ID generation for miner submissions."""

import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WorkIdResult:
    work_id: str
    nonce: str
    time_ns: str
    signature: str = ""


def generate_work_id(config: dict, hotkey: str, wallet=None) -> WorkIdResult:
    """Generate a unique work ID for a submission.

    work_id = sha256(json.dumps(config, sort_keys=True) + hotkey + time_ns + nonce)
    Timestamp is nanosecond resolution for debugging; uniqueness from 128-bit nonce.
    Returns all components so the validator can recompute.

    If a wallet is provided, the work_id is signed with the miner's hotkey
    to prove ownership. The signature is verified server-side by the Central
    API to prevent rogue validators from fabricating consume requests.
    """
    config_json = json.dumps(config, sort_keys=True)
    time_ns = str(time.time_ns())
    nonce = secrets.token_hex(16)  # 128-bit
    payload = config_json + hotkey + time_ns + nonce
    work_id = hashlib.sha256(payload.encode()).hexdigest()

    signature = ""
    if wallet is not None:
        try:
            signature = wallet.hotkey.sign(f"aurelius-worktoken-v1:{work_id}".encode()).hex()
        except Exception as e:
            logger.warning("Failed to sign work_id (miner will send unsigned): %s", e)

    return WorkIdResult(work_id=work_id, nonce=nonce, time_ns=time_ns, signature=signature)


def recompute_work_id(config: dict, hotkey: str, time_ns: str, nonce: str) -> str:
    """Recompute a work ID from its components (used by validators to verify)."""
    config_json = json.dumps(config, sort_keys=True)
    payload = config_json + hotkey + time_ns + nonce
    return hashlib.sha256(payload.encode()).hexdigest()
