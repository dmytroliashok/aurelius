"""Async HTTP client for the Aurelius Central API."""

import asyncio
import logging
import time

import httpx
from pydantic import BaseModel

from aurelius.common.types import ConsumeResult

logger = logging.getLogger(__name__)

# Retry configuration for transient HTTP failures
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds


async def _retry(coro_fn, retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY):
    """Call an async function with exponential backoff on transient errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_fn()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
            last_exc = e
            if attempt < retries - 1:
                delay = base_delay * (2**attempt)
                logger.debug("API call failed (attempt %d/%d), retrying in %.1fs: %s", attempt + 1, retries, delay, e)
                await asyncio.sleep(delay)
    raise last_exc


# --- Response models for schema validation (DM-06 fix) ---


class _BalanceResponse(BaseModel):
    has_balance: bool


class _ClassifierResponse(BaseModel):
    passed: bool
    confidence: float
    version: str


class _NoveltyResponse(BaseModel):
    novel: bool
    similarity: float = 0.0
    message: str = ""


class _ConsumeResponse(BaseModel):
    success: bool
    deducted: bool
    valid: bool
    message: str


class _ConsistencyResponse(BaseModel):
    agreement_rate: float = 1.0
    total_reports: int = 0


class _ChallengeResponse(BaseModel):
    challenge: str


class _VerifyResponse(BaseModel):
    token: str
    expires_at: str


class _RemoteConfigResponse(BaseModel):
    model_config = {"extra": "allow"}


class _SubmissionResponse(BaseModel):
    model_config = {"extra": "allow"}
    id: int | None = None


class _NoveltyAddResponse(BaseModel):
    model_config = {"extra": "allow"}


class _NoveltyRemoveResponse(BaseModel):
    model_config = {"extra": "allow"}
    removed: bool = False
    index_size: int = 0


# Re-authenticate 1 hour before token expires (default expiry is 24h)
_REAUTH_BUFFER_SECONDS = 3600


class CentralAPIClient:
    def __init__(self, base_url: str, auth_token: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._token = auth_token
        self._token_expires_at: float = 0.0  # monotonic time
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    def _headers(self) -> dict:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    def set_token(self, token: str, expires_in_hours: float = 24.0) -> None:
        self._token = token
        self._token_expires_at = time.monotonic() + (expires_in_hours * 3600)

    @property
    def needs_reauth(self) -> bool:
        if not self._token:
            return True
        return time.monotonic() >= (self._token_expires_at - _REAUTH_BUFFER_SECONDS)

    async def authenticate(self, wallet) -> None:
        """Perform challenge-response auth with the Central API.

        Args:
            wallet: A bittensor.Wallet instance used to sign the challenge.
        """
        hotkey_str = wallet.hotkey.ss58_address

        # Step 1: Request challenge
        resp = await self._client.post("/auth/validator/challenge", json={"hotkey": hotkey_str})
        resp.raise_for_status()
        challenge_data = _ChallengeResponse.model_validate(resp.json())

        # Step 2: Sign challenge with hotkey
        signature = wallet.hotkey.sign(challenge_data.challenge.encode()).hex()

        # Step 3: Verify and get JWT
        resp = await self._client.post(
            "/auth/validator/verify",
            json={"hotkey": hotkey_str, "challenge": challenge_data.challenge, "signature": signature},
        )
        resp.raise_for_status()
        data = _VerifyResponse.model_validate(resp.json())
        self.set_token(data.token)
        logger.info("Authenticated with Central API (token expires: %s)", data.expires_at)

    # --- Config ---

    async def get_remote_config(self) -> dict:
        async def _call():
            resp = await self._client.get("/config", headers=self._headers())
            resp.raise_for_status()
            return _RemoteConfigResponse.model_validate(resp.json()).model_dump()

        return await _retry(_call)

    # --- Work Token ---

    async def check_balance(self, miner_hotkey: str) -> bool:
        async def _call():
            resp = await self._client.get(f"/work-token/balance/{miner_hotkey}", headers=self._headers())
            resp.raise_for_status()
            return _BalanceResponse.model_validate(resp.json()).has_balance

        return await _retry(_call)

    async def consume_work_token(
        self,
        miner_hotkey: str,
        work_id: str,
        config_hash: str = "",
        work_id_signature: str = "",
    ) -> ConsumeResult:
        payload: dict = {"miner_hotkey": miner_hotkey, "work_id": work_id}
        if config_hash:
            payload["config_hash"] = config_hash
        if work_id_signature:
            payload["work_id_signature"] = work_id_signature

        async def _call():
            resp = await self._client.post("/work-token/consume", json=payload, headers=self._headers())
            resp.raise_for_status()
            data = _ConsumeResponse.model_validate(resp.json())
            return ConsumeResult(success=data.success, deducted=data.deducted, valid=data.valid, message=data.message)

        return await _retry(_call)

    # --- Novelty ---

    async def check_novelty(
        self,
        embedding: list[float],
        threshold: float = 0.92,
        field_embeddings: dict[str, list[float]] | None = None,
    ) -> dict:
        body: dict = {"embedding": embedding, "threshold": threshold}
        if field_embeddings:
            body["field_embeddings"] = field_embeddings

        async def _call():
            resp = await self._client.post("/novelty/check", json=body, headers=self._headers())
            resp.raise_for_status()
            return _NoveltyResponse.model_validate(resp.json()).model_dump()

        return await _retry(_call)

    # --- Submissions ---

    async def report_submission(
        self,
        work_id: str,
        miner_hotkey: str,
        scenario_config: dict,
        classifier_score: float | None = None,
        simulation_transcript: dict | None = None,
    ) -> dict:
        payload = {
            "work_id": work_id,
            "miner_hotkey": miner_hotkey,
            "scenario_config": scenario_config,
            "classifier_score": classifier_score,
            "simulation_transcript": simulation_transcript,
        }

        async def _call():
            resp = await self._client.post("/submissions", json=payload, headers=self._headers())
            resp.raise_for_status()
            return _SubmissionResponse.model_validate(resp.json()).model_dump()

        return await _retry(_call)

    # --- Classifier ---

    async def classify_config(self, config: dict, threshold: float) -> dict:
        """Run classifier inference via the Central API.

        Returns:
            dict with 'passed' (bool), 'confidence' (float), 'version' (str).
        """

        async def _call():
            resp = await self._client.post(
                "/classifier/predict",
                json={"config": config, "threshold": threshold},
                headers=self._headers(),
            )
            resp.raise_for_status()
            return _ClassifierResponse.model_validate(resp.json()).model_dump()

        return await _retry(_call)

    # --- Consistency ---

    async def get_consistency(self, validator_hotkey: str) -> dict:
        """Get this validator's consistency score from the API."""
        resp = await self._client.get(f"/reports/consistency/{validator_hotkey}", headers=self._headers())
        resp.raise_for_status()
        data = _ConsistencyResponse.model_validate(resp.json())
        return data.model_dump()

    async def add_to_novelty_index(
        self,
        embedding: list[float],
        submission_id: int | None = None,
        config_hash: str | None = None,
    ) -> dict:
        """Add an embedding to the novelty/dedup index.

        Passing `config_hash` enables PO-07 rollback on consume failure.
        """
        body: dict = {"embedding": embedding}
        if submission_id is not None:
            body["submission_id"] = submission_id
        if config_hash is not None:
            body["config_hash"] = config_hash
        resp = await self._client.post("/novelty/add", json=body, headers=self._headers())
        resp.raise_for_status()
        return _NoveltyAddResponse.model_validate(resp.json()).model_dump()

    async def remove_from_novelty_index(self, config_hash: str) -> dict:
        """PO-07: roll back a previously-added embedding by its config hash.

        Called from the pipeline when stage-8 deduction fails after the
        embedding was added in stage 7c. Best-effort — callers should
        catch exceptions and continue.
        """
        resp = await self._client.post(
            "/novelty/remove",
            json={"config_hash": config_hash},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return _NoveltyRemoveResponse.model_validate(resp.json()).model_dump()

    async def close(self) -> None:
        await self._client.aclose()
