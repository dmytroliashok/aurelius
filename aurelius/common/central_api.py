"""Shared synchronous client for the Central API endpoints used by the miner
startup banner and the `aurelius-deposit` CLI.

The validator uses an async client (`aurelius.validator.api_client`) on its hot
path. This module is deliberately synchronous and dependency-light: it serves
two single-shot consumers (a CLI command and a one-time startup log line) where
async machinery would be ceremony.

Errors are surfaced as `CentralAPIError` with the response body included in the
message — bare HTTP status codes are useless for operators diagnosing why a
request was rejected.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel


class CentralAPIError(Exception):
    """Raised on any failure to reach or parse a Central API response."""


class DesignatedAddressResponse(BaseModel):
    address: str
    multisig_threshold: int | None = None
    signatories: list[str] | None = None


class BalanceResponse(BaseModel):
    hotkey: str
    balance: float | None = None
    has_balance: bool


class CentralAPIClient:
    """Sync HTTP client for the public, unauthenticated Central API endpoints
    needed by the miner startup banner and the deposit CLI."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)

    def __enter__(self) -> CentralAPIClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str) -> dict:
        try:
            r = self._client.get(path)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "").strip() or "<empty body>"
            raise CentralAPIError(
                f"{e.response.status_code} {e.response.reason_phrase} from {self._base_url}{path}: {body}"
            ) from e
        except httpx.RequestError as e:
            raise CentralAPIError(f"Could not reach Central API at {self._base_url}: {e}") from e

    def get_designated_address(self) -> DesignatedAddressResponse:
        return DesignatedAddressResponse.model_validate(self._get("/work-token/designated-address"))

    def get_balance(self, hotkey: str) -> BalanceResponse:
        return BalanceResponse.model_validate(self._get(f"/work-token/balance/{hotkey}"))
