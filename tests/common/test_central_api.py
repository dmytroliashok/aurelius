"""Tests for the shared sync CentralAPIClient."""

import httpx
import pytest

from aurelius.common.central_api import (
    BalanceResponse,
    CentralAPIClient,
    CentralAPIError,
    DesignatedAddressResponse,
)


def _client_with_handler(handler) -> CentralAPIClient:
    """Wire an httpx MockTransport into a CentralAPIClient instance."""
    client = CentralAPIClient("http://test.local", timeout=1.0)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test.local")
    return client


def test_get_designated_address_single_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/work-token/designated-address"
        return httpx.Response(200, json={
            "address": "5Gx14QffqwC8wNHv4wUvfCfE2zAUYDNvF9Z7LjNY81WQx7iL",
            "multisig_threshold": None,
            "signatories": None,
        })

    with _client_with_handler(handler) as c:
        resp = c.get_designated_address()

    assert isinstance(resp, DesignatedAddressResponse)
    assert resp.multisig_threshold is None
    assert resp.signatories is None


def test_get_designated_address_multisig():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "address": "5DYHJokMqSGX6fNy7EpZVeZvJUZedpVUvizur6EqeespKtxz",
            "multisig_threshold": 2,
            "signatories": ["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"],
        })

    with _client_with_handler(handler) as c:
        resp = c.get_designated_address()

    assert resp.multisig_threshold == 2
    assert len(resp.signatories) == 2


def test_get_balance_returns_value():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/work-token/balance/HOTKEY"
        return httpx.Response(200, json={"hotkey": "HOTKEY", "balance": 4.2, "has_balance": True})

    with _client_with_handler(handler) as c:
        resp = c.get_balance("HOTKEY")

    assert isinstance(resp, BalanceResponse)
    assert resp.balance == 4.2
    assert resp.has_balance is True


def test_http_error_includes_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Not authenticated"})

    with _client_with_handler(handler) as c:
        with pytest.raises(CentralAPIError) as exc_info:
            c.get_balance("HOTKEY")

    msg = str(exc_info.value)
    assert "401" in msg
    assert "Not authenticated" in msg
    assert "/work-token/balance/HOTKEY" in msg


def test_request_error_surfaces_origin():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client_with_handler(handler) as c:
        with pytest.raises(CentralAPIError) as exc_info:
            c.get_designated_address()

    assert "Could not reach Central API at http://test.local" in str(exc_info.value)
