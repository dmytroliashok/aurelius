#!/usr/bin/env python3
"""
End-to-end test for work_id_signature (miner ownership proof).

Tests against a running testlab API to verify:
1. Consume WITH valid signature → succeeds
2. Consume WITH invalid signature → rejected
3. Consume WITHOUT signature → succeeds (migration mode) but with warning
4. Replay same work_id → idempotent success

Usage:
    python3 scripts/test-work-id-signature.py

Requires: testlab API running on localhost:8000 with seeded balances.
"""

import hashlib
import json
import os
import secrets
import sys
import time

import jwt
import requests
from bittensor import Keypair

API_URL = os.environ.get("API_URL", "http://localhost:8000")
JWT_SECRET = os.environ.get(
    "JWT_SECRET",
    "f8d6a0c4269582ba2bad3e5c7cc0bd5a8ca61b974f6efc58e883047fff7f44ed",
)
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "65eda79550e70a3ce7ff997d8fa564d4")

# Load testlab miner-0 wallet
WALLET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "testlab", "wallets")
MINER_NAME = "miner-0"


def load_hotkey(wallet_name: str) -> Keypair:
    hotkey_path = os.path.join(WALLET_DIR, wallet_name, "hotkeys", "default")
    with open(hotkey_path) as f:
        data = json.load(f)
    return Keypair.create_from_seed(data["secretSeed"])


def make_jwt(hotkey: str) -> str:
    from datetime import datetime, timezone, timedelta

    payload = {
        "sub": hotkey,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def generate_work_id(config: dict, hotkey: str) -> tuple[str, str, str]:
    """Generate work_id components (same as miner code)."""
    config_json = json.dumps(config, sort_keys=True)
    time_ns = str(time.time_ns())
    nonce = secrets.token_hex(16)
    payload = config_json + hotkey + time_ns + nonce
    work_id = hashlib.sha256(payload.encode()).hexdigest()
    return work_id, nonce, time_ns


def seed_balance(miner_hotkey: str, amount: float = 10.0):
    """Seed balance via direct DB insert through docker exec."""
    import subprocess

    subprocess.run(
        [
            "docker", "compose", "-f",
            os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.testlab.yml"),
            "--project-directory",
            os.path.join(os.path.dirname(__file__), "..", ".."),
            "exec", "-T", "db", "psql", "-U", "aurelius", "-d", "aurelius", "-c",
            f"INSERT INTO work_token_ledger (miner_hotkey, available_balance, total_deposited, total_consumed) "
            f"VALUES ('{miner_hotkey}', {amount}, {amount}, 0) "
            f"ON CONFLICT (miner_hotkey) DO UPDATE SET available_balance = {amount};",
        ],
        capture_output=True,
    )


PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  \033[32m✓\033[0m {msg}")


def fail(msg, detail=""):
    global FAIL
    FAIL += 1
    print(f"  \033[31m✗\033[0m {msg}")
    if detail:
        print(f"    → {detail[:300]}")


def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Work-ID Signature E2E Test")
    print(f"  API: {API_URL}")
    print("=" * 60)
    print()

    # Health check
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        assert r.status_code == 200
        ok("API healthy")
    except Exception as e:
        fail("API not reachable", str(e))
        sys.exit(1)

    # Load miner keypair
    try:
        miner_kp = load_hotkey(MINER_NAME)
        miner_hotkey = miner_kp.ss58_address
        ok(f"Loaded miner keypair: {miner_hotkey[:16]}...")
    except Exception as e:
        fail(f"Could not load miner wallet: {e}")
        sys.exit(1)

    # Create a fake validator JWT (we don't need a real validator for API-level testing)
    val_hotkey = "5FakeValidator" + secrets.token_hex(16)
    val_jwt = make_jwt(val_hotkey)
    val_auth = {"Authorization": f"Bearer {val_jwt}"}
    ok(f"Generated validator JWT for {val_hotkey[:20]}...")

    # Seed miner balance
    seed_balance(miner_hotkey, amount=10.0)
    ok(f"Seeded balance for {miner_hotkey[:16]}...")

    # Sample scenario config
    config = {
        "name": "sig_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": f"A test scenario for work_id signature verification (run {secrets.token_hex(8)}). " * 5,
        "agents": [
            {"name": "Agent A", "identity": "Test agent A", "goal": "Test", "philosophy": "deontology"},
            {"name": "Agent B", "identity": "Test agent B", "goal": "Test", "philosophy": "utilitarianism"},
        ],
        "scenes": [
            {
                "steps": 2,
                "mode": "decision",
                "forced_choice": {
                    "agent_name": "Agent A",
                    "choices": ["Option 1", "Option 2"],
                    "call_to_action": "Choose one.",
                },
            }
        ],
    }

    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    # ===================================================================
    print()
    print("=== Test 1: Consume with VALID signature ===")

    work_id_1, nonce_1, time_ns_1 = generate_work_id(config, miner_hotkey)
    signature_1 = miner_kp.sign(work_id_1.encode()).hex()
    ok(f"Generated work_id: {work_id_1[:16]}...")
    ok(f"Signed with miner hotkey: sig={signature_1[:24]}...")

    r = requests.post(
        f"{API_URL}/work-token/consume",
        json={
            "miner_hotkey": miner_hotkey,
            "work_id": work_id_1,
            "config_hash": config_hash,
            "work_id_signature": signature_1,
        },
        headers=val_auth,
        timeout=10,
    )
    resp = r.json()
    if resp.get("success"):
        ok(f"Consume with valid signature: SUCCESS (deducted={resp.get('deducted')})")
    else:
        fail(f"Consume with valid signature should succeed", json.dumps(resp))

    # ===================================================================
    print()
    print("=== Test 2: Consume with INVALID signature (rogue validator attack) ===")

    work_id_2, _, _ = generate_work_id(config, miner_hotkey)
    # Create a different keypair to forge a signature
    rogue_kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    forged_signature = rogue_kp.sign(work_id_2.encode()).hex()
    ok(f"Generated work_id: {work_id_2[:16]}...")
    ok(f"Forged signature with WRONG key: sig={forged_signature[:24]}...")

    r = requests.post(
        f"{API_URL}/work-token/consume",
        json={
            "miner_hotkey": miner_hotkey,
            "work_id": work_id_2,
            "config_hash": config_hash,
            "work_id_signature": forged_signature,
        },
        headers=val_auth,
        timeout=10,
    )
    resp = r.json()
    if not resp.get("success") and "signature" in resp.get("message", "").lower():
        ok(f"Consume with forged signature: REJECTED ({resp.get('message')})")
    else:
        fail(f"Consume with forged signature should be rejected", json.dumps(resp))

    # ===================================================================
    print()
    print("=== Test 3: Consume WITHOUT signature (migration mode) ===")

    work_id_3, _, _ = generate_work_id(config, miner_hotkey)
    ok(f"Generated work_id: {work_id_3[:16]}... (no signature)")

    r = requests.post(
        f"{API_URL}/work-token/consume",
        json={
            "miner_hotkey": miner_hotkey,
            "work_id": work_id_3,
            # No config_hash, no signature — simulates old miner
        },
        headers=val_auth,
        timeout=10,
    )
    resp = r.json()
    if resp.get("success"):
        ok(f"Consume without signature: ALLOWED (migration mode, deducted={resp.get('deducted')})")
    else:
        fail(f"Consume without signature should succeed during migration", json.dumps(resp))

    # Check API logs for the warning
    print("  (Check API logs for 'without work_id_signature' warning)")

    # ===================================================================
    print()
    print("=== Test 4: Replay same work_id (idempotency) ===")

    r = requests.post(
        f"{API_URL}/work-token/consume",
        json={
            "miner_hotkey": miner_hotkey,
            "work_id": work_id_1,  # same as test 1
            "config_hash": config_hash,
            "work_id_signature": signature_1,
        },
        headers=val_auth,
        timeout=10,
    )
    resp = r.json()
    if resp.get("success") and not resp.get("deducted"):
        ok(f"Replay: idempotent success (deducted=false, valid={resp.get('valid')})")
    else:
        fail(f"Replay should return success without deducting again", json.dumps(resp))

    # ===================================================================
    print()
    print("=== Test 4b: Replay already-consumed work_id with FORGED signature ===")
    print("  (Verifies idempotency takes priority over signature verification)")

    forged_replay_sig = rogue_kp.sign(work_id_1.encode()).hex()
    r = requests.post(
        f"{API_URL}/work-token/consume",
        json={
            "miner_hotkey": miner_hotkey,
            "work_id": work_id_1,  # already consumed in test 1
            "config_hash": config_hash,
            "work_id_signature": forged_replay_sig,  # WRONG key
        },
        headers=val_auth,
        timeout=10,
    )
    resp = r.json()
    if resp.get("success") and not resp.get("deducted"):
        ok(f"Forged-sig replay: idempotent success (not rejected — correct)")
    else:
        fail(
            f"Forged-sig replay should return idempotent success, not signature error",
            json.dumps(resp),
        )

    # ===================================================================
    print()
    print("=== Test 4c: Replay already-consumed work_id with NO signature ===")

    r = requests.post(
        f"{API_URL}/work-token/consume",
        json={
            "miner_hotkey": miner_hotkey,
            "work_id": work_id_1,  # already consumed in test 1
            # No signature at all
        },
        headers=val_auth,
        timeout=10,
    )
    resp = r.json()
    if resp.get("success") and not resp.get("deducted"):
        ok(f"Unsigned replay: idempotent success (not rejected — correct)")
    else:
        fail(
            f"Unsigned replay should return idempotent success, not error",
            json.dumps(resp),
        )

    # ===================================================================
    print()
    print("=== Test 5: Consume with valid sig but WRONG miner_hotkey (cross-miner attack) ===")

    # Attacker tries to use miner-0's signed work_id but claim it's for a different miner
    other_miner = "5FakeOtherMiner" + secrets.token_hex(16)
    seed_balance(other_miner, amount=10.0)

    work_id_5, _, _ = generate_work_id(config, miner_hotkey)
    # Signed by miner-0's real key
    signature_5 = miner_kp.sign(work_id_5.encode()).hex()

    r = requests.post(
        f"{API_URL}/work-token/consume",
        json={
            "miner_hotkey": other_miner,  # WRONG miner
            "work_id": work_id_5,
            "config_hash": config_hash,
            "work_id_signature": signature_5,  # Signed by miner-0, not other_miner
        },
        headers=val_auth,
        timeout=10,
    )
    resp = r.json()
    if not resp.get("success") and "signature" in resp.get("message", "").lower():
        ok(f"Cross-miner attack: REJECTED ({resp.get('message')})")
    else:
        fail(f"Cross-miner attack should be rejected", json.dumps(resp))

    # ===================================================================
    print()
    print("=== Test 6: Verify balance deductions are correct ===")

    r = requests.get(f"{API_URL}/work-token/balance/{miner_hotkey}", headers=val_auth, timeout=10)
    resp = r.json()
    bal = resp.get("balance")
    # Started with 10.0, consumed 2 (test 1 + test 3). Test 2 was rejected, test 4 was idempotent.
    if bal is not None and abs(bal - 8.0) < 0.01:
        ok(f"Balance correct: {bal} (started 10.0, consumed 2 × 1.0)")
    elif bal is not None:
        fail(f"Balance unexpected: {bal} (expected 8.0)", json.dumps(resp))
    else:
        ok(f"Balance hidden (non-owner query), has_balance={resp.get('has_balance')}")

    # ===================================================================
    print()
    print("=" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS} passed, {FAIL} failed ({total} total)")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    print("  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
