#!/usr/bin/env python3
"""Scan local wallet hotkeys and check aurelius-deposit balance for those registered on a subnet.

Resolves Central API URL for `aurelius-deposit` in this order:
  1. ``--api-url``
  2. Environment ``CENTRAL_API_URL`` or ``AURELIUS_API_URL`` (non-empty)
  3. ``aurelius.config.Config.CENTRAL_API_URL`` (after ``load_dotenv()`` — uses ``ENVIRONMENT``
     profile, e.g. ``mainnet`` → production collector URL)

Without ``--api-url`` or a profile pointing off localhost, balance checks hit ``localhost:8000``
and fail unless the Central API runs locally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple

SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,49}$")


def discover_local_hotkeys(wallets_dir: Path) -> Dict[str, List[str]]:
    """Return {ss58: [wallet/hotkey labels]} from *pub.txt files."""
    found: Dict[str, List[str]] = {}
    for pub_file in wallets_dir.glob("*/hotkeys/*pub.txt"):
        try:
            data = json.loads(pub_file.read_text())
        except Exception:
            continue

        ss58 = data.get("ss58Address")
        if not isinstance(ss58, str) or not SS58_RE.match(ss58):
            continue

        wallet_name = pub_file.parts[-3]
        hotkey_file = pub_file.name.replace("pub.txt", "")
        label = f"{wallet_name}/{hotkey_file}"
        found.setdefault(ss58, []).append(label)
    return found


def load_registered_hotkeys_from_file(path: Path) -> Set[str]:
    out: Set[str] = set()
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and SS58_RE.match(s):
            out.add(s)
    return out


def fetch_registered_hotkeys_sdk(
    netuid: int,
    network: str,
    chain_endpoint: str | None,
) -> Set[str]:
    try:
        import bittensor as bt
    except ImportError as e:
        raise RuntimeError(
            "bittensor is not installed in this Python environment. "
            "Activate your venv (e.g. btsdk_venv) or: pip install bittensor"
        ) from e

    if chain_endpoint:
        subtensor = bt.Subtensor(network=network, chain_endpoint=chain_endpoint)
    else:
        subtensor = bt.Subtensor(network=network)

    metagraph = bt.Metagraph(
        netuid=netuid,
        network=network,
        subtensor=subtensor,
    )
    metagraph.sync(subtensor=subtensor)

    out: Set[str] = set()
    n = int(metagraph.n)
    for uid in range(n):
        hk = metagraph.hotkeys[uid]
        if hk is None:
            continue
        s = str(hk).strip()
        if SS58_RE.match(s):
            out.add(s)
    return out


def resolve_central_api_url(cli_api_url: str | None) -> str:
    """Pick Central API base URL for ``aurelius-deposit --api-url``."""
    if cli_api_url and cli_api_url.strip():
        return cli_api_url.strip().rstrip("/")
    for key in ("CENTRAL_API_URL", "AURELIUS_API_URL"):
        v = os.environ.get(key, "").strip()
        if v:
            return v.rstrip("/")
    try:
        from aurelius.config import Config

        u = (Config.CENTRAL_API_URL or "").strip()
        if u:
            return u.rstrip("/")
    except Exception:
        pass
    return ""


def central_api_health_ok(api_url: str, timeout_sec: float = 8.0) -> bool:
    base = api_url.rstrip("/")
    health = f"{base}/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout_sec) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def build_deposit_balance_argv(deposit_cmd: str, api_url: str, hotkey: str) -> List[str]:
    """``aurelius-deposit --api-url … balance --hotkey …`` (global flags before subcommand)."""
    return shlex.split(deposit_cmd) + ["--api-url", api_url, "balance", "--hotkey", hotkey]


def run_deposit_balance(deposit_cmd: str, api_url: str, hotkey: str) -> Tuple[int, str, str]:
    cmd = build_deposit_balance_argv(deposit_cmd, api_url, hotkey)
    cp = subprocess.run(cmd, text=True, capture_output=True)
    return cp.returncode, cp.stdout, cp.stderr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find local hotkeys registered on a subnet and check deposit balance one by one."
    )
    parser.add_argument("--netuid", type=int, default=37, help="Subnet netuid (default: 37)")
    parser.add_argument(
        "--wallets-dir",
        default="/root/.bittensor/wallets",
        help="Path to bittensor wallets directory",
    )
    parser.add_argument(
        "--network",
        default="finney",
        help="Bittensor network name for Subtensor/Metagraph (default: finney)",
    )
    parser.add_argument(
        "--chain-endpoint",
        default=None,
        help="Optional custom chain RPC URL (passed to bittensor.Subtensor as chain_endpoint)",
    )
    parser.add_argument(
        "--deposit-cmd",
        default="aurelius-deposit",
        help="aurelius-deposit executable/command prefix (default: aurelius-deposit)",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("CENTRAL_API_URL") or os.environ.get("AURELIUS_API_URL") or None,
        metavar="URL",
        help=(
            "Central API base URL passed to aurelius-deposit (default: CENTRAL_API_URL or "
            "AURELIUS_API_URL env, else aurelius.config from .env / ENVIRONMENT profile)"
        ),
    )
    parser.add_argument(
        "--skip-api-health-check",
        action="store_true",
        help="Do not probe GET /health before running balance checks",
    )
    parser.add_argument(
        "--registered-hotkeys-file",
        help="Optional text file with one registered ss58 hotkey per line (skips metagraph fetch)",
    )
    parser.add_argument(
        "--show-not-registered",
        action="store_true",
        help="Print local hotkeys not registered on target subnet",
    )
    args = parser.parse_args()

    api_url = resolve_central_api_url(args.api_url)
    if not api_url:
        print(
            "[error] Central API URL is empty. Set CENTRAL_API_URL or AURELIUS_API_URL, "
            "use --api-url, or run from a directory whose .env sets ENVIRONMENT=mainnet "
            "(or testnet) so aurelius.config resolves a non-local collector URL.",
            file=sys.stderr,
        )
        return 1

    wallets_dir = Path(args.wallets_dir)
    if not wallets_dir.exists():
        print(f"[error] wallets dir not found: {wallets_dir}", file=sys.stderr)
        return 1

    local_hotkeys = discover_local_hotkeys(wallets_dir)
    if not local_hotkeys:
        print("[error] no local hotkeys found from *pub.txt files", file=sys.stderr)
        return 1

    try:
        if args.registered_hotkeys_file:
            registered = load_registered_hotkeys_from_file(Path(args.registered_hotkeys_file))
        else:
            registered = fetch_registered_hotkeys_sdk(
                args.netuid,
                args.network,
                args.chain_endpoint,
            )
    except Exception as e:
        print(f"[error] failed to get registered hotkeys: {e}", file=sys.stderr)
        return 2

    local_set = set(local_hotkeys.keys())
    targets = sorted(local_set & registered)
    not_registered = sorted(local_set - registered)

    print(f"[info] local hotkeys found: {len(local_set)}")
    print(f"[info] registered on netuid {args.netuid}: {len(targets)}")
    print(f"[info] Central API: {api_url}")

    if "localhost" in api_url or api_url.startswith("127."):
        print(
            "[warn] Using loopback Central API. For Finney/mainnet deposits use ENVIRONMENT=mainnet "
            "in .env or pass --api-url https://new-collector-api-production.up.railway.app",
            file=sys.stderr,
        )

    if not args.skip_api_health_check:
        if not central_api_health_ok(api_url):
            print(
                f"[error] Central API not reachable at {api_url}/health "
                "(connection refused or timeout). Fix --api-url / CENTRAL_API_URL, start the API, "
                "or pass --skip-api-health-check to try anyway.",
                file=sys.stderr,
            )
            return 4

    if args.show_not_registered and not_registered:
        print("\n[not-registered-local-hotkeys]")
        for hk in not_registered:
            print(f"- {hk} ({', '.join(local_hotkeys[hk])})")

    if not targets:
        print("[done] no local hotkeys registered on target subnet")
        return 0

    failures = 0
    print("\n[deposit-balance-check]")
    for idx, hk in enumerate(targets, 1):
        labels = ", ".join(local_hotkeys[hk])
        print(f"\n[{idx}/{len(targets)}] hotkey={hk} labels={labels}")
        code, stdout, stderr = run_deposit_balance(args.deposit_cmd, api_url, hk)
        if stdout.strip():
            print(stdout.strip())
        if code != 0:
            failures += 1
            print(f"[error] command failed (exit {code})", file=sys.stderr)
            if stderr.strip():
                print(stderr.strip(), file=sys.stderr)

    print(f"\n[done] checked={len(targets)} failed={failures}")
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
