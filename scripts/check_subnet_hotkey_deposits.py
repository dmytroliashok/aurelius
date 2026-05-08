#!/usr/bin/env python3
"""Scan local wallet hotkeys and check aurelius-deposit balance for those registered on a subnet."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
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


def run_deposit_balance(deposit_cmd: str, hotkey: str) -> Tuple[int, str, str]:
    cmd = shlex.split(deposit_cmd) + ["balance", "--hotkey", hotkey]
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
        "--registered-hotkeys-file",
        help="Optional text file with one registered ss58 hotkey per line (skips metagraph fetch)",
    )
    parser.add_argument(
        "--show-not-registered",
        action="store_true",
        help="Print local hotkeys not registered on target subnet",
    )
    args = parser.parse_args()

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
        code, stdout, stderr = run_deposit_balance(args.deposit_cmd, hk)
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
