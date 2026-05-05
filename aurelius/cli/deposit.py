"""CLI tool for work-token deposit address verification and balance queries."""

import argparse
import sys

from aurelius.common.central_api import CentralAPIClient, CentralAPIError, DesignatedAddressResponse
from aurelius.common.multisig import derive_multisig_address


def _verify_address(data: DesignatedAddressResponse) -> None:
    """Verify the API-claimed deposit address matches what its declared signatory
    set would derive to. Exits non-zero on inconsistency.

    Multisig accounts have no on-chain "threshold + signatories" record — those
    are deterministically derived from the inputs. So verification is an
    off-chain comparison: derive the multisig AccountId from
    `(signatories, threshold)` ourselves and check it against the API's claim.

    For single-key deposit addresses the API returns null for both fields; in
    that case there is nothing to derive, and we print a clear warning so the
    operator knows the address is not multisig-protected.
    """
    threshold = data.multisig_threshold
    signatories = data.signatories

    if threshold is None and signatories is None:
        print("Single-key deposit address (operator-controlled). No multisig verification possible.")
        print("Confirm out-of-band that the address belongs to a trusted operator before depositing.")
        return

    if threshold is None or signatories is None or threshold < 1 or len(signatories) < 1:
        print(
            "  ✗ VERIFICATION FAILED: API returned an inconsistent designated-address "
            f"record (multisig_threshold={threshold!r}, signatories={signatories!r})",
            file=sys.stderr,
        )
        print("DO NOT deposit to this address until the mismatch is resolved.", file=sys.stderr)
        sys.exit(1)

    if threshold > len(signatories):
        print(
            f"  ✗ VERIFICATION FAILED: threshold ({threshold}) exceeds signatory count ({len(signatories)})",
            file=sys.stderr,
        )
        print("DO NOT deposit to this address until the mismatch is resolved.", file=sys.stderr)
        sys.exit(1)

    try:
        derived = derive_multisig_address(signatories, threshold)
    except Exception as e:
        print(f"  ✗ VERIFICATION FAILED: could not derive multisig address: {e}", file=sys.stderr)
        sys.exit(1)

    if derived != data.address:
        print(
            f"  ✗ VERIFICATION FAILED: derived multisig {derived} does not match API-claimed address {data.address}",
            file=sys.stderr,
        )
        print("DO NOT deposit to this address until the mismatch is resolved.", file=sys.stderr)
        sys.exit(1)

    print(f"  ✓ Verified multisig {threshold}-of-{len(signatories)} → {data.address}")


def cmd_verify_address(args):
    """Fetch designated address from API and verify the multisig claim is internally consistent."""
    try:
        with CentralAPIClient(args.api_url) as client:
            data = client.get_designated_address()
    except CentralAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Designated deposit address: {data.address}")
    if data.multisig_threshold is not None and data.signatories is not None:
        print(f"Multisig threshold:         {data.multisig_threshold}-of-{len(data.signatories)}")
        print("Signatories:")
        for s in data.signatories:
            print(f"  - {s}")
    else:
        print("Multisig threshold:         (single-key)")
    print()

    print("Verifying designated-address record...")
    _verify_address(data)


def cmd_balance(args):
    """Query work-token balance for a hotkey."""
    try:
        with CentralAPIClient(args.api_url) as client:
            data = client.get_balance(args.hotkey)
    except CentralAPIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Hotkey:  {data.hotkey}")
    print(f"Balance: {data.balance if data.balance is not None else '(hidden)'}")
    print(f"Active:  {'yes' if data.has_balance else 'no'}")


def main():
    from aurelius.config import ENVIRONMENT, Config

    parser = argparse.ArgumentParser(prog="aurelius-deposit", description="Aurelius work-token deposit tools")
    parser.add_argument(
        "--api-url", default=Config.CENTRAL_API_URL, help="Central API base URL (default: $CENTRAL_API_URL)"
    )
    default_network = "test" if ENVIRONMENT in ("local", "testnet") else "finney"
    parser.add_argument(
        "--network",
        default=default_network,
        choices=["finney", "test"],
        help=f"Bittensor network (default: {default_network}, based on ENVIRONMENT={ENVIRONMENT})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify-address", help="Show and verify the designated deposit address")

    balance_parser = subparsers.add_parser("balance", help="Query work-token balance")
    balance_parser.add_argument("--hotkey", required=True, help="Miner hotkey (SS58 address)")

    args = parser.parse_args()

    if args.command == "verify-address":
        cmd_verify_address(args)
    elif args.command == "balance":
        cmd_balance(args)


if __name__ == "__main__":
    main()
