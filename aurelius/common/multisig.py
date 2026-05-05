"""Deterministic frame_multisig address derivation.

Substrate's `pallet_multisig` does not store "threshold + signatories" anywhere
on-chain — multisig accounts have no on-chain metadata and the AccountId is
derived purely from the sorted signatory pubkeys plus the threshold:

    account_id = blake2_256(b"modlpy/utilis" || sorted(pubkeys) || u16_le(threshold))

This module exposes that derivation as a pure function so the deposit CLI can
verify a claimed multisig address without a chain RPC.
"""

from __future__ import annotations

import struct
from hashlib import blake2b


def derive_multisig_address(
    signatories: list[str],
    threshold: int,
    ss58_format: int = 42,
) -> str:
    """Return the SS58 address of the multisig defined by `signatories` + `threshold`.

    Mirrors `pallet_multisig::Pallet::multi_account_id` and the canonical
    `MultiAccountId.create_from_account_list` implementation in
    py-substrate-interface. Signatories are sorted by their 32-byte public key
    before hashing, so the result is independent of input ordering.
    """
    from scalecodec.utils.ss58 import ss58_decode, ss58_encode

    if threshold < 1:
        raise ValueError(f"threshold must be >= 1, got {threshold}")
    if len(signatories) < threshold:
        raise ValueError(f"need at least {threshold} signatories to satisfy threshold, got {len(signatories)}")

    pubkeys = [bytes.fromhex(ss58_decode(s)) for s in signatories]
    pubkeys.sort()

    entropy = b"modlpy/utilis" + b"".join(pubkeys) + struct.pack("<H", threshold)
    digest = blake2b(entropy, digest_size=32).digest()
    return ss58_encode(digest, ss58_format)
