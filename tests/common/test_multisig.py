"""Tests for deterministic frame_multisig address derivation."""

import pytest

from aurelius.common.multisig import derive_multisig_address

# Well-known Polkadot dev keys (Alice, Bob, Charlie). The 2-of-3 multisig
# derived from these is a useful known fixture: it's reproducible, matches
# the polkadot.js / py-substrate-interface output, and exercises the
# sort-then-hash codepath end-to-end.
ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
CHARLIE = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"

EXPECTED_2_OF_3 = "5DYHJokMqSGX6fNy7EpZVeZvJUZedpVUvizur6EqeespKtxz"


def test_derive_known_2_of_3():
    assert derive_multisig_address([ALICE, BOB, CHARLIE], 2) == EXPECTED_2_OF_3


def test_derive_is_order_independent():
    a = derive_multisig_address([ALICE, BOB, CHARLIE], 2)
    b = derive_multisig_address([CHARLIE, ALICE, BOB], 2)
    c = derive_multisig_address([BOB, CHARLIE, ALICE], 2)
    assert a == b == c


def test_threshold_changes_result():
    a2 = derive_multisig_address([ALICE, BOB, CHARLIE], 2)
    a3 = derive_multisig_address([ALICE, BOB, CHARLIE], 3)
    assert a2 != a3


def test_signatory_set_changes_result():
    abc = derive_multisig_address([ALICE, BOB, CHARLIE], 2)
    ab = derive_multisig_address([ALICE, BOB], 2)
    assert abc != ab


def test_threshold_below_one_rejected():
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        derive_multisig_address([ALICE, BOB], 0)


def test_threshold_exceeds_signatories_rejected():
    with pytest.raises(ValueError, match="need at least 3 signatories"):
        derive_multisig_address([ALICE, BOB], 3)
