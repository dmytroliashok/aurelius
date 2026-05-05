"""Tests for the rewritten deposit-verification CLI logic."""

import pytest

from aurelius.cli.deposit import _verify_address
from aurelius.common.central_api import DesignatedAddressResponse
from aurelius.common.multisig import derive_multisig_address

ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
CHARLIE = "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y"


def test_single_key_passes_with_warning(capsys):
    data = DesignatedAddressResponse(address=ALICE, multisig_threshold=None, signatories=None)
    _verify_address(data)  # must not raise / sys.exit
    captured = capsys.readouterr()
    assert "Single-key deposit address" in captured.out


def test_correct_multisig_passes(capsys):
    sigs = [ALICE, BOB, CHARLIE]
    derived = derive_multisig_address(sigs, 2)
    data = DesignatedAddressResponse(address=derived, multisig_threshold=2, signatories=sigs)
    _verify_address(data)
    captured = capsys.readouterr()
    assert "Verified multisig 2-of-3" in captured.out


def test_mismatched_multisig_exits_nonzero(capsys):
    sigs = [ALICE, BOB, CHARLIE]
    data = DesignatedAddressResponse(
        address="5Gx14QffqwC8wNHv4wUvfCfE2zAUYDNvF9Z7LjNY81WQx7iL",  # wrong claim
        multisig_threshold=2,
        signatories=sigs,
    )
    with pytest.raises(SystemExit) as exc_info:
        _verify_address(data)
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "does not match" in err


def test_threshold_exceeds_signatory_count_fails(capsys):
    data = DesignatedAddressResponse(address=ALICE, multisig_threshold=3, signatories=[ALICE])
    with pytest.raises(SystemExit) as exc_info:
        _verify_address(data)
    assert exc_info.value.code == 1
    assert "exceeds signatory count" in capsys.readouterr().err


def test_inconsistent_record_threshold_set_no_signatories_fails(capsys):
    # threshold present but signatories missing — exactly the malformed shape
    # the production API was returning before this fix.
    data = DesignatedAddressResponse(address=ALICE, multisig_threshold=2, signatories=None)
    with pytest.raises(SystemExit) as exc_info:
        _verify_address(data)
    assert exc_info.value.code == 1
    assert "inconsistent designated-address record" in capsys.readouterr().err


def test_inconsistent_record_signatories_set_no_threshold_fails(capsys):
    data = DesignatedAddressResponse(address=ALICE, multisig_threshold=None, signatories=[ALICE, BOB])
    with pytest.raises(SystemExit) as exc_info:
        _verify_address(data)
    assert exc_info.value.code == 1
    assert "inconsistent designated-address record" in capsys.readouterr().err
