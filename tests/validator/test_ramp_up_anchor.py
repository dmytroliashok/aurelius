"""H-9: ramp-up window is anchored to a persisted block number, not to
process start time, so a restart doesn't rewind the window.
"""

import json
from unittest.mock import MagicMock

import pytest

from aurelius.common.constants import RAMP_UP_TEMPOS, TEMPO_BLOCKS
from aurelius.validator.validator import Validator


def _make_validator_with_anchor(tmp_path, anchor_file_contents=None, metagraph_block: int = 1000):
    """Build a bare Validator without running __init__, with just the bits
    the ramp-up anchor plumbing needs. Keeps tests quick and
    network-free while exercising production semantics."""
    v = Validator.__new__(Validator)
    v.start_time = 0.0
    v.metagraph = MagicMock()
    v.metagraph.block = metagraph_block
    v._ramp_up_anchor_path = str(tmp_path / "ramp_up_anchor.json")
    if anchor_file_contents is not None:
        with open(v._ramp_up_anchor_path, "w") as f:
            json.dump(anchor_file_contents, f)
    v._ramp_up_start_block = v._load_ramp_up_anchor()
    return v


class TestRampUpAnchorPersistence:
    def test_load_returns_none_when_file_missing(self, tmp_path):
        v = _make_validator_with_anchor(tmp_path)
        assert v._ramp_up_start_block is None

    def test_load_returns_block_when_file_present(self, tmp_path):
        v = _make_validator_with_anchor(tmp_path, anchor_file_contents={"block": 12345})
        assert v._ramp_up_start_block == 12345

    def test_load_rejects_malformed_file(self, tmp_path):
        path = tmp_path / "ramp_up_anchor.json"
        path.write_text("{not json")
        v = Validator.__new__(Validator)
        v._ramp_up_anchor_path = str(path)
        assert v._load_ramp_up_anchor() is None

    def test_ensure_captures_current_block_and_persists(self, tmp_path):
        v = _make_validator_with_anchor(tmp_path, metagraph_block=5000)
        assert v._ramp_up_start_block is None
        v._ensure_ramp_up_anchor()
        assert v._ramp_up_start_block == 5000
        # Persisted to disk.
        with open(v._ramp_up_anchor_path) as f:
            assert json.load(f) == {"block": 5000}

    def test_ensure_is_noop_when_anchor_already_set(self, tmp_path):
        v = _make_validator_with_anchor(tmp_path, anchor_file_contents={"block": 42}, metagraph_block=999)
        v._ensure_ramp_up_anchor()
        assert v._ramp_up_start_block == 42  # unchanged

    def test_ensure_skips_when_metagraph_not_synced(self, tmp_path):
        """Metagraph.block is 0 before first sync — must not capture 0 as
        the anchor or we'd end up with a bogus reference point."""
        v = _make_validator_with_anchor(tmp_path, metagraph_block=0)
        v._ensure_ramp_up_anchor()
        assert v._ramp_up_start_block is None


class TestInRampUpBlockBased:
    """in_ramp_up uses (current_block - anchor_block) < RAMP_UP_TEMPOS *
    TEMPO_BLOCKS. A restart reloads the persisted anchor so the window
    is preserved."""

    def _setup(self, tmp_path, anchor_block: int, current_block: int):
        v = _make_validator_with_anchor(
            tmp_path,
            anchor_file_contents={"block": anchor_block},
            metagraph_block=current_block,
        )
        # Bypass the TESTLAB_MODE short-circuit.
        v.config = MagicMock()
        v.config.TESTLAB_MODE = False
        return v

    def test_within_window_returns_true(self, tmp_path):
        anchor = 100_000
        # 1 tempo into ramp-up, well within RAMP_UP_TEMPOS.
        current = anchor + TEMPO_BLOCKS
        v = self._setup(tmp_path, anchor_block=anchor, current_block=current)
        assert v.in_ramp_up is True

    def test_past_window_returns_false(self, tmp_path):
        anchor = 100_000
        current = anchor + RAMP_UP_TEMPOS * TEMPO_BLOCKS + 1
        v = self._setup(tmp_path, anchor_block=anchor, current_block=current)
        assert v.in_ramp_up is False

    def test_exact_boundary_exits_ramp_up(self, tmp_path):
        anchor = 100_000
        current = anchor + RAMP_UP_TEMPOS * TEMPO_BLOCKS
        v = self._setup(tmp_path, anchor_block=anchor, current_block=current)
        assert v.in_ramp_up is False

    def test_simulated_restart_preserves_window(self, tmp_path):
        """The plain-language H-9 invariant: bouncing the validator does
        not rewind its ramp-up progress."""
        anchor = 100_000
        # "Before the restart", validator has run for 1 tempo.
        before = self._setup(
            tmp_path, anchor_block=anchor, current_block=anchor + TEMPO_BLOCKS
        )
        assert before.in_ramp_up is True
        # Simulate a restart: new Validator instance reads the persisted anchor.
        # Time jumps forward by 1 more tempo while the process was down.
        after = _make_validator_with_anchor(
            tmp_path, metagraph_block=anchor + 2 * TEMPO_BLOCKS
        )
        after.config = MagicMock()
        after.config.TESTLAB_MODE = False
        after.start_time = 0.0  # process just booted
        # Without H-9, this would be "in ramp-up for another 3 tempos".
        # With H-9, we're 2 tempos in — still in ramp-up but with proper
        # progress, and will exit after one more tempo.
        assert after.in_ramp_up is True
        # Fast-forward: one more tempo and we exit.
        after.metagraph.block = anchor + RAMP_UP_TEMPOS * TEMPO_BLOCKS + 1
        assert after.in_ramp_up is False

    def test_testlab_mode_short_circuits_to_false(self, tmp_path):
        v = self._setup(tmp_path, anchor_block=100_000, current_block=100_001)
        v.config.TESTLAB_MODE = True
        assert v.in_ramp_up is False

    def test_unsynced_metagraph_stays_in_ramp_up(self, tmp_path):
        """Before the first sync, metagraph.block is 0. We should assume
        ramp-up until we have a real block reading, to avoid releasing
        weight influence based on stale/zero state."""
        v = self._setup(tmp_path, anchor_block=100_000, current_block=0)
        assert v.in_ramp_up is True

    def test_fallback_to_elapsed_when_no_anchor_yet(self, tmp_path):
        """If no anchor has been captured (e.g. test harness without
        metagraph), the property falls back to wall-clock elapsed —
        strictly no-more-permissive on a running chain."""
        import time

        v = Validator.__new__(Validator)
        v.config = MagicMock()
        v.config.TESTLAB_MODE = False
        v.metagraph = MagicMock()
        v.metagraph.block = 0
        v._ramp_up_start_block = None
        v.start_time = time.monotonic()
        assert v.in_ramp_up is True


class TestRampUpAnchorConfigPath:
    """The config path knob is resolvable and lands under DATA_DIR."""

    def test_config_exposes_ramp_up_anchor_path(self):
        from aurelius.config import Config

        assert hasattr(Config, "RAMP_UP_ANCHOR_PATH")
        assert Config.RAMP_UP_ANCHOR_PATH.endswith("ramp_up_anchor.json")

    def test_ensure_data_dirs_creates_anchor_parent(self, tmp_path, monkeypatch):
        from aurelius.config import Config

        new_path = str(tmp_path / "nested" / "deeper" / "ramp_up_anchor.json")
        monkeypatch.setattr(Config, "RAMP_UP_ANCHOR_PATH", new_path)
        Config.ensure_data_dirs()
        from pathlib import Path as _P

        assert _P(new_path).parent.exists()
