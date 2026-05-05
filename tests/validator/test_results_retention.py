"""Result-retention TTL: a miner's pass result must survive in
self.results across burn-only cycles long enough to reach the next
chain tempo boundary, otherwise the on-chain submission at that
boundary is burn-only and incentive stays at zero.

The semantics under test:
- Pass results are stamped with current_block on write.
- _prune_stale_results drops entries older than RESULT_RETENTION_BLOCKS.
- _record_result never lets a fresh fail (weight=0) overwrite a still-
  fresh prior pass (weight>0).
- _build_cycle_stats with a current_block argument counts only entries
  recorded this cycle so retained passes don't inflate miners_passed.
"""

from aurelius.common.constants import RESULT_RETENTION_BLOCKS
from aurelius.validator.pipeline import PipelineResult, StageResult
from aurelius.validator.validator import Validator


def _bare_validator() -> Validator:
    v = Validator.__new__(Validator)
    v.results = {}
    return v


def _pass_result(weight: float = 0.5) -> PipelineResult:
    return PipelineResult(weight=weight, stages=[StageResult(passed=True, reason="ok", stage="simulate")])


def _fail_result(stage: str = "classifier_gate") -> PipelineResult:
    return PipelineResult(weight=0.0, stages=[StageResult(passed=False, reason="below threshold", stage=stage)])


class TestRecordResult:
    def test_pass_stamps_block_and_stores(self):
        v = _bare_validator()
        v._record_result("hk_a", _pass_result(0.7), current_block=1000)
        assert "hk_a" in v.results
        assert v.results["hk_a"].weight == 0.7
        assert v.results["hk_a"].recorded_at_block == 1000

    def test_fail_stores_when_no_existing_entry(self):
        v = _bare_validator()
        v._record_result("hk_a", _fail_result(), current_block=1000)
        assert "hk_a" in v.results
        assert v.results["hk_a"].weight == 0.0
        assert v.results["hk_a"].recorded_at_block == 1000

    def test_fail_does_not_displace_prior_pass(self):
        v = _bare_validator()
        v._record_result("hk_a", _pass_result(0.7), current_block=1000)
        v._record_result("hk_a", _fail_result(), current_block=1050)
        # The pass at block 1000 must still be there with its original weight.
        assert v.results["hk_a"].weight == 0.7
        assert v.results["hk_a"].recorded_at_block == 1000

    def test_fresh_pass_overwrites_prior_pass(self):
        v = _bare_validator()
        v._record_result("hk_a", _pass_result(0.5), current_block=1000)
        v._record_result("hk_a", _pass_result(0.9), current_block=1050)
        # Latest pass wins — both weight and block update.
        assert v.results["hk_a"].weight == 0.9
        assert v.results["hk_a"].recorded_at_block == 1050

    def test_fail_refreshes_prior_fail(self):
        # A miner that fails every cycle (e.g. no work-token balance)
        # must keep refreshing recorded_at_block — otherwise the
        # per-cycle summary log filters them out as stale and operators
        # see misleading "failures={}" lines on cycles where the same
        # 6+ miners are still failing.
        v = _bare_validator()
        v._record_result("hk_a", _fail_result("rate_limit_check"), current_block=1000)
        v._record_result("hk_a", _fail_result("classifier_gate"), current_block=1050)
        # Latest stage and timestamp win.
        assert v.results["hk_a"].recorded_at_block == 1050
        assert v.results["hk_a"].failed_stage == "classifier_gate"


class TestPruneStaleResults:
    def test_pass_survives_within_ttl(self):
        v = _bare_validator()
        v._record_result("hk_a", _pass_result(0.7), current_block=1000)
        # 50 blocks later — still well within TTL.
        v._prune_stale_results(current_block=1050)
        assert "hk_a" in v.results
        assert v.results["hk_a"].weight == 0.7

    def test_pass_survives_at_exact_ttl_boundary(self):
        v = _bare_validator()
        v._record_result("hk_a", _pass_result(0.7), current_block=1000)
        # Exactly at the boundary — still retained (cutoff is strict <).
        v._prune_stale_results(current_block=1000 + RESULT_RETENTION_BLOCKS)
        assert "hk_a" in v.results

    def test_pass_pruned_past_ttl(self):
        v = _bare_validator()
        v._record_result("hk_a", _pass_result(0.7), current_block=1000)
        v._prune_stale_results(current_block=1000 + RESULT_RETENTION_BLOCKS + 1)
        assert "hk_a" not in v.results

    def test_legacy_unstamped_entry_pruned(self):
        v = _bare_validator()
        # Synthesize a result without recorded_at_block (e.g. from before
        # this change deployed). Should be evicted on first prune.
        v.results["hk_legacy"] = _pass_result(0.5)  # recorded_at_block stays None
        v._prune_stale_results(current_block=1000)
        assert "hk_legacy" not in v.results

    def test_mixed_old_and_new_entries(self):
        v = _bare_validator()
        v._record_result("hk_old", _pass_result(0.5), current_block=500)
        v._record_result("hk_new", _pass_result(0.8), current_block=1000)
        # Prune at a block where hk_old is past TTL but hk_new is not.
        v._prune_stale_results(current_block=500 + RESULT_RETENTION_BLOCKS + 1)
        assert "hk_old" not in v.results
        assert "hk_new" in v.results


class TestBuildCycleStatsWithCurrentBlock:
    """Stale entries from prior cycles must not be counted toward this
    cycle's miners_passed / stage_failures — that would mislead the
    operator-facing cycle_summary log."""

    def _make(self) -> Validator:
        from unittest.mock import MagicMock

        v = Validator.__new__(Validator)
        v.results = {}
        v._last_weights_outcome = "skipped"
        v.remote_config = MagicMock()
        v.remote_config.api_available = True
        v.config = MagicMock()
        v.config.TESTLAB_MODE = False
        v.metagraph = MagicMock()
        v.metagraph.block = 100_000
        v._ramp_up_start_block = 100_000
        v.start_time = 0.0
        return v

    def test_prior_cycle_pass_not_counted_this_cycle(self):
        import time

        v = self._make()
        v._record_result("hk_old_pass", _pass_result(0.7), current_block=500)
        # This cycle: one fresh fail.
        v._record_result("hk_now_fail", _fail_result("classifier_gate"), current_block=1000)
        stats = v._build_cycle_stats(
            responses=[1],  # placeholder, just need len()
            cycle_start=time.monotonic(),
            current_block=1000,
        )
        # Only the this-cycle entries count.
        assert stats["miners_passed"] == 0
        assert stats["stage_failures"] == {"classifier_gate": 1}

    def test_repeat_failures_counted_every_cycle(self):
        """Production reality: 6+ miners with no work-token balance fail
        ``work_token_check`` every cycle. cycle_summary must keep
        reporting them as this-cycle failures, not just on the first
        cycle they appeared. Regression guard for the early review of
        this fix where fails-over-fails weren't refreshing
        recorded_at_block."""
        import time

        v = self._make()
        # Cycle T: 2 miners fail at work_token_check.
        v._record_result("hk1", _fail_result("work_token_check"), current_block=1000)
        v._record_result("hk2", _fail_result("work_token_check"), current_block=1000)
        stats_t = v._build_cycle_stats(responses=[1, 2], cycle_start=time.monotonic(), current_block=1000)
        assert stats_t["stage_failures"] == {"work_token_check": 2}

        # Cycle T+25 (~5 min later, well within TTL): same 2 miners
        # fail again at the same stage. Operator must see the failures
        # counted *this cycle*, not silently ignored as stale.
        v._record_result("hk1", _fail_result("work_token_check"), current_block=1025)
        v._record_result("hk2", _fail_result("work_token_check"), current_block=1025)
        stats_t1 = v._build_cycle_stats(responses=[1, 2], cycle_start=time.monotonic(), current_block=1025)
        assert stats_t1["stage_failures"] == {"work_token_check": 2}

    def test_legacy_call_without_current_block_still_works(self):
        """Existing tests in test_cycle_summary.py construct PipelineResults
        without recorded_at_block and call _build_cycle_stats without
        current_block. That path must keep working."""
        import time

        v = self._make()
        v.results = {
            "hk1": _pass_result(0.5),
            "hk2": _fail_result("rate_limit_check"),
        }
        stats = v._build_cycle_stats(responses=[1, 2], cycle_start=time.monotonic())
        assert stats["miners_passed"] == 1
        assert stats["stage_failures"] == {"rate_limit_check": 1}
