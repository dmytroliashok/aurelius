"""Cycle-summary log line: one structured INFO line per main-loop
iteration so operators don't have to grep across many scattered per-
stage logs to understand a cycle's outcome.

The render function is a pure function of a dict; the stats builder is
a method on Validator that reads from already-computed instance state.
Neither needs the main loop to test.
"""

from unittest.mock import MagicMock

import pytest

from aurelius.validator.pipeline import PipelineResult, StageResult
from aurelius.validator.validator import Validator, _render_cycle_summary


def _failed(stage: str) -> PipelineResult:
    return PipelineResult(
        weight=0.0,
        stages=[StageResult(passed=False, reason="nope", stage=stage)],
    )


class TestRenderCycleSummary:
    def test_minimal_stats_yields_tagged_line(self):
        line = _render_cycle_summary({"miners_queried": 0})
        assert line.startswith("cycle_summary")
        assert "miners_queried=0" in line

    def test_fields_ordered_for_scanability(self):
        line = _render_cycle_summary(
            {
                "miners_queried": 3,
                "miners_passed": 1,
                "cycle_duration_s": 4.12,
                "weights_outcome": "rate_limit",
                "in_ramp_up": False,
                "degraded_mode": False,
            }
        )
        # The leading identifier comes first, then the rest in a fixed
        # order the render function enforces.
        expected_order = [
            "cycle_summary",
            "miners_queried=3",
            "miners_passed=1",
            "cycle_duration_s=4.12",
            "weights_outcome=rate_limit",
            "in_ramp_up=False",
            "degraded_mode=False",
        ]
        for prev, nxt in zip(expected_order, expected_order[1:]):
            assert line.index(prev) < line.index(nxt), line

    def test_stage_failures_rendered_sorted(self):
        line = _render_cycle_summary(
            {
                "miners_queried": 3,
                "stage_failures": {"work_token_check": 2, "classifier_gate": 1},
            }
        )
        assert "failures=classifier_gate:1,work_token_check:2" in line

    def test_empty_failures_omitted(self):
        line = _render_cycle_summary({"miners_queried": 3, "stage_failures": {}})
        assert "failures=" not in line

    def test_unknown_keys_ignored(self):
        """Future additions to the stats dict shouldn't leak into the line
        uncontrolled. The renderer only emits known keys."""
        line = _render_cycle_summary(
            {"miners_queried": 1, "secret_new_thing": "surprise!"}
        )
        assert "secret_new_thing" not in line
        assert "surprise" not in line

    def test_missing_keys_omitted_not_defaulted(self):
        """If the caller didn't report miners_passed (e.g. cycle failed
        early), we don't fabricate a zero."""
        line = _render_cycle_summary({"miners_queried": 0})
        assert "miners_passed" not in line


class TestBuildCycleStats:
    """The method on Validator that collects numbers the renderer will
    format. Exercised directly against a bare Validator instance with
    canned self.results and _last_weights_outcome."""

    def _make(self, api_available: bool = True):
        v = Validator.__new__(Validator)
        v.results = {}
        v._last_weights_outcome = "skipped"
        v.remote_config = MagicMock()
        v.remote_config.api_available = api_available
        v.config = MagicMock()
        v.config.TESTLAB_MODE = False
        v.metagraph = MagicMock()
        v.metagraph.block = 100_000
        v._ramp_up_start_block = 100_000  # just started
        v.start_time = 0.0
        return v

    def test_passing_and_failing_mix(self):
        import time

        v = self._make()
        v.results = {
            "hk1": PipelineResult(weight=0.8, stages=[]),  # passed (weight > 0)
            "hk2": _failed("classifier_gate"),
            "hk3": _failed("classifier_gate"),
        }
        responses = [MagicMock(), MagicMock(), MagicMock()]
        stats = v._build_cycle_stats(responses=responses, cycle_start=time.monotonic() - 1.5)
        assert stats["miners_queried"] == 3
        assert stats["miners_passed"] == 1
        assert stats["stage_failures"] == {"classifier_gate": 2}
        # Duration rounded to 2 decimals and > 0.
        assert stats["cycle_duration_s"] >= 1.5
        assert stats["weights_outcome"] == "skipped"

    def test_empty_cycle(self):
        import time

        v = self._make()
        stats = v._build_cycle_stats(responses=[], cycle_start=time.monotonic())
        assert stats["miners_queried"] == 0
        assert stats["miners_passed"] == 0
        assert stats["stage_failures"] == {}

    def test_degraded_mode_reflected(self):
        import time

        v = self._make(api_available=False)
        stats = v._build_cycle_stats(responses=[MagicMock()], cycle_start=time.monotonic())
        assert stats["degraded_mode"] is True

    def test_weights_outcome_carries_through(self):
        import time

        v = self._make()
        v._last_weights_outcome = "success"
        stats = v._build_cycle_stats(responses=[], cycle_start=time.monotonic())
        assert stats["weights_outcome"] == "success"


class TestRenderUsesBuilderOutput:
    """Sanity: a fresh stats dict from _build_cycle_stats renders without
    throwing and mentions the obvious fields."""

    def test_end_to_end_render(self):
        import time

        v = Validator.__new__(Validator)
        v.results = {}
        v._last_weights_outcome = "rate_limit"
        v.remote_config = MagicMock()
        v.remote_config.api_available = True
        v.config = MagicMock()
        v.config.TESTLAB_MODE = False
        v.metagraph = MagicMock()
        v.metagraph.block = 100_000
        v._ramp_up_start_block = 100_000
        v.start_time = 0.0

        stats = v._build_cycle_stats(responses=[], cycle_start=time.monotonic())
        line = _render_cycle_summary(stats)
        assert "cycle_summary" in line
        assert "weights_outcome=rate_limit" in line
