import json

import pytest

from aurelius.benchmark.evaluate import BenchmarkResult, DimensionScore, load_morebench_scenarios


class TestBenchmarkResult:
    def test_basic_result(self):
        result = BenchmarkResult(
            overall_score=0.75,
            dimensions=[
                DimensionScore(name="procedural", score=0.8, criteria_met=80, criteria_total=100),
                DimensionScore(name="pluralistic", score=0.7, criteria_met=70, criteria_total=100),
            ],
            scenarios_evaluated=500,
            model_path="/path/to/model",
        )
        assert result.overall_score == 0.75
        assert len(result.dimensions) == 2

    def test_delta_computation(self):
        result = BenchmarkResult(overall_score=0.8, delta=0.05)
        assert result.delta == 0.05


class TestLoadMorebench:
    def test_missing_file_returns_empty(self):
        scenarios = load_morebench_scenarios("/nonexistent/path.json")
        assert scenarios == []

    def test_valid_file(self, tmp_path):
        data = [
            {"scenario": "A moral dilemma", "rubric": ["criterion 1"], "dimension": "procedural"},
            {"scenario": "Another dilemma", "rubric": ["criterion 2"], "dimension": "pluralistic"},
        ]
        path = str(tmp_path / "morebench.json")
        with open(path, "w") as f:
            json.dump(data, f)

        scenarios = load_morebench_scenarios(path)
        assert len(scenarios) == 2
