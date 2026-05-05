import pytest

from aurelius.benchmark.evaluate import BenchmarkResult
from aurelius.benchmark.influence import InfluenceScores, compute_influence_scores


class TestComputeInfluenceScores:
    def test_small_batch_returns_uniform(self):
        result = BenchmarkResult(overall_score=0.8, delta=0.05)
        submission_ids = list(range(10))  # Below 30

        scores = compute_influence_scores(
            model_path="/fake",
            dataset_path="/fake",
            benchmark_result=result,
            submission_ids=submission_ids,
        )

        assert scores.method == "uniform"
        assert len(scores.scores) == 10
        # All scores should be equal
        values = list(scores.scores.values())
        assert all(v == values[0] for v in values)

    def test_uniform_scores_sum_to_delta(self):
        result = BenchmarkResult(overall_score=0.8, delta=0.10)
        submission_ids = list(range(10))

        scores = compute_influence_scores(
            model_path="/fake",
            dataset_path="/fake",
            benchmark_result=result,
            submission_ids=submission_ids,
        )

        total = sum(scores.scores.values())
        assert abs(total - 0.10) < 1e-6

    def test_zero_delta(self):
        result = BenchmarkResult(overall_score=0.5, delta=0.0)
        submission_ids = list(range(10))

        scores = compute_influence_scores(
            model_path="/fake",
            dataset_path="/fake",
            benchmark_result=result,
            submission_ids=submission_ids,
        )

        assert all(v == 0.0 for v in scores.scores.values())

    def test_batch_delta_preserved(self):
        result = BenchmarkResult(overall_score=0.8, delta=0.05)
        scores = compute_influence_scores("/fake", "/fake", result, list(range(5)))
        assert scores.batch_delta == 0.05


class TestInfluenceScores:
    def test_dataclass(self):
        scores = InfluenceScores(scores={1: 0.1, 2: 0.2}, batch_delta=0.3, method="fisher")
        assert scores.scores[1] == 0.1
        assert scores.batch_delta == 0.3
        assert scores.method == "fisher"
