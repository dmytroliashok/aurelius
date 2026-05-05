import json

import pytest

from aurelius.benchmark.influence import InfluenceScores
from aurelius.benchmark.labeling import (
    LABEL_EXCLUDED,
    LABEL_HIGH,
    LABEL_LOW,
    WEIGHT_EXCLUDED,
    WEIGHT_HIGH,
    WEIGHT_LOW,
    LabelingResult,
    assign_confidence_labels,
    merge_with_seed_data,
)


class TestAssignConfidenceLabels:
    def test_batch_too_small_all_low(self):
        influence = InfluenceScores(
            scores={1: 0.01, 2: 0.02},
            batch_delta=0.05,
            method="fisher",
        )
        result = assign_confidence_labels(influence, min_batch_size=30)
        assert all(l == LABEL_LOW for l in result.labels.values())
        assert result.counts[LABEL_LOW] == 2

    def test_uniform_method_all_low(self):
        scores = {i: 0.001 for i in range(50)}
        influence = InfluenceScores(scores=scores, batch_delta=0.05, method="uniform")
        result = assign_confidence_labels(influence, min_batch_size=30)
        assert all(l == LABEL_LOW for l in result.labels.values())

    def test_positive_agreement_is_high(self):
        scores = {i: 0.01 for i in range(50)}  # All positive
        influence = InfluenceScores(scores=scores, batch_delta=0.05, method="fisher")
        result = assign_confidence_labels(influence, min_batch_size=30)
        assert all(l == LABEL_HIGH for l in result.labels.values())
        assert result.counts[LABEL_HIGH] == 50

    def test_negative_agreement_is_high(self):
        scores = {i: -0.01 for i in range(50)}  # All negative
        influence = InfluenceScores(scores=scores, batch_delta=-0.05, method="fisher")
        result = assign_confidence_labels(influence, min_batch_size=30)
        assert all(l == LABEL_HIGH for l in result.labels.values())

    def test_contradiction_is_excluded(self):
        # Batch positive but influence negative
        scores = {i: -0.01 for i in range(50)}
        influence = InfluenceScores(scores=scores, batch_delta=0.05, method="fisher")
        result = assign_confidence_labels(influence, min_batch_size=30)
        assert all(l == LABEL_EXCLUDED for l in result.labels.values())
        assert result.counts[LABEL_EXCLUDED] == 50

    def test_near_zero_influence_is_low(self):
        scores = {i: 0.0 for i in range(50)}
        influence = InfluenceScores(scores=scores, batch_delta=0.05, method="fisher")
        result = assign_confidence_labels(influence, min_batch_size=30)
        assert all(l == LABEL_LOW for l in result.labels.values())

    def test_mixed_labels(self):
        scores = {}
        for i in range(20):
            scores[i] = 0.01  # Agrees with positive delta → HIGH
        for i in range(20, 35):
            scores[i] = -0.01  # Contradicts positive delta → EXCLUDED
        for i in range(35, 50):
            scores[i] = 0.0  # Ambiguous → LOW

        influence = InfluenceScores(scores=scores, batch_delta=0.05, method="fisher")
        result = assign_confidence_labels(influence, min_batch_size=30)
        assert result.counts[LABEL_HIGH] == 20
        assert result.counts[LABEL_EXCLUDED] == 15
        assert result.counts[LABEL_LOW] == 15

    def test_weights_match_labels(self):
        scores = {i: 0.01 for i in range(50)}
        influence = InfluenceScores(scores=scores, batch_delta=0.05, method="fisher")
        result = assign_confidence_labels(influence, min_batch_size=30)
        for sid, label in result.labels.items():
            if label == LABEL_HIGH:
                assert result.weights[sid] == WEIGHT_HIGH
            elif label == LABEL_LOW:
                assert result.weights[sid] == WEIGHT_LOW
            elif label == LABEL_EXCLUDED:
                assert result.weights[sid] == WEIGHT_EXCLUDED


class TestMergeWithSeedData:
    def test_seed_data_always_included(self, tmp_path):
        seed_path = str(tmp_path / "seed.jsonl")
        with open(seed_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"config": {"name": f"seed_{i}"}, "label": "GOOD"}) + "\n")

        labels = LabelingResult(
            labels={},
            weights={},
            counts={LABEL_HIGH: 0, LABEL_LOW: 0, LABEL_EXCLUDED: 0},
        )
        configs, weights = merge_with_seed_data(labels, seed_path, [], [])
        assert len(configs) == 5
        assert all(w == WEIGHT_HIGH for w in weights)

    def test_excluded_data_dropped(self, tmp_path):
        seed_path = str(tmp_path / "seed.jsonl")
        with open(seed_path, "w") as f:
            f.write(json.dumps({"config": {"name": "seed_0"}, "label": "GOOD"}) + "\n")

        labels = LabelingResult(
            labels={100: LABEL_EXCLUDED, 101: LABEL_HIGH},
            weights={100: WEIGHT_EXCLUDED, 101: WEIGHT_HIGH},
            counts={LABEL_HIGH: 1, LABEL_LOW: 0, LABEL_EXCLUDED: 1},
        )
        new_data = [{"name": "excluded_config"}, {"name": "good_config"}]
        submission_ids = [100, 101]

        configs, weights = merge_with_seed_data(labels, seed_path, new_data, submission_ids)
        # 1 seed + 1 non-excluded new = 2
        assert len(configs) == 2

    def test_missing_seed_still_works(self, tmp_path):
        labels = LabelingResult(
            labels={1: LABEL_LOW},
            weights={1: WEIGHT_LOW},
            counts={LABEL_HIGH: 0, LABEL_LOW: 1, LABEL_EXCLUDED: 0},
        )
        configs, weights = merge_with_seed_data(
            labels, str(tmp_path / "nonexistent.jsonl"), [{"name": "new"}], [1]
        )
        assert len(configs) == 1
