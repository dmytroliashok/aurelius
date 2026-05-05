import json

import numpy as np
import pytest

from aurelius.benchmark.labeling import LABEL_HIGH, LABEL_LOW, WEIGHT_HIGH, WEIGHT_LOW, LabelingResult
from aurelius.benchmark.retrain import _bump_version, retrain_classifier


def _make_seed_data(tmp_path, n=15):
    """Create a seed dataset."""
    premise = "A" * 250
    path = str(tmp_path / "seed.jsonl")
    with open(path, "w") as f:
        for i in range(n):
            config = {
                "name": f"seed_config_{i:03d}",
                "tension_archetype": "justice_vs_mercy",
                "morebench_context": "Healthcare",
                "premise": premise,
                "agents": [
                    {"name": "Agent A", "identity": "I am agent A with strong opinions.", "goal": "I want to do right."},
                    {"name": "Agent B", "identity": "I am agent B with different views.", "goal": "I want justice."},
                ],
                "scenes": [{"steps": 3}],
            }
            label = "GOOD" if i % 2 == 0 else "BAD"
            f.write(json.dumps({"config": config, "label": label}) + "\n")
    return path


class TestBumpVersion:
    def test_patch_bump(self):
        assert _bump_version("1.0.0") == "1.0.1"
        assert _bump_version("1.0.9") == "1.0.10"
        assert _bump_version("2.3.5") == "2.3.6"

    def test_invalid_version(self):
        assert _bump_version("invalid") == "1.0.0"


class TestRetrainClassifier:
    def test_retrain_from_seed_only(self, tmp_path):
        seed_path = _make_seed_data(tmp_path, n=15)
        model_path = str(tmp_path / "model.joblib")

        labels = LabelingResult(
            labels={},
            weights={},
            counts={LABEL_HIGH: 0, LABEL_LOW: 0, "excluded": 0},
        )

        model = retrain_classifier(
            new_labels=labels,
            new_configs=[],
            submission_ids=[],
            seed_data_path=seed_path,
            output_path=model_path,
            current_version="1.0.0",
            n_estimators=10,
            max_depth=3,
        )

        assert model.is_loaded
        assert model.version == "1.0.1"

    def test_retrain_with_new_data(self, tmp_path):
        seed_path = _make_seed_data(tmp_path, n=10)
        model_path = str(tmp_path / "model.joblib")

        new_configs = []
        submission_ids = []
        for i in range(10):
            new_configs.append({
                "name": f"new_config_{i:03d}",
                "tension_archetype": "care_vs_fairness",
                "morebench_context": "Education",
                "premise": "B" * 250,
                "agents": [
                    {"name": "Teacher", "identity": "I am a dedicated teacher.", "goal": "I want to help students."},
                    {"name": "Principal", "identity": "I am the school principal.", "goal": "I want order."},
                ],
                "scenes": [{"steps": 2}],
            })
            submission_ids.append(1000 + i)

        labels = LabelingResult(
            labels={sid: LABEL_LOW for sid in submission_ids},
            weights={sid: WEIGHT_LOW for sid in submission_ids},
            counts={LABEL_HIGH: 0, LABEL_LOW: 10, "excluded": 0},
        )

        model = retrain_classifier(
            new_labels=labels,
            new_configs=new_configs,
            submission_ids=submission_ids,
            seed_data_path=seed_path,
            output_path=model_path,
            current_version="1.0.5",
            n_estimators=10,
            max_depth=3,
        )

        assert model.is_loaded
        assert model.version == "1.0.6"

    def test_insufficient_data_raises(self, tmp_path):
        seed_path = str(tmp_path / "empty_seed.jsonl")
        with open(seed_path, "w") as f:
            pass  # Empty file

        labels = LabelingResult(labels={}, weights={}, counts={})

        with pytest.raises(ValueError, match="Insufficient"):
            retrain_classifier(
                new_labels=labels,
                new_configs=[],
                submission_ids=[],
                seed_data_path=seed_path,
                output_path=str(tmp_path / "model.joblib"),
            )
