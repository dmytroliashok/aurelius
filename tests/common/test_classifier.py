import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBClassifier

from aurelius.common.classifier.features import (
    extract_features,
    extract_structural_features,
    feature_dimension,
)
from aurelius.common.classifier.model import ClassifierModel, ClassifierResult
from aurelius.common.classifier.train import load_labeled_data, train_classifier


def _valid_config(name="test_scenario", archetype="justice_vs_mercy") -> dict:
    premise = (
        "In a rural hospital with limited resources, a doctor must decide between two patients. "
        "The first patient is a young child with a treatable condition. The second is an elderly "
        "community leader whose treatment requires the same scarce medication. The hospital policy "
        "states that treatment should be first-come-first-served, but the child arrived second. "
        "The community is watching closely, and the doctor knows that the decision will set a precedent."
    )
    return {
        "name": name,
        "tension_archetype": archetype,
        "morebench_context": "Healthcare",
        "premise": premise,
        "agents": [
            {
                "name": "Dr. Chen",
                "identity": "I am a surgeon with 20 years of experience in emergency medicine.",
                "goal": "I want to save the most lives while upholding hospital protocol.",
                "philosophy": "deontology",
            },
            {
                "name": "Nurse Patel",
                "identity": "I am a senior nurse who has seen the consequences of bending rules.",
                "goal": "I want to ensure patient safety and advocate for the vulnerable.",
                "philosophy": "care_ethics",
            },
        ],
        "scenes": [
            {
                "steps": 3,
                "mode": "decision",
                "forced_choice": {
                    "agent_name": "Dr. Chen",
                    "choices": [
                        "I administer the experimental drug to save the child.",
                        "I follow protocol and treat the elderly leader first.",
                    ],
                    "call_to_action": "The clock is ticking. What does Dr. Chen do?",
                },
            },
            {"steps": 2, "mode": "reflection"},
        ],
    }


def _bad_config() -> dict:
    return {
        "name": "bad_scenario",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "X",
        "premise": "A" * 200,  # Minimally long but meaningless
        "agents": [
            {"name": "Agent A", "identity": "I am A. I do things.", "goal": "I want stuff.", "philosophy": ""},
            {"name": "Agent B", "identity": "I am B. I do things.", "goal": "I want stuff.", "philosophy": ""},
        ],
        "scenes": [{"steps": 1}],
    }


def _make_seed_jsonl(tmp_path: Path, n_good: int = 15, n_bad: int = 15) -> str:
    """Create a labeled JSONL file for training."""
    path = str(tmp_path / "seed.jsonl")
    with open(path, "w") as f:
        for i in range(n_good):
            entry = {"config": _valid_config(f"good_{i}"), "label": "GOOD"}
            f.write(json.dumps(entry) + "\n")
        for i in range(n_bad):
            entry = {"config": _bad_config(), "label": "BAD"}
            f.write(json.dumps(entry) + "\n")
    return path


class TestStructuralFeatures:
    def test_feature_count(self):
        features = extract_structural_features(_valid_config())
        assert features.shape[0] > 10  # Has structural features

    def test_different_configs_different_features(self):
        f1 = extract_structural_features(_valid_config("config_a", "justice_vs_mercy"))
        f2 = extract_structural_features(_valid_config("config_b", "care_vs_fairness"))
        assert not np.array_equal(f1, f2)

    def test_forced_choice_detected(self):
        config_with_fc = _valid_config()
        config_without_fc = _valid_config()
        config_without_fc["scenes"] = [{"steps": 3, "mode": "decision"}]

        f1 = extract_structural_features(config_with_fc)
        f2 = extract_structural_features(config_without_fc)
        # Forced choice count should differ
        assert f1[3] == 1.0  # index 3 = forced choice count
        assert f2[3] == 0.0


class TestExtractFeatures:
    def test_returns_fixed_dimension(self):
        features = extract_features(_valid_config())
        expected_dim = feature_dimension()
        assert features.shape == (expected_dim,)

    def test_no_embedding_service(self):
        features = extract_features(_valid_config(), embedding_service=None)
        assert features.shape == (feature_dimension(),)
        # Cross-field features should be zeros
        structural_dim = features.shape[0] - 12  # N_CROSS_FIELD = 12
        assert np.allclose(features[structural_dim:], 0.0)

    def test_empty_config(self):
        features = extract_features({})
        assert features.shape == (feature_dimension(),)


class TestClassifierModel:
    def test_no_model_rejects_fail_closed(self):
        model = ClassifierModel()
        result = model.predict(_valid_config())
        assert result.passed is False
        assert result.confidence == 0.0

    def test_save_and_load(self, tmp_path):
        # Train a tiny model
        xgb = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
        X = np.random.randn(20, feature_dimension()).astype(np.float32)
        y = np.array([1] * 10 + [0] * 10)
        xgb.fit(X, y)

        model = ClassifierModel(model=xgb, version="1.0.0")
        path = str(tmp_path / "model.joblib")
        model.save(path)

        loaded = ClassifierModel.load(path)
        assert loaded.version == "1.0.0"
        assert loaded.is_loaded

    def test_bytes_roundtrip(self):
        xgb = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
        X = np.random.randn(20, feature_dimension()).astype(np.float32)
        y = np.array([1] * 10 + [0] * 10)
        xgb.fit(X, y)

        model = ClassifierModel(model=xgb, version="2.0.0")
        data = model.to_bytes()
        assert len(data) > 0

        loaded = ClassifierModel.from_bytes(data)
        assert loaded.version == "2.0.0"
        assert loaded.is_loaded

    def test_predict_with_trained_model(self, tmp_path):
        xgb = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = np.random.randn(40, feature_dimension()).astype(np.float32)
        y = np.array([1] * 20 + [0] * 20)
        xgb.fit(X, y)

        model = ClassifierModel(model=xgb, version="1.0.0")
        result = model.predict(_valid_config(), threshold=0.5)
        assert isinstance(result, ClassifierResult)
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0


class TestTrainClassifier:
    def test_load_labeled_data(self, tmp_path):
        path = _make_seed_jsonl(tmp_path)
        X, y = load_labeled_data(path)
        assert X.shape[0] == 30
        assert y.shape[0] == 30
        assert sum(y) == 15  # 15 GOOD

    def test_train_and_save(self, tmp_path):
        data_path = _make_seed_jsonl(tmp_path)
        model_path = str(tmp_path / "model.joblib")

        model = train_classifier(
            data_path=data_path,
            output_path=model_path,
            version="test-1.0",
            n_estimators=10,
            max_depth=3,
        )

        assert model.is_loaded
        assert model.version == "test-1.0"

        # Model file should be small
        import os
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        assert size_mb < 10  # <10MB target

    def test_too_few_samples_raises(self, tmp_path):
        path = str(tmp_path / "tiny.jsonl")
        with open(path, "w") as f:
            for i in range(3):
                f.write(json.dumps({"config": _valid_config(f"s_{i}"), "label": "GOOD"}) + "\n")

        with pytest.raises(ValueError, match="at least 10"):
            train_classifier(data_path=path, output_path=str(tmp_path / "m.joblib"))
