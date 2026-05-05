import json
import tempfile
from pathlib import Path

import pytest

from aurelius.miner.config_store import ConfigStore


def _valid_config(name: str = "test_scenario_one") -> dict:
    premise = (
        "In a rural hospital with limited resources, a doctor must decide between two patients. "
        "The first patient is a young child with a treatable condition. The second is an elderly "
        "community leader whose treatment requires the same scarce medication. The hospital policy "
        "states that treatment should be first-come-first-served, but the child arrived second. "
        "The community is watching closely, and the doctor knows that the decision will set a precedent."
    )
    return {
        "name": name,
        "tension_archetype": "justice_vs_mercy",
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
            {"steps": 3, "mode": "decision"},
            {"steps": 2, "mode": "reflection"},
        ],
    }


class TestConfigStore:
    def test_load_valid_configs(self, tmp_path):
        (tmp_path / "config1.json").write_text(json.dumps(_valid_config("scenario_alpha")))
        (tmp_path / "config2.json").write_text(json.dumps(_valid_config("scenario_beta")))

        store = ConfigStore(tmp_path)
        assert store.count == 2

    def test_round_robin(self, tmp_path):
        (tmp_path / "config1.json").write_text(json.dumps(_valid_config("scenario_alpha")))
        (tmp_path / "config2.json").write_text(json.dumps(_valid_config("scenario_beta")))

        store = ConfigStore(tmp_path)
        first = store.next()
        second = store.next()
        third = store.next()
        assert first["name"] == "scenario_alpha"
        assert second["name"] == "scenario_beta"
        assert third["name"] == "scenario_alpha"  # wraps around

    def test_empty_directory(self, tmp_path):
        store = ConfigStore(tmp_path)
        assert store.count == 0
        assert store.next() is None

    def test_nonexistent_directory(self):
        store = ConfigStore("/nonexistent/path")
        assert store.count == 0

    def test_invalid_config_skipped(self, tmp_path):
        (tmp_path / "good.json").write_text(json.dumps(_valid_config("scenario_good")))
        (tmp_path / "bad.json").write_text(json.dumps({"invalid": True}))

        store = ConfigStore(tmp_path)
        assert store.count == 1
        assert store.next()["name"] == "scenario_good"

    def test_malformed_json_skipped(self, tmp_path):
        (tmp_path / "good.json").write_text(json.dumps(_valid_config("scenario_good")))
        (tmp_path / "broken.json").write_text("{not valid json")

        store = ConfigStore(tmp_path)
        assert store.count == 1

    def test_reload(self, tmp_path):
        (tmp_path / "config1.json").write_text(json.dumps(_valid_config("scenario_alpha")))
        store = ConfigStore(tmp_path)
        assert store.count == 1

        (tmp_path / "config2.json").write_text(json.dumps(_valid_config("scenario_beta")))
        store.reload()
        assert store.count == 2
