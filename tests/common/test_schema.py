import copy

import pytest

from aurelius.common.schema import get_schema, validate_scenario_config


def _valid_config() -> dict:
    """Return a minimal valid scenario config dict."""
    premise = (
        "In a rural hospital with limited resources, a doctor must decide between two patients. "
        "The first patient is a young child with a treatable condition. The second is an elderly "
        "community leader whose treatment requires the same scarce medication. The hospital policy "
        "states that treatment should be first-come-first-served, but the child arrived second. "
        "The community is watching closely, and the doctor knows that the decision will set a precedent."
    )
    return {
        "name": "hospital_dilemma_one",
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


class TestGetSchema:
    def test_schema_loads(self):
        schema = get_schema()
        assert schema["title"] == "ScenarioConfig"
        assert "properties" in schema

    def test_schema_cached(self):
        s1 = get_schema()
        s2 = get_schema()
        assert s1 is s2


class TestValidateScenarioConfig:
    def test_valid_config_passes(self):
        result = validate_scenario_config(_valid_config())
        assert result.valid
        assert result.errors == []

    def test_missing_required_field(self):
        config = _valid_config()
        del config["name"]
        result = validate_scenario_config(config)
        assert not result.valid
        assert any("name" in e for e in result.errors)

    def test_invalid_tension_archetype(self):
        config = _valid_config()
        config["tension_archetype"] = "invalid_archetype"
        result = validate_scenario_config(config)
        assert not result.valid

    def test_custom_tension_without_description(self):
        config = _valid_config()
        config["tension_archetype"] = "custom"
        # No tension_description
        result = validate_scenario_config(config)
        assert not result.valid
        assert any("tension_description" in e for e in result.errors)

    def test_custom_tension_with_description(self):
        config = _valid_config()
        config["tension_archetype"] = "custom"
        config["tension_description"] = "A novel tension between technological progress and cultural preservation."
        result = validate_scenario_config(config)
        assert result.valid

    def test_premise_too_short(self):
        config = _valid_config()
        config["premise"] = "Too short."
        result = validate_scenario_config(config)
        assert not result.valid

    def test_invalid_name_format(self):
        config = _valid_config()
        config["name"] = "Invalid Name!"
        result = validate_scenario_config(config)
        assert not result.valid

    def test_too_few_agents(self):
        config = _valid_config()
        config["agents"] = [config["agents"][0]]
        result = validate_scenario_config(config)
        assert not result.valid

    def test_empty_scenes(self):
        config = _valid_config()
        config["scenes"] = []
        result = validate_scenario_config(config)
        assert not result.valid

    def test_invalid_scene_steps(self):
        config = _valid_config()
        config["scenes"] = [{"steps": 10}]
        result = validate_scenario_config(config)
        assert not result.valid

    def test_forced_choice_invalid_agent_name(self):
        config = _valid_config()
        config["scenes"] = [
            {
                "steps": 3,
                "forced_choice": {
                    "agent_name": "Nonexistent Agent",
                    "choices": ["I take action A in this scenario.", "I take action B in this scenario."],
                    "call_to_action": "A critical moment arrives. What does Nonexistent Agent do?",
                },
            }
        ]
        result = validate_scenario_config(config)
        assert not result.valid
        assert any("does not match" in e for e in result.errors)

    def test_forced_choice_valid_agent_name(self):
        config = _valid_config()
        config["scenes"] = [
            {
                "steps": 3,
                "forced_choice": {
                    "agent_name": "Dr. Chen",
                    "choices": ["I administer the experimental drug.", "I follow standard protocol instead."],
                    "call_to_action": "The patient is fading fast. What does Dr. Chen do?",
                },
            }
        ]
        result = validate_scenario_config(config)
        assert result.valid

    def test_max_agents_override(self):
        config = _valid_config()
        # Default max_agents=2, config has 2 agents — should pass
        assert validate_scenario_config(config, max_agents=2).valid
        # Override to 1 — should fail
        result = validate_scenario_config(config, max_agents=1)
        assert not result.valid
        assert any("too many agents" in e for e in result.errors)

    def test_invalid_philosophy(self):
        config = _valid_config()
        config["agents"][0]["philosophy"] = "invalid_philosophy"
        result = validate_scenario_config(config)
        assert not result.valid

    def test_additional_properties_rejected(self):
        config = _valid_config()
        config["extra_field"] = "should not be here"
        result = validate_scenario_config(config)
        assert not result.valid
