"""Assertion tests: Schema & Protocol (SP-01..SP-09)."""

import pytest
from pydantic import ValidationError

from aurelius.common.schema import validate_scenario_config
from aurelius.common.types import ScenarioConfig
from aurelius.common.version import VersionResult, check_compatibility


def _valid_config(**overrides) -> dict:
    config = {
        "name": "sp_assertion_test",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A" * 200,
        "agents": [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Bob", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "utilitarianism"},
        ],
        "scenes": [{"steps": 3, "mode": "decision"}, {"steps": 2, "mode": "reflection"}],
    }
    config.update(overrides)
    return config


class TestSP01MaxAgents:
    def test_sp01_too_many_agents_rejected(self):
        """A config with agent count > max_agents must be rejected."""
        config = _valid_config()
        config["agents"].append(
            {"name": "Carol", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "pragmatism"}
        )
        result = validate_scenario_config(config, max_agents=2)
        assert not result.valid
        assert any("agents" in e for e in result.errors)

    def test_sp01_within_limit_passes(self):
        result = validate_scenario_config(_valid_config(), max_agents=2)
        assert result.valid


class TestSP02CustomArchetypeRequiresDescription:
    def test_sp02_custom_without_description_rejected(self):
        """tension_archetype == 'custom' without tension_description must be rejected."""
        config = _valid_config(tension_archetype="custom")
        result = validate_scenario_config(config)
        assert not result.valid

    def test_sp02_custom_with_description_passes(self):
        config = _valid_config(tension_archetype="custom", tension_description="A" * 20)
        result = validate_scenario_config(config)
        assert result.valid


class TestSP03DescriptionWithoutCustomRejected:
    def test_sp03_description_with_non_custom_rejected_schema(self):
        """tension_description with non-custom archetype rejected at JSON schema layer."""
        config = _valid_config(tension_description="A" * 20)
        result = validate_scenario_config(config)
        assert not result.valid

    def test_sp03_description_with_non_custom_rejected_pydantic(self):
        """tension_description with non-custom archetype rejected at Pydantic layer."""
        with pytest.raises(ValidationError, match="tension_description"):
            ScenarioConfig(**_valid_config(tension_description="A" * 20))


class TestSP04ForcedChoiceAgentName:
    def test_sp04_agent_name_must_match(self):
        """forced_choice.agent_name must match an existing agent name."""
        config = _valid_config()
        config["scenes"][0]["forced_choice"] = {
            "agent_name": "NonExistent",
            "choices": ["A", "B"],
            "call_to_action": "What do you decide to do now?",
        }
        result = validate_scenario_config(config)
        assert not result.valid
        assert any("agent_name" in e for e in result.errors)


class TestSP05ForcedChoiceExactlyTwo:
    def test_sp05_one_choice_rejected(self):
        """forced_choice.choices must contain exactly 2 items."""
        config = _valid_config()
        config["scenes"][0]["forced_choice"] = {
            "agent_name": "Alice",
            "choices": ["OnlyOne"],
            "call_to_action": "What do you decide to do now?",
        }
        result = validate_scenario_config(config)
        assert not result.valid

    def test_sp05_three_choices_rejected(self):
        config = _valid_config()
        config["scenes"][0]["forced_choice"] = {
            "agent_name": "Alice",
            "choices": ["A", "B", "C"],
            "call_to_action": "What do you decide to do now?",
        }
        result = validate_scenario_config(config)
        assert not result.valid


class TestSP06PremiseLength:
    def test_sp06_too_short_rejected(self):
        """Premise length must be within [min_premise_length, max_premise_length]."""
        config = _valid_config(premise="Too short")
        result = validate_scenario_config(config)
        assert not result.valid

    def test_sp06_too_long_rejected(self):
        config = _valid_config(premise="A" * 2001)
        result = validate_scenario_config(config)
        assert not result.valid

    def test_sp06_at_minimum_passes(self):
        config = _valid_config(premise="A" * 200)
        result = validate_scenario_config(config)
        assert result.valid


class TestSP07MajorVersionMismatch:
    def test_sp07_major_mismatch_rejects(self):
        """A major protocol version mismatch must result in rejection."""
        assert check_compatibility("1.0.0", "2.0.0") == VersionResult.REJECT
        assert check_compatibility("2.0.0", "1.0.0") == VersionResult.REJECT

    def test_sp07_minor_mismatch_warns(self):
        assert check_compatibility("1.0.0", "1.1.0") == VersionResult.WARN

    def test_sp07_patch_accepts(self):
        assert check_compatibility("1.0.0", "1.0.1") == VersionResult.ACCEPT


class TestSP09DuplicateAgentNames:
    def test_sp09_duplicate_names_rejected_schema(self):
        """Agent names must be unique — schema validation layer."""
        config = _valid_config()
        config["agents"] = [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Alice", "identity": "J" * 20, "goal": "H" * 20, "philosophy": "utilitarianism"},
        ]
        result = validate_scenario_config(config)
        assert not result.valid
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_sp09_duplicate_names_rejected_pydantic(self):
        """Agent names must be unique — Pydantic validation layer."""
        config = _valid_config()
        config["agents"] = [
            {"name": "Alice", "identity": "I" * 20, "goal": "G" * 20, "philosophy": "deontology"},
            {"name": "Alice", "identity": "J" * 20, "goal": "H" * 20, "philosophy": "utilitarianism"},
        ]
        with pytest.raises(ValidationError, match="unique"):
            ScenarioConfig(**config)

    def test_sp09_unique_names_pass(self):
        result = validate_scenario_config(_valid_config())
        assert result.valid
