import pytest
from pydantic import ValidationError

from aurelius.common.enums import Philosophy, SceneMode, TensionArchetype
from aurelius.common.types import AgentConfig, ForcedChoice, ScenarioConfig, SceneSpec


def _make_premise(length: int = 250) -> str:
    """Generate a premise string of approximately the given length."""
    base = "A doctor faces a difficult decision in a rural hospital where resources are limited. "
    return (base * (length // len(base) + 1))[:length]


def _make_config(**overrides) -> dict:
    """Build a valid ScenarioConfig dict, applying overrides."""
    defaults = {
        "name": "test_scenario_one",
        "tension_archetype": TensionArchetype.JUSTICE_VS_MERCY,
        "morebench_context": "Healthcare",
        "premise": _make_premise(250),
        "agents": [
            AgentConfig(
                name="Dr. Chen",
                identity="I am a surgeon with 20 years of experience in emergency medicine.",
                goal="I want to save the patient while following hospital protocol.",
                philosophy=Philosophy.DEONTOLOGY,
            ),
            AgentConfig(
                name="Nurse Patel",
                identity="I am a senior nurse who has seen the consequences of shortcuts.",
                goal="I want to ensure patient safety above all bureaucratic concerns.",
                philosophy=Philosophy.CARE_ETHICS,
            ),
        ],
        "scenes": [
            SceneSpec(steps=3, mode=SceneMode.DECISION),
            SceneSpec(steps=2, mode=SceneMode.REFLECTION),
        ],
    }
    defaults.update(overrides)
    return defaults


class TestAgentConfig:
    def test_valid_agent(self):
        agent = AgentConfig(
            name="Dr. Chen",
            identity="I am a surgeon with years of experience in emergency medicine.",
            goal="I want to save the patient while following protocol.",
            philosophy=Philosophy.UTILITARIANISM,
        )
        assert agent.name == "Dr. Chen"
        assert agent.philosophy == Philosophy.UTILITARIANISM

    def test_default_philosophy_is_none(self):
        agent = AgentConfig(
            name="Dr. Chen",
            identity="I am a surgeon with years of experience in emergency medicine.",
            goal="I want to save the patient while following protocol.",
        )
        assert agent.philosophy == Philosophy.NONE

    def test_name_too_short(self):
        with pytest.raises(ValidationError, match="name"):
            AgentConfig(name="A", identity="I am a valid identity string.", goal="I am a valid goal string here.")

    def test_name_too_long(self):
        with pytest.raises(ValidationError, match="name"):
            AgentConfig(
                name="A" * 31, identity="I am a valid identity string.", goal="I am a valid goal string here."
            )


class TestForcedChoice:
    def test_valid_forced_choice(self):
        fc = ForcedChoice(
            agent_name="Dr. Chen",
            choices=["I administer the experimental drug.", "I follow standard protocol."],
            call_to_action="The patient is fading fast. What does Dr. Chen do?",
        )
        assert len(fc.choices) == 2

    def test_must_have_exactly_two_choices(self):
        with pytest.raises(ValidationError, match="choices"):
            ForcedChoice(
                agent_name="Dr. Chen",
                choices=["Only one choice"],
                call_to_action="The patient is fading fast. What does Dr. Chen do?",
            )

    def test_cannot_have_three_choices(self):
        with pytest.raises(ValidationError, match="choices"):
            ForcedChoice(
                agent_name="Dr. Chen",
                choices=["Choice A", "Choice B", "Choice C"],
                call_to_action="The patient is fading fast. What does Dr. Chen do?",
            )


class TestScenarioConfig:
    def test_valid_config(self):
        config = ScenarioConfig(**_make_config())
        assert config.name == "test_scenario_one"
        assert len(config.agents) == 2
        assert len(config.scenes) == 2

    def test_invalid_name_format(self):
        with pytest.raises(ValidationError, match="name"):
            ScenarioConfig(**_make_config(name="Invalid Name!"))

    def test_name_uppercase_rejected(self):
        with pytest.raises(ValidationError, match="name"):
            ScenarioConfig(**_make_config(name="TestScenario"))

    def test_premise_too_short(self):
        with pytest.raises(ValidationError, match="premise"):
            ScenarioConfig(**_make_config(premise="Too short."))

    def test_custom_tension_requires_description(self):
        with pytest.raises(ValidationError, match="tension_description"):
            ScenarioConfig(**_make_config(tension_archetype=TensionArchetype.CUSTOM))

    def test_custom_tension_with_description_valid(self):
        config = ScenarioConfig(
            **_make_config(
                tension_archetype=TensionArchetype.CUSTOM,
                tension_description="A novel tension between technological progress and cultural preservation.",
            )
        )
        assert config.tension_archetype == TensionArchetype.CUSTOM

    def test_forced_choice_agent_must_match(self):
        scenes = [
            SceneSpec(
                steps=3,
                forced_choice=ForcedChoice(
                    agent_name="Nonexistent Agent",
                    choices=["Choice A action to take.", "Choice B action to take."],
                    call_to_action="A critical moment arrives. What does Nonexistent Agent do?",
                ),
            ),
        ]
        with pytest.raises(ValidationError, match="does not match any agent"):
            ScenarioConfig(**_make_config(scenes=scenes))

    def test_forced_choice_matching_agent_valid(self):
        scenes = [
            SceneSpec(
                steps=3,
                forced_choice=ForcedChoice(
                    agent_name="Dr. Chen",
                    choices=["I administer the experimental drug.", "I follow standard protocol."],
                    call_to_action="The patient is fading fast. What does Dr. Chen do?",
                ),
            ),
        ]
        config = ScenarioConfig(**_make_config(scenes=scenes))
        assert config.scenes[0].forced_choice.agent_name == "Dr. Chen"

    def test_too_few_agents(self):
        agents = [
            AgentConfig(
                name="Solo Agent",
                identity="I am alone in this scenario with nobody else.",
                goal="I must decide on my own what the right course is.",
            )
        ]
        with pytest.raises(ValidationError, match="agents"):
            ScenarioConfig(**_make_config(agents=agents))

    def test_empty_scenes_rejected(self):
        with pytest.raises(ValidationError, match="scenes"):
            ScenarioConfig(**_make_config(scenes=[]))
