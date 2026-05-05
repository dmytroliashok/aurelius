from aurelius.simulation.translator import translate_config, PHILOSOPHY_PROMPTS
from aurelius.simulation.concordia_types import ConcordiaSetup


def _valid_config() -> dict:
    return {
        "name": "hospital_dilemma_one",
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A doctor faces a difficult decision in a hospital.",
        "agents": [
            {
                "name": "Dr. Chen",
                "identity": "I am a surgeon with experience.",
                "goal": "I want to save lives.",
                "philosophy": "deontology",
            },
            {
                "name": "Nurse Patel",
                "identity": "I am a senior nurse.",
                "goal": "I want patient safety.",
                "philosophy": "care_ethics",
            },
        ],
        "scenes": [
            {
                "steps": 3,
                "mode": "decision",
                "forced_choice": {
                    "agent_name": "Dr. Chen",
                    "choices": ["Administer the drug.", "Follow protocol."],
                    "call_to_action": "What does Dr. Chen do?",
                },
            },
            {"steps": 2, "mode": "reflection"},
        ],
    }


class TestTranslateConfig:
    def test_basic_translation(self):
        setup = translate_config(_valid_config())
        assert isinstance(setup, ConcordiaSetup)

    def test_game_master_fields(self):
        setup = translate_config(_valid_config())
        assert "doctor" in setup.game_master.shared_context.lower()
        assert "justice" in setup.game_master.tension_framing.lower()
        assert setup.game_master.domain_label == "Healthcare"

    def test_agents_translated(self):
        setup = translate_config(_valid_config())
        assert len(setup.agents) == 2
        assert setup.agents[0].name == "Dr. Chen"
        assert setup.agents[0].philosophy == "deontology"
        assert "rules and duties" in setup.agents[0].philosophy_prompt

    def test_scenes_translated(self):
        setup = translate_config(_valid_config())
        assert len(setup.scenes) == 2
        assert setup.scenes[0].steps == 3
        assert setup.scenes[0].mode == "decision"
        assert setup.scenes[1].mode == "reflection"

    def test_forced_choice_translated(self):
        setup = translate_config(_valid_config())
        fc = setup.scenes[0].forced_choice
        assert fc is not None
        assert fc.agent_name == "Dr. Chen"
        assert len(fc.choices) == 2

    def test_custom_tension_description(self):
        config = _valid_config()
        config["tension_archetype"] = "custom"
        config["tension_description"] = "A novel tension between innovation and tradition"
        setup = translate_config(config)
        assert "novel tension" in setup.game_master.tension_framing.lower()

    def test_metadata(self):
        setup = translate_config(_valid_config())
        assert setup.metadata["name"] == "hospital_dilemma_one"
        assert setup.metadata["tension_archetype"] == "justice_vs_mercy"

    def test_no_philosophy(self):
        config = _valid_config()
        config["agents"][0]["philosophy"] = ""
        setup = translate_config(config)
        assert setup.agents[0].philosophy_prompt == ""

    def test_all_philosophies_have_prompts(self):
        for key in PHILOSOPHY_PROMPTS:
            assert isinstance(PHILOSOPHY_PROMPTS[key], str)
