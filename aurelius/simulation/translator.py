"""Translates Aurelius scenario configs to Concordia simulation setups.

Mapping (from spec Section 3.4.1):
- premise → Game Master shared context
- agents[].identity → agent Identity component
- agents[].goal → agent Goal component
- agents[].philosophy → agent reasoning component prompt modifier
- scenes[].steps → engine step count per scene
- scenes[].forced_choice → custom ForcedChoice component
- scenes[].mode='reflection' → reflection mode
- tension_archetype + tension_description → GM context framing
- morebench_context → domain label in GM prompting
"""

from aurelius.simulation.concordia_types import (
    AgentPrefab,
    ConcordiaSetup,
    ForcedChoiceComponent,
    GameMasterConfig,
    SceneSetup,
)

# Philosophy prompt expansions for Concordia agent reasoning
PHILOSOPHY_PROMPTS = {
    "utilitarianism": (
        "You reason about moral decisions by evaluating which action produces "
        "the greatest good for the greatest number. You weigh consequences and "
        "seek to maximize overall well-being."
    ),
    "deontology": (
        "You reason about moral decisions based on rules and duties. You believe "
        "certain actions are inherently right or wrong regardless of their consequences. "
        "You follow moral principles as categorical imperatives."
    ),
    "virtue_ethics": (
        "You reason about moral decisions by asking what a virtuous person would do. "
        "You focus on character traits like courage, temperance, justice, and wisdom "
        "rather than rules or consequences."
    ),
    "care_ethics": (
        "You reason about moral decisions through the lens of relationships and "
        "responsibilities to others. You prioritize empathy, compassion, and the "
        "needs of those who are vulnerable or dependent."
    ),
    "contractualism": (
        "You reason about moral decisions by considering what principles people "
        "could reasonably agree to. You seek rules that no one could reasonably reject "
        "as a basis for general agreement."
    ),
    "natural_law": (
        "You reason about moral decisions based on an inherent moral order in nature. "
        "You believe certain rights and values are universal and discoverable through reason."
    ),
    "pragmatism": (
        "You reason about moral decisions by focusing on practical outcomes and what works. "
        "You evaluate moral claims by their real-world effects and adapt your approach "
        "based on context and experience."
    ),
    "existentialism": (
        "You reason about moral decisions from the perspective of individual freedom "
        "and authentic choice. You believe each person must define their own values "
        "and take full responsibility for their decisions."
    ),
    "moral_relativism": (
        "You reason about moral decisions recognizing that moral judgments are shaped "
        "by cultural and personal context. You consider multiple perspectives and avoid "
        "imposing a single moral framework."
    ),
    "divine_command": (
        "You reason about moral decisions based on divine authority and religious teachings. "
        "You believe moral obligations come from a higher power and seek guidance from "
        "spiritual principles."
    ),
    "": "",  # No philosophy modifier
}


def translate_config(config: dict) -> ConcordiaSetup:
    """Translate a scenario config dict into a ConcordiaSetup.

    Args:
        config: Validated scenario config dict.

    Returns:
        ConcordiaSetup ready for simulation execution.
    """
    # Game Master
    archetype = config.get("tension_archetype", "")
    description = config.get("tension_description", "")
    if description:
        tension_framing = f"The moral tension in this scenario is: {description}"
    else:
        tension_framing = f"The moral tension in this scenario follows the archetype: {archetype.replace('_', ' ')}"

    gm = GameMasterConfig(
        shared_context=config.get("premise", ""),
        tension_framing=tension_framing,
        domain_label=config.get("morebench_context", ""),
    )

    # Agents
    agents = []
    for agent_data in config.get("agents", []):
        philosophy = agent_data.get("philosophy", "")
        agents.append(
            AgentPrefab(
                name=agent_data.get("name", ""),
                identity=agent_data.get("identity", ""),
                goal=agent_data.get("goal", ""),
                philosophy=philosophy,
                philosophy_prompt=PHILOSOPHY_PROMPTS.get(philosophy, ""),
            )
        )

    # Scenes
    scenes = []
    for scene_data in config.get("scenes", []):
        fc = None
        fc_data = scene_data.get("forced_choice")
        if fc_data:
            fc = ForcedChoiceComponent(
                agent_name=fc_data.get("agent_name", ""),
                choices=fc_data.get("choices", []),
                call_to_action=fc_data.get("call_to_action", ""),
            )
        scenes.append(
            SceneSetup(
                steps=scene_data.get("steps", 1),
                mode=scene_data.get("mode", "decision"),
                forced_choice=fc,
            )
        )

    return ConcordiaSetup(
        game_master=gm,
        agents=agents,
        scenes=scenes,
        metadata={"name": config.get("name", ""), "tension_archetype": archetype},
    )
