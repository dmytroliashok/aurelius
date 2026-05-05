"""Pydantic models representing Concordia simulation setup.

These are intermediate types used by the translator to produce a
structured setup that can be serialized to JSON and consumed by
the Concordia entrypoint inside the Docker container.
"""

from __future__ import annotations

from pydantic import BaseModel


class ChainOfThoughtStep(BaseModel):
    """A single reasoning step in the 7-step moral reasoning chain."""

    step: str  # e.g. "situation_perception", "theory_of_mind"
    response: str


class AgentPrefab(BaseModel):
    """A Concordia agent configuration."""

    name: str
    identity: str  # First-person backstory → Identity component
    goal: str  # First-person goal → Goal component
    philosophy: str  # Moral framework → reasoning component modifier
    philosophy_prompt: str  # Expanded prompt for the reasoning component


class ForcedChoiceComponent(BaseModel):
    """A forced binary choice injected into a scene."""

    agent_name: str
    choices: list[str]
    call_to_action: str


class SceneSetup(BaseModel):
    """Configuration for a single simulation scene."""

    steps: int
    mode: str  # "decision" or "reflection"
    forced_choice: ForcedChoiceComponent | None = None


class GameMasterConfig(BaseModel):
    """Configuration for the Concordia Game Master."""

    shared_context: str  # From premise
    tension_framing: str  # From tension_archetype + tension_description
    domain_label: str  # From morebench_context


class ConcordiaSetup(BaseModel):
    """Complete setup for a Concordia simulation run."""

    game_master: GameMasterConfig
    agents: list[AgentPrefab]
    scenes: list[SceneSetup]
    metadata: dict = {}
