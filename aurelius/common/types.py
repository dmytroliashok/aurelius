from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator

from aurelius.common.enums import Philosophy, SceneMode, TensionArchetype


@dataclass
class ConsumeResult:
    """Result of a work-token consumption attempt. Shared between API and validator."""

    success: bool
    deducted: bool
    valid: bool
    message: str


class ForcedChoice(BaseModel):
    """A binary forced choice presented to an agent during a scene."""

    agent_name: str = Field(min_length=2, max_length=30)
    choices: list[str] = Field(min_length=2, max_length=2)
    call_to_action: str = Field(min_length=10, max_length=500)


class AgentConfig(BaseModel):
    """Configuration for a single agent in a moral dilemma scenario."""

    name: str = Field(min_length=2, max_length=30)
    identity: str = Field(min_length=10, max_length=500)
    goal: str = Field(min_length=10, max_length=500)
    philosophy: Philosophy = Philosophy.NONE


class SceneSpec(BaseModel):
    """Specification for a single scene in a scenario."""

    steps: int = Field(ge=1, le=5)
    mode: SceneMode = SceneMode.DECISION
    forced_choice: ForcedChoice | None = None


class ScenarioConfig(BaseModel):
    """Top-level scenario configuration for a moral dilemma."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,59}$")
    tension_archetype: TensionArchetype
    tension_description: str | None = Field(default=None, min_length=20, max_length=200)
    morebench_context: str = Field(min_length=1, max_length=100)
    premise: str = Field(min_length=200, max_length=2000)
    agents: list[AgentConfig] = Field(min_length=2, max_length=2)
    scenes: list[SceneSpec] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_custom_tension(self):
        if self.tension_archetype == TensionArchetype.CUSTOM and not self.tension_description:
            raise ValueError("tension_description is required when tension_archetype is 'custom'")
        if self.tension_archetype != TensionArchetype.CUSTOM and self.tension_description:
            raise ValueError("tension_description is only allowed when tension_archetype is 'custom'")
        return self

    @model_validator(mode="after")
    def validate_unique_agent_names(self):
        names = [a.name for a in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")
        return self

    @model_validator(mode="after")
    def validate_forced_choice_agent_names(self):
        agent_names = {a.name for a in self.agents}
        for scene in self.scenes:
            if scene.forced_choice and scene.forced_choice.agent_name not in agent_names:
                raise ValueError(
                    f"forced_choice.agent_name '{scene.forced_choice.agent_name}' "
                    f"does not match any agent: {agent_names}"
                )
        return self
