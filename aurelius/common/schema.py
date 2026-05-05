from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources

import jsonschema


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def _load_schema() -> dict:
    schema_file = resources.files("aurelius.common").joinpath("schema_v1.json")
    return json.loads(schema_file.read_text(encoding="utf-8"))


_SCHEMA: dict | None = None


def get_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = _load_schema()
    return _SCHEMA


def validate_scenario_config(config: dict, *, max_agents: int = 2) -> ValidationResult:
    """Validate a scenario config dict against the JSON Schema and additional business rules.

    Args:
        config: Raw scenario config dictionary.
        max_agents: Maximum allowed agents (from remote config). Overrides schema default.

    Returns:
        ValidationResult with valid=True if all checks pass, else errors populated.
    """
    schema = get_schema()
    errors: list[str] = []

    # JSON Schema validation
    validator = jsonschema.Draft202012Validator(schema)
    for error in validator.iter_errors(config):
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "(root)"
        errors.append(f"{path}: {error.message}")

    if errors:
        return ValidationResult(valid=False, errors=errors)

    # Dynamic max_agents check (overrides the schema's static maxItems)
    agents = config.get("agents", [])
    if len(agents) > max_agents:
        errors.append(f"agents: too many agents ({len(agents)} > max {max_agents})")

    # Unique agent names
    agent_names = {a["name"] for a in agents}
    if len(agent_names) != len(agents):
        errors.append("agents: duplicate agent names are not allowed")

    # Cross-field: forced_choice.agent_name must match an agent
    for i, scene in enumerate(config.get("scenes", [])):
        fc = scene.get("forced_choice")
        if fc and fc.get("agent_name") not in agent_names:
            errors.append(
                f"scenes[{i}].forced_choice.agent_name: "
                f"'{fc.get('agent_name')}' does not match any agent: {agent_names}"
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors)
