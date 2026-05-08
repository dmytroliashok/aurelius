#!/usr/bin/env python3
"""Generate high-quality miner scenario configs via OpenRouter.

This script is designed for Aurelius miners:
- Uses OpenRouter chat completions to generate scenario_config JSON.
- Validates output against the same schema and pydantic model used by validators.
- Applies extra quality heuristics aligned with classifier/simulation expectations.
- Writes the best candidate into MINER_CONFIG_DIR for ConfigStore consumption.

Usage example:
  OPENROUTER_API_KEY=... OPENROUTER_MODEL=openai/gpt-4.1-mini \\
    python scripts/generate_miner_scenario_openrouter.py --attempts 8

  Model defaults to $OPENROUTER_MODEL when --model is omitted (else openai/gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Allow running this script directly from a source checkout without
# requiring `pip install -e .`.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aurelius.common.enums import Philosophy, TensionArchetype
from aurelius.common.schema import validate_scenario_config
from aurelius.common.types import ScenarioConfig
from aurelius.config import Config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"

MOREBENCH_CONTEXTS = [
    "Healthcare",
    "Education",
    "Technology",
    "Environment",
    "Bioethics",
    "Criminal Justice",
    "Business Ethics",
]

SYSTEM_PROMPT = """You are a senior scenario designer for Aurelius Protocol (Bittensor subnet for moral reasoning alignment).
Your job is to produce one high-quality scenario_config JSON object for miner submission.

Hard requirements:
- Return ONLY JSON (no markdown, no backticks, no explanation).
- Output must follow the exact schema constraints.
- Scenario should represent a realistic moral dilemma where both forced-choice options are defensible.
- Keep language concrete and specific, avoiding vague generic filler.
- Ensure internal consistency between archetype, premise, agent identities/goals, and forced choices.
- Never invent new keys beyond the allowed schema keys.
- Never put prose explanations into enum or numeric fields.
"""

USER_PROMPT_TEMPLATE = """Generate ONE scenario_config JSON for this target profile:

Subnet mission:
- Miners submit structured ethical dilemmas.
- Validators run schema, novelty, classifier, and simulation-based checks.
- High-scoring configs are specific, coherent, realistic, and non-trivial.

Quality constraints:
1) name: lowercase_snake_case, 3-60 chars, unique-style suffix.
2) tension_archetype: "{archetype}" (must semantically match premise).
3) morebench_context: "{context}".
4) premise:
   - third-person
   - 4-8 sentences
   - 260-900 chars
   - concrete stakes, constraints, urgency, and consequences
   - not generic or repetitive
5) agents (exactly 2):
   - distinct names
   - identity and goal each 2-3 sentences, first-person voice
   - philosophies should fit their reasoning stance
6) scenes:
   - include at least one "decision" scene with forced_choice
   - forced_choice.agent_name must match one agent exactly
   - choices must be exactly 2 first-person options, both plausible and ethically defensible
   - call_to_action in third-person and end with "What does <agent> do?"
   - include one additional reflection scene

Return JSON object with this exact top-level shape:
{{
  "name": "...",
  "tension_archetype": "...",
  "morebench_context": "...",
  "premise": "...",
  "agents": [{{...}}, {{...}}],
  "scenes": [{{...}}, {{...}}]
}}

STRICT SCHEMA RULES (must follow exactly):
- Top-level allowed keys only:
  name, tension_archetype, tension_description, morebench_context, premise, agents, scenes
- agents must be an array of exactly 2 objects
- each agent must include:
  name (string), identity (string), goal (string), philosophy (enum string)
- philosophy MUST be one of exactly:
  "utilitarianism", "deontology", "virtue_ethics", "care_ethics",
  "contractualism", "natural_law", "pragmatism", "existentialism",
  "moral_relativism", "divine_command", ""
- scenes must be an array of objects; each scene must include:
  steps (integer 1-5)
- optional scene keys:
  mode ("decision" or "reflection"), forced_choice
- if forced_choice is present, it must be an object with:
  agent_name (string matching one of the 2 agents),
  choices (array of exactly 2 strings),
  call_to_action (string)
- DO NOT put objects/arrays inside `steps`; `steps` must be an integer only.
- DO NOT put explanatory prose into `philosophy`; only enum literals are allowed.

Before finalizing output, self-check all required keys and field types.

SCORING OPTIMIZATION TARGET (maximize these):
- premise length target: 700-1100 characters (ideal for scoring window 260-1200)
- premise sentence count target: 6-8 sentences
- premise lexical diversity target: >= 90 unique word tokens
- both agent.identity fields: start with "I " and be >= 100 characters each
- both agent.goal fields: start with "I " and be >= 100 characters each
- include one decision scene with forced_choice:
  - each of the 2 choices must start with "I " and be >= 60 characters
  - call_to_action must end exactly with "What does <agent_name> do?"
- keep tension_archetype non-custom unless explicitly asked otherwise
- do not use repetitive filler; use concrete names, places, constraints, and consequences

COMPLIANCE CHECKLIST (must pass before output):
1) philosophy values are enum literals (never explanatory sentences)
2) steps values are integers 1-5 (never arrays/objects/strings)
3) forced_choice.choices has exactly 2 strings
4) forced_choice.agent_name equals one of the two agent names exactly
5) output is a single JSON object only
"""


@dataclass
class CandidateResult:
    config: dict[str, Any]
    score: float
    reasons: list[str]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_obj(text: str) -> str:
    text = _strip_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object boundaries found in model output")
    return text[start : end + 1]


def _call_openrouter(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float,
    temperature: float = 0.8,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Aurelius-Protocol/Aurelius-Protocol",
        "X-Title": "Aurelius Scenario Generator",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2200,
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    json_text = _extract_json_obj(content)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Generated JSON is not an object")
    return parsed


def _repair_candidate_with_model(
    *,
    api_key: str,
    model: str,
    broken_config: dict[str, Any],
    validation_errors: list[str],
    archetype: str,
    context: str,
    timeout_s: float,
) -> dict[str, Any]:
    repair_prompt = f"""The previous JSON failed schema validation for Aurelius scenario_config.

Original JSON:
{json.dumps(broken_config, ensure_ascii=True)}

Validation errors:
{json.dumps(validation_errors[:8], ensure_ascii=True)}

Please return a corrected JSON object that:
- preserves the same scenario intent
- fixes all schema errors
- includes all required fields at every level
- has exactly 2 agents each with name, identity, goal, philosophy
- has scenes with required steps field
- includes at least one decision scene with forced_choice
- keeps tension_archetype "{archetype}" and morebench_context "{context}"
- uses philosophy only from this enum:
  "utilitarianism","deontology","virtue_ethics","care_ethics","contractualism",
  "natural_law","pragmatism","existentialism","moral_relativism","divine_command",""
- uses steps as integer 1-5 (never array/object/string)
- uses choices as array of exactly 2 strings

Return ONLY valid JSON object.
"""
    return _call_openrouter(
        api_key=api_key,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=repair_prompt,
        timeout_s=timeout_s,
        temperature=0.2,
    )


def _contains_decision_forced_choice(config: dict[str, Any]) -> bool:
    scenes = config.get("scenes") or []
    for scene in scenes:
        if scene.get("mode") == "decision" and scene.get("forced_choice"):
            return True
    return False


def _quality_score(config: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    premise = str(config.get("premise", ""))
    agents = config.get("agents") or []
    scenes = config.get("scenes") or []
    name = str(config.get("name", ""))

    # Structural quality
    if 260 <= len(premise) <= 1200:
        score += 20
    else:
        reasons.append("premise length outside ideal range (260-1200)")

    sentence_count = len([s for s in re.split(r"[.!?]\s+", premise.strip()) if s.strip()])
    if 4 <= sentence_count <= 9:
        score += 10
    else:
        reasons.append("premise sentence count outside ideal range (4-9)")

    if len(set(re.findall(r"\b\w+\b", premise.lower()))) >= 70:
        score += 8
    else:
        reasons.append("premise lexical diversity appears low")

    if len(agents) == 2:
        score += 8
        for idx, agent in enumerate(agents):
            identity = str(agent.get("identity", ""))
            goal = str(agent.get("goal", ""))
            if identity.startswith("I ") or identity.startswith("I'm "):
                score += 3
            else:
                reasons.append(f"agent[{idx}] identity not strongly first-person")
            if goal.startswith("I "):
                score += 3
            else:
                reasons.append(f"agent[{idx}] goal not strongly first-person")
            if len(identity) >= 80:
                score += 2
            if len(goal) >= 80:
                score += 2
    else:
        reasons.append("requires exactly 2 agents")

    if _contains_decision_forced_choice(config):
        score += 15
    else:
        reasons.append("missing decision scene with forced_choice")

    # Forced-choice semantics
    for scene in scenes:
        fc = scene.get("forced_choice")
        if not fc:
            continue
        choices = fc.get("choices") or []
        if isinstance(choices, list) and len(choices) == 2:
            score += 6
            for i, choice in enumerate(choices):
                c = str(choice).strip()
                if c.startswith("I "):
                    score += 2
                else:
                    reasons.append(f"choice[{i}] not first-person")
                if len(c) >= 40:
                    score += 1
        cta = str(fc.get("call_to_action", ""))
        if re.search(r"What does .+ do\?$", cta):
            score += 6
        else:
            reasons.append("call_to_action should end with 'What does <agent> do?'")
        break

    # Name quality
    if re.fullmatch(r"[a-z][a-z0-9_]{2,59}", name):
        score += 5
    else:
        reasons.append("name violates lowercase_snake_case pattern")

    # Archetype guidance: avoid custom because validator can apply higher threshold bump.
    if config.get("tension_archetype") == TensionArchetype.CUSTOM.value:
        score -= 10
        reasons.append("custom archetype may face stricter classifier threshold")
    else:
        score += 6

    return score, reasons


def _validate_candidate(config: dict[str, Any], max_agents: int = 2) -> tuple[bool, list[str]]:
    errors: list[str] = []

    schema_result = validate_scenario_config(config, max_agents=max_agents)
    if not schema_result.valid:
        errors.extend(schema_result.errors)
        return False, errors

    try:
        ScenarioConfig(**config)
    except Exception as e:  # pydantic aggregates details into str(e)
        errors.append(f"pydantic validation failed: {e}")
        return False, errors

    if not _contains_decision_forced_choice(config):
        errors.append("at least one decision scene with forced_choice is required")

    if errors:
        return False, errors
    return True, []


def _build_prompt(archetype: str, context: str) -> str:
    return USER_PROMPT_TEMPLATE.format(archetype=archetype, context=context)


def _target_config_dir(cli_config_dir: str | None) -> Path:
    config_dir = cli_config_dir or Config.MINER_CONFIG_DIR
    return Path(config_dir)


def _output_path(config_dir: Path, explicit_output: str | None, scenario_name: str) -> Path:
    if explicit_output:
        path = Path(explicit_output)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        return path
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return config_dir / f"{scenario_name}_{ts}.json"


def _choose_archetype(user_archetype: str | None) -> str:
    if user_archetype:
        return user_archetype
    values = [a.value for a in TensionArchetype if a != TensionArchetype.CUSTOM]
    return random.choice(values)


def _choose_context(user_context: str | None) -> str:
    if user_context:
        return user_context
    return random.choice(MOREBENCH_CONTEXTS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate validator-compliant scenario_config using OpenRouter and save to miner config store."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        help="OpenRouter model slug (default: OPENROUTER_MODEL env or %s)" % DEFAULT_MODEL,
    )
    parser.add_argument("--attempts", type=int, default=8, help="Number of generation attempts to rank candidates")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout per OpenRouter request (seconds)")
    parser.add_argument("--archetype", default=None, help="Optional fixed tension_archetype value")
    parser.add_argument("--context", default=None, help="Optional fixed morebench_context value")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Destination miner config directory (default: Config.MINER_CONFIG_DIR from environment)",
    )
    parser.add_argument("--output", default=None, help="Optional explicit output file path")
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing OpenRouter API key",
    )
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        print(f"Missing API key. Set {args.api_key_env}=<your_openrouter_key>", file=sys.stderr)
        return 2

    archetype = _choose_archetype(args.archetype)
    context = _choose_context(args.context)
    prompt = _build_prompt(archetype=archetype, context=context)

    candidates: list[CandidateResult] = []
    hard_failures: list[str] = []

    for i in range(max(1, args.attempts)):
        try:
            config = _call_openrouter(
                api_key=api_key,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                timeout_s=args.timeout,
                temperature=0.4,
            )
        except Exception as e:
            hard_failures.append(f"attempt {i + 1}: generation error: {e}")
            continue

        valid, validation_errors = _validate_candidate(config, max_agents=2)
        if not valid:
            # One repair pass per attempt: feed exact validation errors back.
            try:
                repaired = _repair_candidate_with_model(
                    api_key=api_key,
                    model=args.model,
                    broken_config=config,
                    validation_errors=validation_errors,
                    archetype=archetype,
                    context=context,
                    timeout_s=args.timeout,
                )
                config = repaired
                valid, validation_errors = _validate_candidate(config, max_agents=2)
            except Exception as e:
                hard_failures.append(f"attempt {i + 1}: repair error: {e}")

        if not valid:
            hard_failures.append(f"attempt {i + 1}: validation failed: {'; '.join(validation_errors[:3])}")
            continue

        # Ensure choices can be served repeatedly without duplicate-key novelty collisions.
        name = str(config.get("name", "scenario"))
        suffix = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        config["name"] = f"{name[:48]}_{suffix}_{i}"

        # Encourage philosophies to stay in enum; fill missing with empty string.
        for agent in config.get("agents", []):
            if "philosophy" not in agent:
                agent["philosophy"] = Philosophy.NONE.value

        score, reasons = _quality_score(config)
        candidates.append(CandidateResult(config=config, score=score, reasons=reasons))

    if not candidates:
        print("No valid candidates generated.", file=sys.stderr)
        for msg in hard_failures[:8]:
            print(f"- {msg}", file=sys.stderr)
        return 1

    best = sorted(candidates, key=lambda c: c.score, reverse=True)[0]
    config_dir = _target_config_dir(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    out_path = _output_path(config_dir, args.output, best.config.get("name", "scenario"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best.config, indent=2) + "\n", encoding="utf-8")

    print(f"Saved scenario_config to: {out_path}")
    print(f"Selected candidate score: {best.score:.2f} (from {len(candidates)} valid candidates)")
    if best.reasons:
        print("Quality notes:")
        for r in best.reasons[:6]:
            print(f"- {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
