"""Coherence validation for simulation transcripts.

Two levels of checking:
  1. Structural checks (fast, no LLM needed): completion, participation,
     empty content, repetition, minimum events.
  2. Semantic checks (optional, LLM-based): contextual relevance of agent
     actions to the scenario premise, and whether forced choices are
     meaningfully exercised (not just a restatement of one option).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aurelius.simulation.transcript import EventType, Transcript

logger = logging.getLogger(__name__)


@dataclass
class CoherenceResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    semantic_checks: dict[str, bool] = field(default_factory=dict)


def validate_coherence(
    transcript: Transcript,
    expected_agents: list[str] | None = None,
    min_events: int = 5,
    max_repetition_ratio: float = 0.5,
) -> CoherenceResult:
    """Validate that a simulation transcript is coherent (structural checks).

    Checks:
    1. Simulation completed all scenes
    2. All expected agents participated (at least one event each)
    3. Forced choices were exercised (if present)
    4. Minimum number of events
    5. No degenerate patterns (empty content, excessive repetition)

    Args:
        transcript: Parsed simulation transcript.
        expected_agents: List of agent names that should appear. If None, uses transcript.agent_names.
        min_events: Minimum number of events for a coherent simulation.
        max_repetition_ratio: Maximum ratio of identical content to total events.

    Returns:
        CoherenceResult with pass/fail and reasons.
    """
    reasons = []

    # 1. Simulation completed
    if not transcript.completed:
        reasons.append("Simulation did not complete")

    # 2. Minimum events
    if len(transcript.events) < min_events:
        reasons.append(f"Too few events: {len(transcript.events)} < {min_events}")

    # 3. All agents participated
    participating_agents = {e.agent for e in transcript.events if e.agent}
    if expected_agents:
        expected_set = set(expected_agents)
        missing = expected_set - participating_agents
        if missing:
            reasons.append(f"Missing agents: {missing}")
        unexpected = participating_agents - expected_set
        if unexpected:
            reasons.append(f"Unexpected agents (not in config): {unexpected}")
    elif len(participating_agents) < 2:
        reasons.append(f"Only {len(participating_agents)} agent(s) participated, expected ≥2")

    # 4. Forced choices exercised
    forced_choices = [e for e in transcript.events if e.type == EventType.FORCED_CHOICE]
    # We don't enforce a specific count — just verify they're not empty
    for fc in forced_choices:
        if not fc.content.strip():
            reasons.append("Empty forced choice response")

    # 5. No empty content
    content_events = [e for e in transcript.events if e.type in (EventType.ACTION, EventType.REFLECTION)]
    empty_count = sum(1 for e in content_events if not e.content.strip())
    if content_events and empty_count > len(content_events) * 0.3:
        reasons.append(f"Too many empty responses: {empty_count}/{len(content_events)}")

    # 6. Repetition detection
    if content_events:
        contents = [e.content.strip().lower() for e in content_events if e.content.strip()]
        if contents:
            unique_ratio = len(set(contents)) / len(contents)
            if unique_ratio < (1 - max_repetition_ratio):
                reasons.append(f"Excessive repetition: {1 - unique_ratio:.0%} duplicate content")

    # 7. Chain-of-thought validation (when present)
    cot_events = [e for e in content_events if e.chain_of_thought]
    if cot_events:
        for event in cot_events:
            empty_steps = sum(1 for s in event.chain_of_thought if not s.get("response", "").strip())
            if empty_steps > len(event.chain_of_thought) * 0.5:
                reasons.append(
                    f"Too many empty CoT steps for {event.agent} at scene {event.scene_index}: "
                    f"{empty_steps}/{len(event.chain_of_thought)}"
                )
                break  # One failure is enough

    return CoherenceResult(passed=len(reasons) == 0, reasons=reasons)


async def validate_semantic_coherence(
    transcript: Transcript,
    scenario_config: dict,
    llm_provider=None,
) -> CoherenceResult:
    """Validate semantic coherence using an LLM judge.

    This is an optional deeper check that verifies:
    1. Agent actions are contextually relevant to the scenario premise
    2. Forced choices are meaningfully exercised (not trivially restated)
    3. Agents stay in character with their assigned philosophies

    Args:
        transcript: Parsed simulation transcript.
        scenario_config: The original scenario config dict.
        llm_provider: An LLM provider instance (from aurelius.common.llm).
                      If None, falls back to structural checks only.

    Returns:
        CoherenceResult with semantic check details.
    """
    # First run structural checks — extract agent names from config dicts
    agent_cfgs = scenario_config.get("agents", [])
    agent_names = [a.get("name", "") if isinstance(a, dict) else a for a in agent_cfgs if a]
    structural = validate_coherence(transcript, expected_agents=agent_names)
    if not structural.passed:
        return structural

    if llm_provider is None:
        # No LLM available — structural checks are sufficient
        return structural

    semantic_checks = {}
    reasons = list(structural.reasons)

    premise = scenario_config.get("premise", "")
    agents = scenario_config.get("agents", [])

    # Build a summary of agent actions for the judge
    agent_actions = {}
    for event in transcript.events:
        if event.type in (EventType.ACTION, EventType.REFLECTION, EventType.FORCED_CHOICE) and event.agent:
            agent_actions.setdefault(event.agent, []).append(event.content)

    # --- Check 1: Contextual relevance ---
    actions_summary = "\n".join(f"{name}: {' | '.join(actions[:3])}" for name, actions in agent_actions.items())

    relevance_prompt = (
        f"You are evaluating whether agents in a moral dilemma simulation stayed contextually relevant.\n\n"
        f"PREMISE: {premise[:500]}\n\n"
        f"AGENT ACTIONS:\n{actions_summary[:1000]}\n\n"
        f"Are the agent actions relevant to the premise? Do they engage with the moral dilemma described?\n"
        f"Respond with exactly 'YES' or 'NO' followed by a one-sentence explanation."
    )

    try:
        response = await llm_provider.complete(
            relevance_prompt,
            system="You are a precise evaluator. Answer YES or NO then explain briefly.",
        )
        is_relevant = response.strip().upper().startswith("YES")
        semantic_checks["contextual_relevance"] = is_relevant
        if not is_relevant:
            reasons.append(f"Agents not contextually relevant to premise: {response[:100]}")
    except Exception as e:
        logger.warning("Semantic relevance check failed (fail closed): %s", e)
        semantic_checks["contextual_relevance"] = False
        reasons.append(f"Semantic relevance check unavailable: {e}")

    # --- Check 2: Meaningful forced choice ---
    forced_choices = [e for e in transcript.events if e.type == EventType.FORCED_CHOICE]
    for fc in forced_choices:
        choices = fc.metadata.get("choices", [])
        if len(choices) < 2:
            continue

        fc_prompt = (
            f"A character was asked to choose between:\n"
            f"  1. {choices[0]}\n"
            f"  2. {choices[1]}\n\n"
            f"Their response was: {fc.content[:300]}\n\n"
            f"Did the character meaningfully engage with the choice (explaining reasoning, "
            f"showing internal conflict, or justifying their decision)? Or did they simply "
            f"restate one option without reasoning?\n"
            f"Respond with exactly 'MEANINGFUL' or 'TRIVIAL' followed by a one-sentence explanation."
        )

        try:
            response = await llm_provider.complete(
                fc_prompt,
                system="You are a precise evaluator. Answer MEANINGFUL or TRIVIAL then explain briefly.",
            )
            is_meaningful = response.strip().upper().startswith("MEANINGFUL")
            semantic_checks[f"forced_choice_{fc.agent}"] = is_meaningful
            if not is_meaningful:
                reasons.append(f"Forced choice by {fc.agent} was trivial: {response[:100]}")
        except Exception as e:
            logger.warning("Semantic forced choice check failed (fail closed): %s", e)
            semantic_checks[f"forced_choice_{fc.agent}"] = False
            reasons.append(f"Semantic forced choice check unavailable for {fc.agent}: {e}")

    # --- Check 3: Philosophy alignment ---
    for agent_cfg in agents:
        name = agent_cfg.get("name", "")
        philosophy = agent_cfg.get("philosophy", "")
        if not philosophy or philosophy == "" or name not in agent_actions:
            continue

        agent_text = " ".join(agent_actions[name][:3])[:500]
        philosophy_prompt = (
            f"Agent '{name}' is assigned the moral philosophy: {philosophy}.\n\n"
            f"Their actions in the simulation: {agent_text}\n\n"
            f"Does the agent's reasoning reflect their assigned philosophy? "
            f"They don't need to name it explicitly, but their reasoning should be "
            f"consistent with {philosophy} principles.\n"
            f"Respond with exactly 'ALIGNED' or 'MISALIGNED' followed by a one-sentence explanation."
        )

        try:
            response = await llm_provider.complete(
                philosophy_prompt,
                system="You are a precise evaluator. Answer ALIGNED or MISALIGNED then explain briefly.",
            )
            is_aligned = response.strip().upper().startswith("ALIGNED")
            semantic_checks[f"philosophy_{name}"] = is_aligned
            if not is_aligned:
                reasons.append(f"Agent {name} not aligned with {philosophy}: {response[:100]}")
        except Exception as e:
            logger.warning("Semantic philosophy check failed for %s (fail closed): %s", name, e)
            semantic_checks[f"philosophy_{name}"] = False
            reasons.append(f"Semantic philosophy check unavailable for {name}: {e}")

    return CoherenceResult(
        passed=len(reasons) == 0,
        reasons=reasons,
        semantic_checks=semantic_checks,
    )
