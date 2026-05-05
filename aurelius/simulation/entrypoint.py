#!/usr/bin/env python3
"""Entrypoint for the Concordia simulation container.

Reads a ConcordiaSetup JSON from the input path, runs the simulation
using the GDM Concordia framework (gdm-concordia), and writes the
structured output to the output path.

Uses component-based agents following the basic entity prefab pattern,
extended with a 7-step moral reasoning chain per agent turn:
  1. Situation perception
  2. Self perception
  3. Theory of mind
  4. Self interest
  5. Other interest
  6. Third-party / neutral observer
  7. Final action (ConcatActComponent)

Concordia is a hard requirement. If it is not installed the simulation
fails — there is no fallback path.

This script runs INSIDE the Docker container.

Environment variables:
  LLM_MODEL     - model name (default: "deepseek-chat")
  LLM_API_KEY   - API key (direct)
  LLM_API_KEY_FILE - path to file containing API key (preferred over env var)
  LLM_BASE_URL  - base URL for OpenAI-compatible API (default: DeepSeek)
"""

import json
import os
import sys
import time


def _load_api_key() -> str:
    """Load API key from file or environment variable.

    Docker runner passes the key via a mounted secrets file to avoid
    exposure in `docker inspect` output.
    """
    key_file = os.environ.get("LLM_API_KEY_FILE")
    if key_file and os.path.exists(key_file):
        with open(key_file) as f:
            return f.read().strip()
    return os.environ.get("LLM_API_KEY", "")


# ---------------------------------------------------------------------------
# 7-step moral reasoning chain configuration
# ---------------------------------------------------------------------------

COT_STEPS = [
    {
        "key": "situation_perception",
        "question": ("What kind of situation is {agent_name} in right now? Respond using 1-5 sentences."),
        "answer_prefix": "{agent_name} is currently ",
        "memory_tag": "",
        "pre_act_label": "Situation perception",
    },
    {
        "key": "self_perception",
        "question": ("What kind of person is {agent_name}? Respond using 1-5 sentences."),
        "answer_prefix": "{agent_name} is ",
        "memory_tag": "[self reflection]",
        "pre_act_label": "Self perception",
    },
    {
        "key": "theory_of_mind",
        "question": (
            "What do the other people in this situation think and feel"
            " about what is happening? What are their likely intentions"
            " and concerns? Respond using 1-5 sentences."
        ),
        "answer_prefix": "{agent_name} thinks the others ",
        "memory_tag": "[theory of mind]",
        "pre_act_label": "Theory of mind",
    },
    {
        "key": "self_interest",
        "question": (
            "What would be best for {agent_name} personally in this"
            " situation? What outcomes would most benefit them?"
            " Respond using 1-5 sentences."
        ),
        "answer_prefix": "For {agent_name} personally, the best outcome would be ",
        "memory_tag": "[self interest]",
        "pre_act_label": "Self interest",
    },
    {
        "key": "other_interest",
        "question": (
            "What would be best for the other people involved in this"
            " situation? What outcomes would most benefit them?"
            " Respond using 1-5 sentences."
        ),
        "answer_prefix": "What would be best for the others is ",
        "memory_tag": "[other interest]",
        "pre_act_label": "Other interest",
    },
    {
        "key": "neutral_observer",
        "question": (
            "What would a neutral, impartial third-party observer say is"
            " the right thing to do in this situation?"
            " Respond using 1-5 sentences."
        ),
        "answer_prefix": "An impartial observer would say ",
        "memory_tag": "[neutral observer]",
        "pre_act_label": "Third-party perspective",
    },
]


# ---------------------------------------------------------------------------
# Concordia path helpers
# ---------------------------------------------------------------------------


def _create_embedder(api_key: str):
    """Create an embedding function for AssociativeMemoryBank.

    Tries the OpenAI-compatible embedding endpoint first, falls back to
    a simple hash-based embedder for in-simulation memory retrieval.
    """
    import numpy as np

    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")

    # Try OpenAI-compatible embedding endpoint
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        # Test with a short string to see if embeddings are supported
        test = client.embeddings.create(input="test", model="text-embedding-3-small")
        embed_dim = len(test.data[0].embedding)

        def _api_embed(text: str) -> np.ndarray:
            resp = client.embeddings.create(input=text, model="text-embedding-3-small")
            return np.array(resp.data[0].embedding, dtype=np.float32)

        print(f"Using API embedder (dim={embed_dim})", file=sys.stderr)
        return _api_embed
    except Exception:
        pass

    # Fallback: deterministic hash-based embedder
    import hashlib

    embed_dim = 128

    def _hash_embed(text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # Tile to desired dimension
        vec = np.tile(arr, embed_dim // len(arr) + 1)[:embed_dim]
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    print("Using hash-based embedder (API embedding unavailable)", file=sys.stderr)
    return _hash_embed


def _build_agent(
    agent_cfg: dict,
    model,
    memory_bank_factory,
    shared_context: str,
):
    """Build a Concordia agent with the 7-step moral reasoning chain.

    Uses the same component architecture as the basic.py entity prefab,
    extended with 4 custom QuestionOfRecentMemories components for
    moral reasoning (theory of mind, self/other/third-party interest).

    Returns:
        Tuple of (agent, cot_component_keys) where cot_component_keys maps
        step names to the keys in context_components for CoT extraction.
    """
    from concordia.agents import entity_agent_with_logging
    from concordia.components import agent as agent_components
    from concordia.components.agent import memory as memory_module
    from concordia.components.agent import observation as obs_module

    name = agent_cfg["name"]

    # Create per-agent memory bank and seed with identity
    memory_bank = memory_bank_factory()
    memory_bank.add(f"[identity] {agent_cfg['identity']}")
    memory_bank.add(f"[goal] {agent_cfg['goal']}")
    if agent_cfg.get("philosophy_prompt"):
        memory_bank.add(f"[philosophy] {agent_cfg['philosophy_prompt']}")

    # --- Core components (from basic.py prefab pattern) ---
    memory = agent_components.memory.AssociativeMemory(memory_bank)
    instructions = agent_components.instructions.Instructions(agent_name=name)
    obs_to_memory = obs_module.ObservationToMemory()
    last_n_obs = obs_module.LastNObservations(
        history_length=1_000_000,
        pre_act_label="Observations",
    )

    # --- Constants for goal and philosophy ---
    context_components = {
        memory_module.DEFAULT_MEMORY_COMPONENT_KEY: memory,
        "Instructions": instructions,
        obs_module.DEFAULT_OBSERVATION_COMPONENT_KEY: obs_to_memory,
        "Observations": last_n_obs,
    }
    component_order = ["Instructions", "Observations"]

    if agent_cfg.get("goal"):
        goal = agent_components.constant.Constant(
            state=agent_cfg["goal"],
            pre_act_label="Goal",
        )
        context_components["Goal"] = goal
        component_order.append("Goal")

    if agent_cfg.get("philosophy_prompt"):
        philosophy = agent_components.constant.Constant(
            state=agent_cfg["philosophy_prompt"],
            pre_act_label="Moral philosophy",
        )
        context_components["Moral philosophy"] = philosophy
        component_order.append("Moral philosophy")

    # --- 7-step CoT reasoning chain ---
    cot_component_keys = {}

    for step_cfg in COT_STEPS:
        key = step_cfg["key"]
        label = step_cfg["pre_act_label"]

        # First two steps use Concordia's built-in subclasses
        if key == "situation_perception":
            component = agent_components.question_of_recent_memories.SituationPerception(
                model=model,
                pre_act_label=label,
                num_memories_to_retrieve=25,
            )
        elif key == "self_perception":
            component = agent_components.question_of_recent_memories.SelfPerception(
                model=model,
                pre_act_label=label,
                num_memories_to_retrieve=25,
            )
        else:
            # Custom moral reasoning steps
            component = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
                model=model,
                pre_act_label=label,
                question=step_cfg["question"].format(agent_name=name),
                answer_prefix=step_cfg["answer_prefix"].format(agent_name=name),
                add_to_memory=True,
                memory_tag=step_cfg["memory_tag"],
                num_memories_to_retrieve=10,
            )

        context_components[label] = component
        component_order.append(label)
        cot_component_keys[key] = label

    # --- Act component (final action) ---
    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=context_components,
    )

    return agent, cot_component_keys


def _extract_cot(agent, cot_component_keys: dict) -> list[dict]:
    """Extract chain-of-thought from an agent's last logged component outputs."""
    cot = []
    try:
        last_log = agent.get_last_log()
        for step_key, component_label in cot_component_keys.items():
            entry = last_log.get(component_label, "")
            # Concordia logs may be dicts with 'State' key or plain strings
            if isinstance(entry, dict):
                response = entry.get("State", entry.get("state", str(entry)))
            else:
                response = str(entry)
            cot.append({"step": step_key, "response": response})
    except Exception:
        # Degrade gracefully — return empty CoT rather than fail
        for step_key in cot_component_keys:
            cot.append({"step": step_key, "response": ""})
    return cot


def run_with_concordia(setup: dict, api_key: str) -> dict:
    """Run simulation using the GDM Concordia library with prefab patterns.

    Builds agents following the basic.py entity prefab component
    architecture, extended with a 7-step moral reasoning chain.
    Uses the generic.py Game Master pattern for narration and
    event resolution.
    """
    from concordia.associative_memory import basic_associative_memory
    from concordia.typing import entity as entity_lib
    from llm_model import make_model

    llm_model = os.environ.get("LLM_MODEL", "deepseek-chat")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")

    gm_config = setup["game_master"]
    agents_config = setup["agents"]
    scenes_config = setup["scenes"]

    # Use the Aurelius-patched model so reasoning_effort is pinned to a
    # DeepSeek-acceptable value (upstream Concordia passes 'minimal',
    # which DeepSeek rejects with HTTP 400).
    model = make_model(
        model_name=llm_model,
        api_key=api_key,
        api_base=base_url,
    )

    # Create embedder for associative memory
    embedder = _create_embedder(api_key)

    def _make_memory_bank():
        return basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder,
        )

    events = []

    # Build agents with 7-step moral reasoning chain
    built_agents = []
    agent_cot_keys = {}
    for agent_cfg in agents_config:
        agent, cot_keys = _build_agent(
            agent_cfg=agent_cfg,
            model=model,
            memory_bank_factory=_make_memory_bank,
            shared_context=gm_config["shared_context"],
        )
        built_agents.append(agent)
        agent_cot_keys[agent_cfg["name"]] = cot_keys

    agent_names = [a.name for a in built_agents]

    # Feed shared context as initial observation to all agents
    premise_observation = f"{gm_config['shared_context']}\n\nMoral tension: {gm_config['tension_framing']}"
    for agent in built_agents:
        agent.observe(premise_observation)

    # --- Scene loop ---
    for scene_idx, scene in enumerate(scenes_config):
        mode = scene.get("mode", "decision")
        steps = scene.get("steps", 2)

        events.append(
            {
                "type": "scene_start",
                "content": f"Scene {scene_idx + 1}: {mode} mode",
                "scene_index": scene_idx,
                "step_index": 0,
            }
        )

        # GM narration at scene start
        narration = f"Scene {scene_idx + 1} begins. {gm_config['shared_context'][:300]}"
        events.append(
            {
                "type": "narration",
                "content": narration,
                "scene_index": scene_idx,
                "step_index": 0,
            }
        )

        # Feed narration as observation
        for agent in built_agents:
            agent.observe(narration)

        for step in range(steps):
            for agent in built_agents:
                cot_keys = agent_cot_keys[agent.name]

                if mode == "reflection":
                    action_spec = entity_lib.free_action_spec(
                        call_to_action=(
                            f"Reflect on what has happened so far."
                            f" Consider the moral implications of the"
                            f" decisions made. How do you feel about"
                            f" the outcome? What would you do differently?"
                            f" Respond as {agent.name} in first person."
                        ),
                    )
                    action = agent.act(action_spec)
                    cot = _extract_cot(agent, cot_keys)

                    events.append(
                        {
                            "type": "reflection",
                            "agent": agent.name,
                            "content": action,
                            "scene_index": scene_idx,
                            "step_index": step,
                            "chain_of_thought": cot,
                        }
                    )
                else:
                    # Decision mode
                    other_agents = [n for n in agent_names if n != agent.name]
                    action_spec = entity_lib.free_action_spec(
                        call_to_action=(
                            f"Given the ethical dilemma at hand and"
                            f" the perspectives of {', '.join(other_agents)},"
                            f" what does {agent.name} say or do?"
                            f" Respond in first person as {agent.name}."
                        ),
                    )
                    action = agent.act(action_spec)
                    cot = _extract_cot(agent, cot_keys)

                    events.append(
                        {
                            "type": "action",
                            "agent": agent.name,
                            "content": action,
                            "scene_index": scene_idx,
                            "step_index": step,
                            "chain_of_thought": cot,
                        }
                    )

                # Feed action as observation to other agents
                observation = f"{agent.name}: {action}"
                for other in built_agents:
                    if other.name != agent.name:
                        other.observe(observation)

        # Handle forced choice
        fc = scene.get("forced_choice")
        if fc:
            target_agent = next(
                (a for a in built_agents if a.name == fc["agent_name"]),
                None,
            )
            if target_agent:
                # Present forced choice as free-form prompt rather than
                # Concordia's strict choice_action_spec, which requires
                # exact option text in the response.  This works reliably
                # across model sizes.
                choices_text = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(fc["choices"]))
                action_spec = entity_lib.free_action_spec(
                    call_to_action=(
                        f"{fc['call_to_action']}\n\n"
                        f"You must choose one of the following:\n{choices_text}\n\n"
                        f"State your choice and explain your reasoning."
                    ),
                )
                choice = target_agent.act(action_spec)
                cot = _extract_cot(target_agent, agent_cot_keys[fc["agent_name"]])

                events.append(
                    {
                        "type": "forced_choice",
                        "agent": fc["agent_name"],
                        "content": choice,
                        "scene_index": scene_idx,
                        "step_index": steps,
                        "metadata": {"choices": fc["choices"]},
                        "chain_of_thought": cot,
                    }
                )

                # Feed forced choice as observation
                fc_observation = f"{fc['agent_name']} chose: {choice}"
                for agent in built_agents:
                    agent.observe(fc_observation)

        events.append(
            {
                "type": "scene_end",
                "content": f"Scene {scene_idx + 1} concluded",
                "scene_index": scene_idx,
                "step_index": steps,
            }
        )

    return {
        "events": events,
        "metadata": {
            "llm_tokens_consumed": 0,
            "concordia_version": _get_concordia_version(),
            "llm_model": llm_model,
            "execution_path": "concordia",
        },
        "completed": True,
    }


def _get_concordia_version() -> str:
    try:
        from importlib.metadata import version

        return version("gdm-concordia")
    except Exception:
        return "unknown"


def run_simulation(setup_path: str, output_path: str):
    """Run a simulation using the Concordia library.

    Concordia (gdm-concordia) is a hard requirement. If it is not
    installed or fails, the simulation fails loudly — no silent
    fallback to a degraded path.
    """
    start = time.monotonic()

    with open(setup_path) as f:
        setup = json.load(f)

    api_key = _load_api_key()

    try:
        result = run_with_concordia(setup, api_key)
    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(f"Simulation failed: {e}", file=sys.stderr)
        result = {
            "events": [
                {
                    "type": "narration",
                    "content": f"Simulation error: {e}",
                    "scene_index": 0,
                    "step_index": 0,
                }
            ],
            "metadata": {"llm_tokens_consumed": 0, "concordia_version": "", "llm_model": ""},
            "completed": False,
        }

    elapsed = time.monotonic() - start
    result["metadata"]["wall_clock_seconds"] = elapsed
    result["metadata"].setdefault("agent_names", [a["name"] for a in setup.get("agents", [])])

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    event_count = len(result.get("events", []))
    print(f"Simulation complete: {event_count} events in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <setup.json> <output.json>", file=sys.stderr)
        sys.exit(1)
    run_simulation(sys.argv[1], sys.argv[2])
