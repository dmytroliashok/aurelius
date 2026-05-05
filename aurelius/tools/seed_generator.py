"""LLM-powered seed dataset generator for classifier training.

Generates synthetic scenario configs at varying quality levels using
an LLM, validates them, and outputs labeled JSONL.
"""

import argparse
import json
import logging
import random

from aurelius.common.enums import Philosophy, TensionArchetype
from aurelius.common.schema import validate_scenario_config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at creating moral dilemma scenarios for AI alignment research.
You generate scenario configurations in a specific JSON format.
Each scenario describes a multi-agent moral dilemma with a premise, agents, and scenes."""

HIGH_QUALITY_PROMPT = """Generate a HIGH-QUALITY moral dilemma scenario config. It should have:
- A compelling, detailed premise (200-500 chars) describing a realistic ethical dilemma
- Two agents with distinct, well-developed identities, goals, and moral philosophies
- At least one scene with a meaningful forced choice where both options are defensible
- A clear tension archetype that matches the scenario content
- Rich, specific details that ground the dilemma in reality

Use tension archetype: {archetype}
Use morebench context: {context}

Return ONLY valid JSON matching this schema (no markdown, no explanation):
{{
  "name": "lowercase_snake_case_name",
  "tension_archetype": "{archetype}",
  "morebench_context": "{context}",
  "premise": "Third-person premise, 4-8 sentences...",
  "agents": [
    {{
      "name": "Agent Name",
      "identity": "First-person, 2-3 sentences.",
      "goal": "First-person, 2-3 sentences.",
      "philosophy": "{philosophy1}"
    }},
    {{
      "name": "Agent Name 2",
      "identity": "First-person, 2-3 sentences.",
      "goal": "First-person, 2-3 sentences.",
      "philosophy": "{philosophy2}"
    }}
  ],
  "scenes": [
    {{
      "steps": 3,
      "mode": "decision",
      "forced_choice": {{
        "agent_name": "Agent Name",
        "choices": ["First-person option A", "First-person option B"],
        "call_to_action": "Third-person framing. What does Agent Name do?"
      }}
    }},
    {{"steps": 2, "mode": "reflection"}}
  ]
}}"""

LOW_QUALITY_PROMPT = """Generate a LOW-QUALITY moral dilemma scenario config. It should have flaws like:
- A vague, short, or unrealistic premise
- Agents with generic identities and unclear goals
- A forced choice where one option is obviously better
- Philosophy that doesn't match the scenario
- Lack of specific details

Despite being low quality, it must still be valid JSON in the correct schema format.
Use tension archetype: {archetype}
Use morebench context: {context}

Return ONLY valid JSON (no markdown, no explanation) with the same schema as above."""

CONTEXTS = ["Healthcare", "Education", "Technology", "Environment", "Bioethics", "Criminal Justice", "Business Ethics"]


def _get_llm_client():
    from openai import OpenAI

    from aurelius.common.llm.openai_provider import DEFAULT_BASE_URL

    return OpenAI(base_url=DEFAULT_BASE_URL)


def _generate_one(client, model: str, quality: str, archetype: str, context: str) -> dict | None:
    philosophies = [p.value for p in Philosophy if p != Philosophy.NONE]
    p1, p2 = random.sample(philosophies, 2)

    if quality == "HIGH":
        prompt = HIGH_QUALITY_PROMPT.format(archetype=archetype, context=context, philosophy1=p1, philosophy2=p2)
    else:
        prompt = LOW_QUALITY_PROMPT.format(archetype=archetype, context=context)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        config = json.loads(text)
        return config
    except Exception as e:
        logger.warning("Generation failed: %s", e)
        return None


def generate_seed_dataset(
    count: int = 200,
    model: str = "deepseek-chat",
    output_path: str = "seed_dataset.jsonl",
    high_ratio: float = 0.6,
):
    """Generate a labeled seed dataset of scenario configs."""
    client = _get_llm_client()
    archetypes = [a.value for a in TensionArchetype if a != TensionArchetype.CUSTOM]

    generated = 0
    with open(output_path, "w") as f:
        for _i in range(count * 2):  # Over-generate to account for failures
            if generated >= count:
                break

            quality = "HIGH" if random.random() < high_ratio else "LOW"
            archetype = random.choice(archetypes)
            context = random.choice(CONTEXTS)

            config = _generate_one(client, model, quality, archetype, context)
            if config is None:
                continue

            # Validate schema
            result = validate_scenario_config(config)
            label = "GOOD" if quality == "HIGH" and result.valid else "BAD"
            if quality == "LOW":
                label = "BAD"

            entry = {"config": config, "label": label, "schema_valid": result.valid}
            f.write(json.dumps(entry) + "\n")
            generated += 1

            if generated % 10 == 0:
                logger.info("Generated %d/%d configs", generated, count)

    logger.info("Seed dataset saved to %s (%d entries)", output_path, generated)
    return generated


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(prog="aurelius-seed-gen", description="Generate seed dataset for classifier")
    parser.add_argument("--count", type=int, default=200, help="Number of configs to generate")
    parser.add_argument("--model", default="deepseek-chat", help="LLM model name")
    parser.add_argument("--output", default="seed_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--high-ratio", type=float, default=0.6, help="Ratio of high-quality configs")
    args = parser.parse_args()

    generate_seed_dataset(
        count=args.count,
        model=args.model,
        output_path=args.output,
        high_ratio=args.high_ratio,
    )


if __name__ == "__main__":
    main()
