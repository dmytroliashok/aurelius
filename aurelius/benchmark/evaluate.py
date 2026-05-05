"""MoReBench evaluation pipeline.

Evaluates a fine-tuned model against the MoReBench public benchmark
(500 scenarios, 23,000+ rubric criteria, 5 dimensions).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from aurelius.benchmark.config import EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score for a single MoReBench dimension."""

    name: str
    score: float  # 0.0–1.0
    criteria_met: int
    criteria_total: int


@dataclass
class BenchmarkResult:
    """Results from a MoReBench evaluation run."""

    overall_score: float  # 0.0–1.0
    dimensions: list[DimensionScore] = field(default_factory=list)
    scenarios_evaluated: int = 0
    model_path: str = ""
    base_model: str = ""
    delta: float | None = None  # Improvement over baseline, if available


def load_morebench_scenarios(path: str) -> list[dict]:
    """Load MoReBench public evaluation scenarios.

    Expected format: list of {"scenario": ..., "rubric": [...], "dimension": ...}
    """
    if not Path(path).exists():
        logger.warning("MoReBench data not found at %s, returning empty", path)
        return []

    with open(path) as f:
        return json.load(f)


def evaluate_on_morebench(
    model_path: str,
    config: EvalConfig | None = None,
    baseline_score: float | None = None,
) -> BenchmarkResult:
    """Evaluate a fine-tuned model against MoReBench.

    This runs the model on MoReBench scenarios and uses an LLM judge
    to score criteria fulfillment.

    Args:
        model_path: Path to fine-tuned model/adapter.
        config: Evaluation configuration.
        baseline_score: Previous overall score for delta computation.

    Returns:
        BenchmarkResult with per-dimension and overall scores.
    """
    if config is None:
        config = EvalConfig()

    logger.info("Starting MoReBench evaluation: model=%s", model_path)

    scenarios = load_morebench_scenarios(config.morebench_path)
    if not scenarios:
        logger.warning("No MoReBench scenarios loaded. Returning empty result.")
        return BenchmarkResult(
            overall_score=0.0,
            scenarios_evaluated=0,
            model_path=model_path,
        )

    # Limit to configured number
    scenarios = scenarios[: config.num_scenarios]

    # Load model for generation
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model_name = _detect_base_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
    except Exception as e:
        logger.error("Failed to load model for evaluation: %s", e)
        return BenchmarkResult(overall_score=0.0, model_path=model_path)

    # Evaluate each scenario
    dimension_scores: dict[str, list[float]] = {}
    total_criteria_met = 0
    total_criteria = 0

    for scenario in scenarios:
        dimension = scenario.get("dimension", "general")
        rubric = scenario.get("rubric", [])
        prompt = scenario.get("scenario", "")

        # Generate response
        response = _generate_response(model, tokenizer, prompt)

        # Judge response against rubric
        criteria_met = _judge_response(response, rubric, config)

        score = criteria_met / max(len(rubric), 1)
        dimension_scores.setdefault(dimension, []).append(score)
        total_criteria_met += criteria_met
        total_criteria += len(rubric)

    # Aggregate
    dimensions = []
    for dim_name, scores in dimension_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        dimensions.append(
            DimensionScore(
                name=dim_name,
                score=avg,
                criteria_met=int(avg * len(scores)),
                criteria_total=len(scores),
            )
        )

    overall = total_criteria_met / max(total_criteria, 1)
    delta = (overall - baseline_score) if baseline_score is not None else None

    result = BenchmarkResult(
        overall_score=overall,
        dimensions=dimensions,
        scenarios_evaluated=len(scenarios),
        model_path=model_path,
        delta=delta,
    )

    logger.info(
        "MoReBench evaluation complete: overall=%.3f, scenarios=%d, delta=%s",
        overall,
        len(scenarios),
        f"{delta:+.3f}" if delta is not None else "N/A",
    )

    return result


def _detect_base_model(adapter_path: str) -> str:
    """Detect the base model from a LoRA adapter's config."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", "")
    return ""


def _generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the fine-tuned model."""
    import torch

    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return response.strip()


def _judge_response(response: str, rubric: list[dict], config: EvalConfig) -> int:
    """Use an LLM judge to score how many rubric criteria the response meets.

    Uses the configured LLM judge (config.judge_model)
    to evaluate each criterion. Falls back to keyword heuristic only if the
    LLM provider is unavailable.

    Returns the number of criteria met.
    """
    if not rubric:
        return 0

    # Try LLM judge first
    try:
        return _judge_response_llm(response, rubric, config)
    except Exception as e:
        logger.warning("LLM judge unavailable (%s), falling back to keyword heuristic", e)
        return _judge_response_heuristic(response, rubric)


def _judge_response_llm(response: str, rubric: list[dict], config: EvalConfig) -> int:
    """LLM-based rubric evaluation."""
    import asyncio

    from aurelius.common.llm import create_llm

    provider = create_llm(model=config.judge_model)

    criteria_texts = []
    for criterion in rubric:
        criteria_texts.append(criterion.get("text", criterion) if isinstance(criterion, dict) else str(criterion))

    # Batch all criteria into one prompt to minimize API calls
    criteria_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria_texts))
    prompt = (
        f"You are evaluating a response against specific rubric criteria.\n\n"
        f"RESPONSE:\n{response[:2000]}\n\n"
        f"CRITERIA:\n{criteria_block}\n\n"
        f"For each criterion, respond with its number followed by YES or NO.\n"
        f"Example: 1. YES\n2. NO\n3. YES\n"
        f"Evaluate each criterion independently."
    )

    result = asyncio.get_event_loop().run_until_complete(
        provider.complete(prompt, system="You are a precise rubric evaluator. Answer YES or NO for each criterion.")
    )

    # Parse YES/NO responses
    criteria_met = 0
    for line in result.strip().split("\n"):
        line = line.strip().upper()
        if "YES" in line:
            criteria_met += 1

    return criteria_met


def _judge_response_heuristic(response: str, rubric: list[dict]) -> int:
    """Keyword-based heuristic fallback (unreliable — use LLM judge in production)."""
    logger.warning("Using keyword heuristic for rubric evaluation — results will be unreliable")
    criteria_met = 0
    for criterion in rubric:
        criterion_text = criterion.get("text", criterion) if isinstance(criterion, dict) else str(criterion)
        keywords = criterion_text.lower().split()[:3]
        if any(kw in response.lower() for kw in keywords if len(kw) > 3):
            criteria_met += 1
    return criteria_met
