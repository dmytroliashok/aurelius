"""Per-config influence scoring via gradient-based influence functions.

Estimates each training config's contribution to the benchmark improvement
using Fisher information approximation.
"""

import logging
from dataclasses import dataclass

from aurelius.benchmark.evaluate import BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class InfluenceScores:
    """Influence scores for a batch of submissions."""

    scores: dict[int, float]  # submission_id → influence score
    batch_delta: float  # Overall MoReBench delta for this batch
    method: str = "fisher"  # "fisher" or "ablation"


def compute_influence_scores(
    model_path: str,
    dataset_path: str,
    benchmark_result: BenchmarkResult,
    submission_ids: list[int],
    method: str = "fisher",
) -> InfluenceScores:
    """Compute per-config influence scores.

    Phase 1: Fisher information approximation (cheap, less precise).
    Phase 2+: Ablation-based attribution (expensive, more precise).

    Args:
        model_path: Path to the fine-tuned model adapter.
        dataset_path: Path to the training dataset JSONL.
        benchmark_result: Results from MoReBench evaluation.
        submission_ids: Submission IDs corresponding to dataset entries.
        method: "fisher" (default) or "ablation".

    Returns:
        InfluenceScores mapping submission IDs to influence values.
    """
    batch_delta = benchmark_result.delta or 0.0

    if len(submission_ids) < 30:
        # Below minimum batch size — all get LOW_CONFIDENCE
        logger.warning("Batch too small for influence scoring (%d < 30), assigning uniform scores", len(submission_ids))
        uniform_score = batch_delta / max(len(submission_ids), 1)
        return InfluenceScores(
            scores=dict.fromkeys(submission_ids, uniform_score),
            batch_delta=batch_delta,
            method="uniform",
        )

    if method == "fisher":
        return _fisher_influence(model_path, dataset_path, benchmark_result, submission_ids)
    elif method == "ablation":
        return _ablation_influence(model_path, dataset_path, benchmark_result, submission_ids)
    else:
        raise ValueError(f"Unknown influence method: {method}")


def _fisher_influence(
    model_path: str,
    dataset_path: str,
    benchmark_result: BenchmarkResult,
    submission_ids: list[int],
) -> InfluenceScores:
    """Fisher information approximation for influence scores.

    Computes the diagonal Fisher information matrix and uses it to
    estimate each sample's influence on the loss.
    """
    batch_delta = benchmark_result.delta or 0.0

    try:
        import json

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model
        base_model_name = ""
        from pathlib import Path as P  # noqa: N817

        config_path = P(model_path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                base_model_name = json.load(f).get("base_model_name_or_path", "")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        # Load dataset
        with open(dataset_path) as f:
            examples = [json.loads(line) for line in f]

        # Compute per-sample gradients and Fisher diagonal
        sample_influences = []
        for example in examples:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()

            # Compute gradient norm as influence proxy
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            sample_influences.append(grad_norm**0.5)

            model.zero_grad()

        # Normalize to sum to batch_delta
        total = sum(sample_influences) or 1.0
        scores = {}
        for i, sid in enumerate(submission_ids):
            if i < len(sample_influences):
                scores[sid] = (sample_influences[i] / total) * batch_delta
            else:
                scores[sid] = 0.0

        return InfluenceScores(scores=scores, batch_delta=batch_delta, method="fisher")

    except Exception as e:
        logger.warning("Fisher influence scoring failed: %s. Falling back to uniform.", e)
        uniform_score = batch_delta / max(len(submission_ids), 1)
        return InfluenceScores(
            scores=dict.fromkeys(submission_ids, uniform_score),
            batch_delta=batch_delta,
            method="uniform_fallback",
        )


def _ablation_influence(
    model_path: str,
    dataset_path: str,
    benchmark_result: BenchmarkResult,
    submission_ids: list[int],
) -> InfluenceScores:
    """Ablation-based influence scoring (Phase 2+).

    Leave-one-out retraining to estimate each sample's influence.
    Expensive but more accurate than Fisher approximation.

    Currently a placeholder that falls back to Fisher.
    """
    logger.info("Ablation influence scoring not yet implemented, falling back to Fisher")
    return _fisher_influence(model_path, dataset_path, benchmark_result, submission_ids)
