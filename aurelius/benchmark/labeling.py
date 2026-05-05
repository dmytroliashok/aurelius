"""Confidence-weighted labeling for classifier training.

Three-tier taxonomy (from spec Section 10.1.2):
- HIGH_CONFIDENCE: batch delta and influence score agree (both positive or both negative)
- LOW_CONFIDENCE: batch-level label only; influence unavailable, ambiguous, or batch < 30
- EXCLUDED: batch delta and influence score contradict

The hand-labeled seed dataset is permanently HIGH_CONFIDENCE and never relabeled.
"""

import logging
from dataclasses import dataclass

from aurelius.benchmark.influence import InfluenceScores

logger = logging.getLogger(__name__)


LABEL_HIGH = "high_confidence"
LABEL_LOW = "low_confidence"
LABEL_EXCLUDED = "excluded"

# Sample weights for classifier training
WEIGHT_HIGH = 4.0
WEIGHT_LOW = 1.0
WEIGHT_EXCLUDED = 0.0  # Excluded from training


@dataclass
class LabelingResult:
    """Results from confidence-weighted labeling."""

    labels: dict[int, str]  # submission_id → label
    weights: dict[int, float]  # submission_id → sample weight
    counts: dict[str, int]  # label → count


def assign_confidence_labels(
    influence: InfluenceScores,
    min_batch_size: int = 30,
) -> LabelingResult:
    """Assign confidence labels based on influence scores and batch delta.

    Args:
        influence: InfluenceScores from influence computation.
        min_batch_size: Minimum batch size for HIGH_CONFIDENCE labels.

    Returns:
        LabelingResult with labels, weights, and counts.
    """
    labels = {}
    weights = {}
    counts = {LABEL_HIGH: 0, LABEL_LOW: 0, LABEL_EXCLUDED: 0}

    batch_delta = influence.batch_delta
    batch_positive = batch_delta > 0

    # If batch is too small or no delta info, everything is LOW_CONFIDENCE
    if len(influence.scores) < min_batch_size or influence.method in ("uniform", "uniform_fallback"):
        for sid in influence.scores:
            labels[sid] = LABEL_LOW
            weights[sid] = WEIGHT_LOW
        counts[LABEL_LOW] = len(influence.scores)
        logger.info("All %d submissions labeled LOW_CONFIDENCE (batch size or uniform method)", len(influence.scores))
        return LabelingResult(labels=labels, weights=weights, counts=counts)

    for sid, score in influence.scores.items():
        score_positive = score > 0

        if batch_positive == score_positive:
            # Batch and influence agree
            labels[sid] = LABEL_HIGH
            weights[sid] = WEIGHT_HIGH
            counts[LABEL_HIGH] += 1
        elif abs(score) < 1e-6:
            # Ambiguous (near-zero influence)
            labels[sid] = LABEL_LOW
            weights[sid] = WEIGHT_LOW
            counts[LABEL_LOW] += 1
        else:
            # Batch and influence contradict
            labels[sid] = LABEL_EXCLUDED
            weights[sid] = WEIGHT_EXCLUDED
            counts[LABEL_EXCLUDED] += 1

    logger.info(
        "Labeling complete: HIGH=%d, LOW=%d, EXCLUDED=%d",
        counts[LABEL_HIGH],
        counts[LABEL_LOW],
        counts[LABEL_EXCLUDED],
    )
    return LabelingResult(labels=labels, weights=weights, counts=counts)


def merge_with_seed_data(
    new_labels: LabelingResult,
    seed_data_path: str,
    new_data: list[dict],
    submission_ids: list[int],
    batch_positive: bool = True,
) -> tuple[list[dict], list[float]]:
    """Merge new labeled data with the permanent seed dataset for classifier retraining.

    Seed data is always HIGH_CONFIDENCE. New EXCLUDED data is dropped.

    Args:
        new_labels: Labels from assign_confidence_labels().
        seed_data_path: Path to seed JSONL (permanent HIGH_CONFIDENCE).
        new_data: New config dicts to merge.
        submission_ids: Corresponding submission IDs.
        batch_positive: Whether the batch delta was positive (True=GOOD batch).

    Returns:
        (configs_with_labels, sample_weights) ready for training.
    """
    import json

    configs = []
    sample_weights = []

    # Load seed data (permanent HIGH_CONFIDENCE)
    try:
        with open(seed_data_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    configs.append(entry)
                    sample_weights.append(WEIGHT_HIGH)
                except json.JSONDecodeError:
                    logger.warning("Malformed JSONL at %s line %d, skipping", seed_data_path, line_num)
        logger.info("Loaded %d seed entries (permanent HIGH_CONFIDENCE)", len(configs))
    except FileNotFoundError:
        logger.warning("Seed data not found at %s", seed_data_path)

    # Add new data (excluding EXCLUDED)
    added = 0
    for i, sid in enumerate(submission_ids):
        label = new_labels.labels.get(sid, LABEL_LOW)
        weight = new_labels.weights.get(sid, WEIGHT_LOW)

        if label == LABEL_EXCLUDED:
            continue

        if i < len(new_data):
            # Training label derived from batch delta direction:
            # positive batch delta means this batch improved the model → GOOD
            training_label = "GOOD" if batch_positive else "BAD"
            configs.append({"config": new_data[i], "label": training_label})
            sample_weights.append(weight)
            added += 1

    logger.info("Merged dataset: %d seed + %d new = %d total", len(configs) - added, added, len(configs))
    return configs, sample_weights
