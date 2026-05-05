"""Classifier retraining from confidence-weighted labeled data.

After each benchmark cycle, merges new labeled data with the permanent
seed dataset and retrains the quality classifier with differential
sample weights.
"""

import logging
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from aurelius.benchmark.labeling import LabelingResult, merge_with_seed_data
from aurelius.common.classifier.features import extract_features
from aurelius.common.classifier.model import ClassifierModel

logger = logging.getLogger(__name__)


def retrain_classifier(
    new_labels: LabelingResult,
    new_configs: list[dict],
    submission_ids: list[int],
    seed_data_path: str,
    output_path: str,
    current_version: str = "0.0.0",
    embedding_service=None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    batch_positive: bool = True,
) -> ClassifierModel:
    """Retrain the classifier with confidence-weighted data.

    Merges seed data (permanent HIGH_CONFIDENCE) with new labeled data,
    applies differential sample weights, and trains a new XGBoost model.

    Args:
        new_labels: Confidence labels from the labeling pipeline.
        new_configs: New scenario config dicts.
        submission_ids: Corresponding submission IDs.
        seed_data_path: Path to seed JSONL.
        output_path: Where to save the new model.
        current_version: Current model version (incremented).
        embedding_service: Optional embedding service for features.

    Returns:
        New ClassifierModel with incremented version.
    """
    # Merge datasets
    merged_data, sample_weights = merge_with_seed_data(
        new_labels=new_labels,
        seed_data_path=seed_data_path,
        new_data=new_configs,
        submission_ids=submission_ids,
        batch_positive=batch_positive,
    )

    if len(merged_data) < 10:
        raise ValueError(f"Insufficient training data after merge: {len(merged_data)} < 10")

    # Extract features
    X_list = []  # noqa: N806
    y_list = []
    w_list = []

    for i, entry in enumerate(merged_data):
        config = entry.get("config", entry)
        label = 1 if entry.get("label") == "GOOD" else 0

        features = extract_features(config, embedding_service=embedding_service)
        X_list.append(features)
        y_list.append(label)
        w_list.append(sample_weights[i] if i < len(sample_weights) else 1.0)

    X = np.array(X_list)  # noqa: N806
    y = np.array(y_list)
    w = np.array(w_list)

    # Filter out zero-weight samples
    mask = w > 0
    X, y, w = X[mask], y[mask], w[mask]  # noqa: N806

    logger.info(
        "Retraining classifier: %d samples (%d GOOD, %d BAD), weighted",
        len(y),
        sum(y),
        len(y) - sum(y),
    )

    # Train
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X, y, sample_weight=w)

    # Version bump
    new_version = _bump_version(current_version)

    model = ClassifierModel(model=xgb, version=new_version)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    logger.info("Classifier retrained: version %s → %s, saved to %s", current_version, new_version, output_path)
    return model


def _bump_version(version: str) -> str:
    """Increment the patch version."""
    parts = version.split(".")
    if len(parts) != 3:
        return "1.0.0"
    return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
