"""Classifier training pipeline.

Loads labeled JSONL, extracts features, trains XGBoost binary classifier.
"""

import argparse
import json
import logging

import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from aurelius.common.classifier.features import extract_features
from aurelius.common.classifier.model import ClassifierModel

logger = logging.getLogger(__name__)


def load_labeled_data(path: str, embedding_service=None) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled JSONL and extract feature vectors.

    Args:
        path: Path to JSONL file with {"config": {...}, "label": "GOOD"|"BAD"} entries.
        embedding_service: Optional embedding service for semantic features.

    Returns:
        (features_matrix, labels_array) where labels are 1=GOOD, 0=BAD.
    """
    features_list = []
    labels = []

    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            config = entry["config"]
            label = 1 if entry["label"] == "GOOD" else 0

            feat = extract_features(config, embedding_service=embedding_service)
            features_list.append(feat)
            labels.append(label)

    return np.array(features_list), np.array(labels)


def train_classifier(
    data_path: str,
    output_path: str = "classifier_model.joblib",
    version: str = "1.0.0",
    embedding_service=None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> ClassifierModel:
    """Train an XGBoost classifier on labeled data.

    Args:
        data_path: Path to labeled JSONL.
        output_path: Path to save trained model.
        version: Model version string.
        embedding_service: Optional embedding service.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.

    Returns:
        Trained ClassifierModel.
    """
    logger.info("Loading labeled data from %s", data_path)
    X, y = load_labeled_data(data_path, embedding_service=embedding_service)  # noqa: N806
    logger.info("Loaded %d samples (%d GOOD, %d BAD)", len(y), sum(y), len(y) - sum(y))

    if len(y) < 10:
        raise ValueError(f"Need at least 10 labeled samples, got {len(y)}")

    # Train XGBoost
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=42,
    )

    # Cross-validation
    n_folds = min(5, len(y) // 2)
    if n_folds >= 2:
        scores = cross_val_score(xgb, X, y, cv=n_folds, scoring="accuracy")
        logger.info("Cross-validation accuracy: %.3f ± %.3f (%d folds)", scores.mean(), scores.std(), n_folds)

    # Final fit on all data
    xgb.fit(X, y)

    model = ClassifierModel(model=xgb, version=version)
    model.save(output_path)
    logger.info("Model saved to %s", output_path)

    return model


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(prog="aurelius-train-classifier", description="Train quality classifier")
    parser.add_argument("data", help="Path to labeled JSONL file")
    parser.add_argument("--output", default="classifier_model.joblib", help="Output model path")
    parser.add_argument("--version", default="1.0.0", help="Model version string")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    train_classifier(
        data_path=args.data,
        output_path=args.output,
        version=args.version,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
