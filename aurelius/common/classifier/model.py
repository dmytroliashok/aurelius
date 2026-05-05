"""Quality classifier model: load, predict, serialize.

Uses XGBoost native JSON format for serialization (safe, no pickle/joblib).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aurelius.common.classifier.features import extract_features

logger = logging.getLogger(__name__)


@dataclass
class ClassifierResult:
    passed: bool
    confidence: float


class ClassifierModel:
    """Wraps a trained XGBoost model for scenario config quality classification."""

    def __init__(self, model=None, version: str = "0.0.0"):
        self._model = model
        self.version = version

    @classmethod
    def load(cls, path: str | Path) -> ClassifierModel:
        """Load a serialized model from disk (XGBoost native format).

        Verifies SHA-256 hash from the sidecar .meta file if present.
        """
        from xgboost import XGBClassifier

        model_path = Path(path)
        meta_path = Path(str(path) + ".meta")
        version = "0.0.0"
        expected_hash: str | None = None

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            version = meta.get("version", "0.0.0")
            expected_hash = meta.get("sha256")

        # Verify hash before loading
        if expected_hash:
            actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Classifier model hash mismatch: expected {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}... — model may be corrupted or tampered with"
                )

        model = XGBClassifier()
        model.load_model(str(path))
        return cls(model=model, version=version)

    def save(self, path: str | Path) -> None:
        """Save the model to disk (XGBoost native format, safe)."""
        model_path = str(path)
        self._model.save_model(model_path)
        # Compute SHA-256 hash of the model file
        model_bytes = Path(model_path).read_bytes()
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        # Write version + hash to sidecar metadata file
        meta_path = model_path + ".meta"
        Path(meta_path).write_text(json.dumps({"version": self.version, "sha256": model_hash}))
        logger.info("Classifier model saved to %s (version=%s, sha256=%s)", path, self.version, model_hash[:16])

    def predict(self, config: dict, threshold: float = 0.5, embedding_service=None) -> ClassifierResult:
        """Predict PASS/FAIL for a scenario config.

        Args:
            config: Scenario config dict.
            threshold: Confidence threshold for PASS.
            embedding_service: Optional embedding service for cross-field features.

        Returns:
            ClassifierResult with passed flag and confidence score.
        """
        if self._model is None:
            # No model loaded — fail closed to prevent gaming during bootstrap phase
            logger.warning("Classifier has no model loaded — rejecting (fail closed)")
            return ClassifierResult(passed=False, confidence=0.0)

        features = extract_features(config, embedding_service=embedding_service)
        features_2d = features.reshape(1, -1)

        proba = self._model.predict_proba(features_2d)[0]
        # proba[1] = probability of GOOD class
        confidence = float(proba[1])

        return ClassifierResult(passed=confidence >= threshold, confidence=confidence)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def to_bytes(self) -> bytes:
        """Serialize model to bytes for API transfer (safe, no pickle).

        Format: JSON envelope with base64-encoded XGBoost native model + SHA-256 hash.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".ubj")
        try:
            os.close(fd)
            self._model.save_model(tmp_path)
            model_data = Path(tmp_path).read_bytes()
        finally:
            os.unlink(tmp_path)
        model_hash = hashlib.sha256(model_data).hexdigest()
        envelope = {
            "version": self.version,
            "format": "xgboost",
            "sha256": model_hash,
            "model": base64.b64encode(model_data).decode("ascii"),
        }
        return json.dumps(envelope).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> ClassifierModel:
        """Deserialize model from bytes (safe, no pickle).

        Expects JSON envelope with format=xgboost, base64-encoded model,
        and optional sha256 hash for integrity verification.
        """
        envelope = json.loads(data)
        fmt = envelope.get("format")
        if fmt != "xgboost":
            raise ValueError(f"Unsupported model format: {fmt!r} (expected 'xgboost')")
        model_data = base64.b64decode(envelope["model"])

        # Verify hash if present
        expected_hash = envelope.get("sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(model_data).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Classifier model hash mismatch: expected {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}... — model may be corrupted or tampered with"
                )

        fd, tmp_path = tempfile.mkstemp(suffix=".ubj")
        try:
            os.close(fd)
            Path(tmp_path).write_bytes(model_data)
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(tmp_path)
        finally:
            os.unlink(tmp_path)
        return cls(model=model, version=envelope.get("version", "0.0.0"))
