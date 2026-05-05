#!/usr/bin/env python3
"""Evaluate a scenario config with a validator-like pipeline.

This is an offline preflight tool for miners/operators. It mirrors validator
stage ordering and weight computation as closely as possible without requiring
live Central API state or full simulation infrastructure.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import secrets
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Allow running from source checkout without `pip install -e .`.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aurelius.common.version import PROTOCOL_VERSION
from aurelius.common.types import ConsumeResult
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.validator.pipeline import ValidationPipeline
from aurelius.validator.rate_limiter import RateLimiter
from aurelius.validator.remote_config import RemoteConfig


class _MockAPIClient:
    """Minimal async client interface expected by ValidationPipeline."""

    def __init__(
        self,
        *,
        has_balance: bool,
        novelty_is_novel: bool,
        novelty_message: str,
        classifier_passed: bool,
        classifier_confidence: float,
        classifier_version: str,
        consume_success: bool,
        consume_message: str,
    ) -> None:
        self._has_balance = has_balance
        self._novelty_is_novel = novelty_is_novel
        self._novelty_message = novelty_message
        self._classifier_passed = classifier_passed
        self._classifier_confidence = classifier_confidence
        self._classifier_version = classifier_version
        self._consume_success = consume_success
        self._consume_message = consume_message

    async def check_balance(self, miner_hotkey: str) -> bool:
        _ = miner_hotkey
        return self._has_balance

    async def check_novelty(
        self,
        pooled_embedding: list[float],
        threshold: float,
        field_embeddings: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        _ = pooled_embedding, threshold, field_embeddings
        return {"novel": self._novelty_is_novel, "message": self._novelty_message, "similarity": 0.42}

    async def classify_config(self, config: dict, threshold: float) -> dict[str, Any]:
        _ = config, threshold
        return {
            "passed": self._classifier_passed,
            "confidence": self._classifier_confidence,
            "version": self._classifier_version,
        }

    async def add_to_novelty_index(self, pooled_embedding: list[float], config_hash: str | None = None) -> None:
        _ = pooled_embedding, config_hash
        return None

    async def remove_from_novelty_index(self, config_hash: str) -> None:
        _ = config_hash
        return None

    async def consume_work_token(
        self,
        miner_hotkey: str,
        work_id: str,
        config_hash: str = "",
        work_id_signature: str = "",
    ) -> ConsumeResult:
        _ = miner_hotkey, work_id, config_hash, work_id_signature
        return ConsumeResult(
            success=self._consume_success,
            deducted=self._consume_success,
            valid=self._consume_success,
            message=self._consume_message,
        )


class _MockEmbeddingService:
    def embed_config(self, config: dict) -> "_Vector":
        # Stable pseudo-embedding from config hash to avoid extra deps.
        h = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).digest()
        vals = [((b / 255.0) * 2.0 - 1.0) for b in h[:32]]
        return _Vector(vals)

    def extract_field_embeddings(self, config: dict, parsed_config=None) -> dict[str, list[float]]:
        _ = parsed_config
        premise = str(config.get("premise", ""))
        val = (len(premise) % 100) / 100.0
        return {"premise": [val] * 8}


class _Vector:
    def __init__(self, vals: list[float]) -> None:
        self._vals = vals

    def tolist(self) -> list[float]:
        return list(self._vals)


class _MockSimulationRunner:
    def __init__(self, should_pass: bool, error_message: str) -> None:
        self._should_pass = should_pass
        self._error_message = error_message

    def run_simulation(self, config: dict):
        _ = config
        if not self._should_pass:
            return SimpleNamespace(
                success=False,
                error=self._error_message,
                coherence=None,
                transcript=None,
                wall_clock_seconds=0.2,
            )
        transcript = SimpleNamespace(
            events=[{"event": "decision"}],
            model_dump=lambda: {"events": [{"event": "decision"}], "summary": "mock transcript"},
        )
        return SimpleNamespace(
            success=True,
            error="",
            coherence=SimpleNamespace(passed=True, reasons=[]),
            transcript=transcript,
            wall_clock_seconds=0.2,
        )


def _build_synapse(config: dict, miner_hotkey: str, miner_protocol_version: str) -> ScenarioConfigSynapse:
    time_ns = str(time.time_ns())
    nonce = secrets.token_hex(16)
    payload = json.dumps(config, sort_keys=True) + miner_hotkey + time_ns + nonce
    work_id = hashlib.sha256(payload.encode()).hexdigest()

    syn = ScenarioConfigSynapse()
    syn.scenario_config = config
    syn.work_id = work_id
    syn.work_id_nonce = nonce
    syn.work_id_time_ns = time_ns
    syn.work_id_signature = ""  # optional for local preflight
    syn.miner_version = "local-preflight"
    syn.miner_protocol_version = miner_protocol_version
    return syn


async def _run(args) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rc = RemoteConfig()
    rc._config["classifier_threshold"] = args.classifier_threshold
    rc._config["novelty_threshold"] = args.novelty_threshold
    rc._config["work_id_freshness_seconds"] = args.work_id_freshness_seconds
    rc._config["max_agents"] = args.max_agents
    rc._config["custom_archetype_threshold_bump"] = args.custom_archetype_threshold_bump

    api = _MockAPIClient(
        has_balance=args.has_balance,
        novelty_is_novel=args.novelty_is_novel,
        novelty_message=args.novelty_message,
        classifier_passed=args.classifier_passed,
        classifier_confidence=args.classifier_confidence,
        classifier_version=args.classifier_version,
        consume_success=args.consume_success,
        consume_message=args.consume_message,
    )

    pipeline = ValidationPipeline(
        api_client=api,
        remote_config=rc,
        rate_limiter=RateLimiter(max_submissions=args.rate_limit_max, window_seconds=args.rate_limit_window_s),
        embedding_service=_MockEmbeddingService(),
        simulation_runner=_MockSimulationRunner(args.simulate_pass, args.simulate_error),
        llm_provider=None,  # optional stages disabled for offline tool
    )

    syn = _build_synapse(config, args.miner_hotkey, args.miner_protocol_version)
    result = await pipeline.run(syn, args.miner_hotkey, anchor_ns=time.time_ns())

    print(f"passed={result.passed}")
    print(f"weight={result.weight:.6f}")
    print(f"classifier_score={result.classifier_score}")
    print("stages:")
    for idx, st in enumerate(result.stages, start=1):
        status = "PASS" if st.passed else "FAIL"
        print(f"  {idx:02d}. {st.stage:<20} {status}  reason={st.reason}")
    if args.json_output:
        print(
            json.dumps(
                {
                    "passed": result.passed,
                    "weight": result.weight,
                    "classifier_score": result.classifier_score,
                    "stages": [{"stage": s.stage, "passed": s.passed, "reason": s.reason} for s in result.stages],
                },
                indent=2,
            )
        )
    return 0 if result.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validator-like pipeline preflight on a scenario config JSON.")
    parser.add_argument("--config", required=True, help="Path to scenario_config JSON file")
    parser.add_argument("--miner-hotkey", default="mock_miner_hotkey_ss58")
    parser.add_argument("--miner-protocol-version", default=PROTOCOL_VERSION)
    parser.add_argument("--max-agents", type=int, default=2)
    parser.add_argument("--work-id-freshness-seconds", type=int, default=300)

    # Stage behavior controls (offline approximation).
    parser.add_argument("--has-balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--novelty-is-novel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--novelty-message", default="OK")
    parser.add_argument("--classifier-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-confidence", type=float, default=0.88)
    parser.add_argument("--classifier-version", default="mock-1.0.0")
    parser.add_argument("--simulate-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulate-error", default="Simulation failed")
    parser.add_argument("--consume-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--consume-message", default="Consumed")

    # Tunables mirrored from remote config.
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--custom-archetype-threshold-bump", type=float, default=0.1)
    parser.add_argument("--novelty-threshold", type=float, default=0.88)

    # Local rate-limiter settings for this process.
    parser.add_argument("--rate-limit-max", type=int, default=100)
    parser.add_argument("--rate-limit-window-s", type=int, default=4320)
    parser.add_argument("--json-output", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
