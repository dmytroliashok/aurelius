#!/usr/bin/env python3
"""Manual runner for the Aurelius MoReBench benchmark pipeline.

Orchestrates the flow that POST /benchmark/trigger?full=true would run if the
Central API had a worker. Invoke when an operator wants to score an
accumulated batch of SIMULATED submissions.

Flow:
  1. POST /benchmark/trigger?full=false              -> claim a batch
  2. SELECT transcripts from Postgres                -> materials
  3. prepare_dataset + save_dataset                  -> JSONL
  4. finetune()                                      -> LoRA adapter
  5. evaluate_on_morebench()                         -> score + delta
  6. compute_influence_scores + labels               -> per-submission attribution
  7. retrain_classifier() (optional)                 -> new .xgb artifact
  8. POST /benchmark/result                          -> record batch outcome
  9. UPDATE submission rows (optional, --write-influence)

Rescue mode:
  --reset-stuck-batch BATCH_ID                       -> revert stuck BENCHMARKING rows to SIMULATED

Prerequisites:
  - v3 installed with [ml,benchmark] extras + asyncpg
  - ADMIN_API_KEY          (env or --admin-key)
  - AURELIUS_API_URL       (env or --api-url)
  - AURELIUS_DB_URL        (env or --db-url)  — Postgres URL for the Central API's DB
  - data/morebench_public.json reachable (or --morebench-path)
  - GPU with ~16 GB VRAM for Llama 3.1 8B + LoRA

Examples:
  # Dry-run: claim a batch, then stop (useful for auth smoke test)
  python scripts/run_benchmark_pipeline.py --dry-run

  # Full run, posting aggregate result but skipping classifier retrain
  python scripts/run_benchmark_pipeline.py --skip-retrain

  # Full run with per-submission writeback
  python scripts/run_benchmark_pipeline.py --write-influence --current-version 1.2.3

  # Recover from a crashed run:
  python scripts/run_benchmark_pipeline.py --reset-stuck-batch 4f9a...
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import sys
from pathlib import Path

import httpx

logger = logging.getLogger("benchmark-worker")


def _auth_headers(admin_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_key}"}


async def trigger_batch(client: httpx.AsyncClient, api_url: str, admin_key: str) -> dict:
    r = await client.post(
        f"{api_url}/benchmark/trigger",
        params={"full": "false"},
        headers=_auth_headers(admin_key),
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


async def fetch_baseline_score(client: httpx.AsyncClient, api_url: str, admin_key: str) -> float | None:
    r = await client.get(
        f"{api_url}/benchmark/history",
        params={"limit": 1},
        headers=_auth_headers(admin_key),
        timeout=30,
    )
    r.raise_for_status()
    hist = r.json()
    if not hist:
        return None
    return hist[0].get("morebench_score")


async def post_benchmark_result(
    client: httpx.AsyncClient,
    api_url: str,
    admin_key: str,
    *,
    batch_id: str,
    score: float,
    delta: float | None,
    base_model: str,
    submission_count: int,
) -> dict:
    r = await client.post(
        f"{api_url}/benchmark/result",
        headers=_auth_headers(admin_key),
        json={
            "batch_id": batch_id,
            "morebench_score": score,
            "morebench_delta": delta,
            "base_model": base_model,
            "submission_count": submission_count,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


async def fetch_batch_rows(db_url: str, batch_id: str) -> tuple[list[int], list[dict], list[dict]]:
    """Returns aligned lists: (submission_ids, configs, transcripts)."""
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT id, scenario_config, simulation_transcript
            FROM submission
            WHERE batch_id = $1 AND status = 'benchmarking'
            ORDER BY id
            """,
            batch_id,
        )
    finally:
        await conn.close()

    submission_ids: list[int] = []
    configs: list[dict] = []
    transcripts: list[dict] = []
    for row in rows:
        submission_ids.append(int(row["id"]))
        cfg = row["scenario_config"]
        if isinstance(cfg, str):
            cfg = json.loads(cfg)
        configs.append(cfg or {})
        tr = row["simulation_transcript"]
        if isinstance(tr, str):
            tr = json.loads(tr)
        transcripts.append(tr or {})
    return submission_ids, configs, transcripts


async def write_influence_and_complete(
    db_url: str,
    batch_id: str,
    influence_scores: dict[int, float],
    labels: dict[int, str],
) -> dict[str, int]:
    """Mirror update_influence_scores + mark_batch_complete from benchmark_service.py.

    - Test-set rows (is_test_set=TRUE) are never relabeled.
    - EXCLUDED rows get no influence_score/confidence_label written.
    - All non-test rows in the batch transition to status='benchmarked'.
    """
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        test_rows = await conn.fetch(
            "SELECT id FROM submission WHERE batch_id = $1 AND is_test_set = TRUE",
            batch_id,
        )
        test_set_ids = {int(r["id"]) for r in test_rows}

        written = 0
        skipped_test = 0
        skipped_excluded = 0
        async with conn.transaction():
            for sid, score in influence_scores.items():
                if sid in test_set_ids:
                    skipped_test += 1
                    continue
                label = labels.get(sid)
                if label == "excluded":
                    skipped_excluded += 1
                    continue
                await conn.execute(
                    "UPDATE submission SET influence_score = $1, confidence_label = $2 WHERE id = $3",
                    float(score),
                    label,
                    sid,
                )
                written += 1

            await conn.execute(
                "UPDATE submission SET status = 'benchmarked' "
                "WHERE batch_id = $1 AND is_test_set = FALSE",
                batch_id,
            )
    finally:
        await conn.close()

    return {"written": written, "skipped_test_set": skipped_test, "skipped_excluded": skipped_excluded}


async def reset_stuck_batch(db_url: str, batch_id: str) -> int:
    """Revert a stuck BENCHMARKING batch back to SIMULATED for a fresh run."""
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        result = await conn.execute(
            "UPDATE submission SET status = 'simulated' "
            "WHERE batch_id = $1 AND status = 'benchmarking'",
            batch_id,
        )
    finally:
        await conn.close()
    parts = result.split()
    return int(parts[-1]) if parts and parts[-1].isdigit() else 0


async def run_pipeline(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient() as client:
        logger.info("Requesting batch from %s", args.api_url)
        trigger = await trigger_batch(client, args.api_url, args.admin_key)
        if trigger.get("status") == "no_batch":
            logger.info("No batch available: %s", trigger.get("message"))
            return 2
        batch_id = trigger["batch_id"]
        claimed_count = int(trigger["count"])
        logger.info("Claimed batch %s (%d submissions)", batch_id, claimed_count)

        if args.dry_run:
            logger.info("--dry-run set, exiting after trigger")
            return 0

        from aurelius.benchmark.config import EvalConfig, FinetuneConfig
        from aurelius.benchmark.evaluate import evaluate_on_morebench
        from aurelius.benchmark.finetune import finetune, prepare_dataset, save_dataset
        from aurelius.benchmark.influence import compute_influence_scores
        from aurelius.benchmark.labeling import assign_confidence_labels

        logger.info("Fetching batch rows from database")
        submission_ids, configs, transcripts = await fetch_batch_rows(args.db_url, batch_id)
        if not submission_ids:
            logger.error(
                "Batch %s has 0 rows in DB. Roll back with: --reset-stuck-batch %s",
                batch_id,
                batch_id,
            )
            return 1
        if len(submission_ids) != claimed_count:
            logger.warning("Claimed %d, fetched %d rows from DB", claimed_count, len(submission_ids))

        batch_dir = Path(args.output_dir) / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Preparing dataset from %d transcripts", len(transcripts))
        examples = prepare_dataset(transcripts, min_rubric_score=None)
        if not examples:
            logger.error("prepare_dataset produced 0 examples (no usable transcripts).")
            return 1
        dataset_path = str(batch_dir / "dataset.jsonl")
        save_dataset(examples, dataset_path)

        ft_cfg = FinetuneConfig(output_dir=str(batch_dir / "adapter"))
        logger.info("Fine-tuning %s -> %s", ft_cfg.base_model, ft_cfg.output_dir)
        adapter_path = await asyncio.to_thread(finetune, dataset_path, ft_cfg)

        baseline = await fetch_baseline_score(client, args.api_url, args.admin_key)
        logger.info(
            "Evaluating on MoReBench (baseline=%s)",
            f"{baseline:.3f}" if baseline is not None else "none",
        )
        eval_cfg = EvalConfig(morebench_path=args.morebench_path)
        # to_thread: evaluate uses asyncio.run_until_complete internally for the LLM judge,
        # which clashes with our running loop unless we run it in a fresh thread.
        result = await asyncio.to_thread(evaluate_on_morebench, adapter_path, eval_cfg, baseline)
        logger.info(
            "MoReBench result: overall=%.3f delta=%s scenarios=%d",
            result.overall_score,
            f"{result.delta:+.3f}" if result.delta is not None else "n/a",
            result.scenarios_evaluated,
        )

        summary = {
            "batch_id": batch_id,
            "overall_score": result.overall_score,
            "delta": result.delta,
            "scenarios_evaluated": result.scenarios_evaluated,
            "base_model": ft_cfg.base_model,
            "submission_count": len(submission_ids),
            "dimensions": [
                {"name": d.name, "score": d.score, "criteria_total": d.criteria_total}
                for d in result.dimensions
            ],
        }
        (batch_dir / "result.json").write_text(json.dumps(summary, indent=2))

        logger.info("Computing influence scores (method=fisher)")
        influence = await asyncio.to_thread(
            compute_influence_scores, adapter_path, dataset_path, result, submission_ids
        )
        labels = assign_confidence_labels(influence)
        logger.info("Label counts: %s", labels.counts)

        model_path: str | None = None
        if not args.skip_retrain:
            from aurelius.benchmark.retrain import retrain_classifier

            model_path = str(batch_dir / "classifier.xgb")
            logger.info("Retraining classifier -> %s", model_path)
            retrain = functools.partial(
                retrain_classifier,
                new_labels=labels,
                new_configs=configs,
                submission_ids=submission_ids,
                seed_data_path=args.seed_data_path,
                output_path=model_path,
                current_version=args.current_version,
                batch_positive=(result.delta or 0.0) > 0,
            )
            await asyncio.to_thread(retrain)
        else:
            logger.info("Skipping classifier retrain (--skip-retrain)")

        logger.info("Posting aggregate result to %s", args.api_url)
        await post_benchmark_result(
            client,
            args.api_url,
            args.admin_key,
            batch_id=batch_id,
            score=result.overall_score,
            delta=result.delta,
            base_model=ft_cfg.base_model,
            submission_count=len(submission_ids),
        )

        if args.write_influence:
            logger.info("Writing per-submission influence + labels to DB")
            stats = await write_influence_and_complete(
                args.db_url, batch_id, influence.scores, labels.labels
            )
            logger.info(
                "Wrote %d rows; skipped %d test-set, %d excluded; batch marked BENCHMARKED",
                stats["written"],
                stats["skipped_test_set"],
                stats["skipped_excluded"],
            )
        else:
            logger.info(
                "Skipping per-submission write-back (--write-influence not set). "
                "Submissions remain in BENCHMARKING."
            )

        logger.info("Artifacts: %s", batch_dir)
        if model_path:
            logger.info("New classifier: %s (+.meta). Deploy manually.", model_path)

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manual runner for the Aurelius MoReBench benchmark pipeline."
    )
    p.add_argument("--api-url", default=os.environ.get("AURELIUS_API_URL"),
                   help="Central API base URL (env: AURELIUS_API_URL)")
    p.add_argument("--admin-key", default=os.environ.get("ADMIN_API_KEY"),
                   help="Admin bearer token (env: ADMIN_API_KEY)")
    p.add_argument("--db-url", default=os.environ.get("AURELIUS_DB_URL"),
                   help="Postgres URL for submission table (env: AURELIUS_DB_URL)")
    p.add_argument("--morebench-path", default="data/morebench_public.json",
                   help="Path to MoReBench JSON (default: %(default)s)")
    p.add_argument("--seed-data-path", default="data/seed.jsonl",
                   help="Path to seed JSONL for classifier retrain (default: %(default)s)")
    p.add_argument("--output-dir", default="output/batches",
                   help="Base dir for per-batch artifacts (default: %(default)s)")
    p.add_argument("--current-version", default="0.0.0",
                   help="Current classifier version; patch auto-bumped (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true",
                   help="Trigger then exit (auth smoke test)")
    p.add_argument("--skip-retrain", action="store_true",
                   help="Run eval + post result, skip classifier retrain")
    p.add_argument("--write-influence", action="store_true",
                   help="UPDATE submission influence_score/confidence_label and mark batch BENCHMARKED")
    p.add_argument("--reset-stuck-batch", metavar="BATCH_ID",
                   help="Revert a stuck BENCHMARKING batch to SIMULATED, then exit")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = p.parse_args()

    missing: list[str] = []
    if not args.admin_key and not args.reset_stuck_batch:
        missing.append("--admin-key / ADMIN_API_KEY")
    if not args.api_url and not args.reset_stuck_batch:
        missing.append("--api-url / AURELIUS_API_URL")
    if not args.db_url:
        missing.append("--db-url / AURELIUS_DB_URL")
    if missing:
        p.error("missing required config: " + ", ".join(missing))
    return args


async def _main_async(args: argparse.Namespace) -> int:
    if args.reset_stuck_batch:
        n = await reset_stuck_batch(args.db_url, args.reset_stuck_batch)
        logger.info("Reset %d rows in batch %s to SIMULATED", n, args.reset_stuck_batch)
        return 0
    return await run_pipeline(args)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    except httpx.HTTPError as e:
        logger.error("HTTP error: %s", e)
        return 1
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
