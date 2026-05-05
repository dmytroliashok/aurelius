"""Local submission queue for degraded mode.

When the Central API is unreachable, accepted configs are queued locally
and reported when the API returns.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

from aurelius.config import Config

if TYPE_CHECKING:
    from aurelius.validator.remote_config import RemoteConfig

logger = logging.getLogger(__name__)

# T-7: on-disk queue format. Bump this when QueuedSubmission gains a
# mandatory field. The header lives on the first JSONL line as
# ``{"schema_version": N}`` so a new binary can detect legacy files
# (written before versioning landed) and future binaries can refuse to
# load a forward-dated file instead of silently dropping its entries.
SCHEMA_VERSION = 1


@dataclass
class QueuedSubmission:
    work_id: str
    miner_hotkey: str
    scenario_config: dict
    classifier_score: float | None = None
    simulation_transcript: dict | None = None
    queued_at: float = field(default_factory=time.time)


class LocalSubmissionQueue:
    """In-memory queue with optional disk persistence for crash recovery."""

    def __init__(
        self,
        persist_path: str | None = None,
        max_size: int | None = None,
        max_age_seconds: float | None = None,
        remote_config: "RemoteConfig | None" = None,
        max_file_size_mb: int | None = None,
    ):
        def _cfg(remote_attr: str, config_attr: str):
            if remote_config is not None:
                return getattr(remote_config, remote_attr)
            return getattr(Config, config_attr)

        resolved_max_size = max_size if max_size is not None else _cfg("queue_max_size", "QUEUE_MAX_SIZE")
        self._max_age_seconds: float = (
            max_age_seconds if max_age_seconds is not None else _cfg("queue_max_age_seconds", "QUEUE_MAX_AGE_SECONDS")
        )
        self._max_file_size_mb: int = (
            max_file_size_mb
            if max_file_size_mb is not None
            else _cfg("queue_max_file_size_mb", "QUEUE_MAX_FILE_SIZE_MB")
        )
        self._queue: deque[QueuedSubmission] = deque(maxlen=resolved_max_size)
        self._persist_path = persist_path
        self._load()

    def enqueue(self, submission: QueuedSubmission) -> None:
        """Add a submission to the queue."""
        self._queue.append(submission)
        self._save()
        logger.debug("Queued submission %s (%d in queue)", submission.work_id[:16], len(self._queue))

    def drain(self, max_count: int = 50) -> list[QueuedSubmission]:
        """Remove and return up to max_count submissions from the queue.

        Submissions older than max_age_seconds are discarded (not returned).
        This prevents reporting stale submissions that may reference expired
        work IDs or outdated classifier models.
        """
        results = []
        discarded = 0
        now = time.time()
        for _ in range(min(max_count + (self._queue.maxlen or 0), len(self._queue))):
            if not self._queue:
                break
            sub = self._queue.popleft()
            age = now - sub.queued_at
            if age > self._max_age_seconds:
                discarded += 1
                continue
            results.append(sub)
            if len(results) >= max_count:
                break
        if discarded:
            logger.info("Discarded %d stale queued submissions (age > %ds)", discarded, self._max_age_seconds)
        if results or discarded:
            self._save()
        return results

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                # T-7: version header on first line so the reader can
                # distinguish legacy files (no header) from current and
                # future-dated ones without guessing.
                f.write(json.dumps({"schema_version": SCHEMA_VERSION}) + "\n")
                for sub in self._queue:
                    entry = {
                        "work_id": sub.work_id,
                        "miner_hotkey": sub.miner_hotkey,
                        "scenario_config": sub.scenario_config,
                        "classifier_score": sub.classifier_score,
                        "simulation_transcript": sub.simulation_transcript,
                        "queued_at": sub.queued_at,
                    }
                    f.write(json.dumps(entry) + "\n")
            # Guard against unbounded disk usage
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self._max_file_size_mb:
                logger.warning(
                    "Queue file %.1fMB exceeds %dMB limit — truncating to newest entries",
                    file_size_mb,
                    self._max_file_size_mb,
                )
                # Keep only the newest half of entries
                half = len(self._queue) // 2
                discarded_ids = []
                while len(self._queue) > half:
                    discarded = self._queue.popleft()
                    discarded_ids.append(discarded.work_id[:16])
                logger.warning("Truncated %d queued submissions: %s", len(discarded_ids), discarded_ids[:5])
                self._save()
        except OSError as e:
            logger.warning("Failed to persist submission queue: %s", e)

    def _load(self) -> None:
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self._max_file_size_mb:
                logger.warning(
                    "Queue file %.1fMB exceeds %dMB limit — loading only tail entries",
                    file_size_mb,
                    self._max_file_size_mb,
                )
            with open(path) as f:
                lines = f.readlines()
            # If oversized, keep only the newest entries that fit within maxlen
            if file_size_mb > self._max_file_size_mb:
                lines = lines[-self._queue.maxlen :] if self._queue.maxlen else lines
            if not lines:
                return

            # T-7: detect the schema_version header. A valid header is a
            # single-field dict {"schema_version": N}. Anything else is a
            # legacy file written before versioning landed and is treated
            # as v0 — readable, but noisy warning so operators know.
            entry_lines = lines
            try:
                first = json.loads(lines[0])
                if isinstance(first, dict) and "schema_version" in first and len(first) == 1:
                    version = int(first["schema_version"])
                    entry_lines = lines[1:]
                    if version > SCHEMA_VERSION:
                        logger.error(
                            "Queue file schema v%d exceeds supported v%d; "
                            "refusing to load to preserve data. Downgrade the "
                            "validator or migrate the queue file.",
                            version,
                            SCHEMA_VERSION,
                        )
                        return
                else:
                    logger.warning(
                        "Queue file %s has no schema_version header; treating as legacy v0. Next save will upgrade it.",
                        self._persist_path,
                    )
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning(
                    "Queue file %s header unparseable; treating as legacy v0.",
                    self._persist_path,
                )

            known_fields = {f.name for f in fields(QueuedSubmission)}
            skipped = 0
            for line in entry_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not isinstance(entry, dict):
                    skipped += 1
                    continue
                # Drop unknown keys so a newer-schema file written by a
                # later validator binary loads cleanly here (forward compat).
                filtered = {k: v for k, v in entry.items() if k in known_fields}
                try:
                    self._queue.append(QueuedSubmission(**filtered))
                except TypeError:
                    # Missing a required field — e.g. a future mandatory
                    # column was added. Skip loudly instead of silently.
                    skipped += 1
            if skipped:
                logger.warning(
                    "Skipped %d unparseable queue entries from %s",
                    skipped,
                    self._persist_path,
                )
            logger.info("Loaded %d queued submissions from %s", len(self._queue), self._persist_path)
        except OSError as e:
            logger.warning("Failed to load submission queue from %s: %s", self._persist_path, e)
