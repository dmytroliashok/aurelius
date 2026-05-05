"""Per-UID sliding-window rate limiter for miner submissions."""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_submissions: int, window_seconds: float, persist_path: str | None = None):
        """
        Args:
            max_submissions: Maximum submissions allowed per UID in the window.
            window_seconds: Sliding window duration in seconds.
            persist_path: Optional path to persist rate limit state across restarts.
        """
        self.max_submissions = max_submissions
        self.window_seconds = window_seconds
        self._persist_path = persist_path
        self._timestamps: dict[str, list[float]] = defaultdict(list)
        self._load()

    def check(self, uid_key: str) -> bool:
        """Return True if the UID is within rate limits, False if exceeded."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune old entries
        timestamps = self._timestamps[uid_key]
        self._timestamps[uid_key] = [t for t in timestamps if t > cutoff]

        return len(self._timestamps[uid_key]) < self.max_submissions

    def record(self, uid_key: str) -> None:
        """Record a submission for the UID."""
        self._timestamps[uid_key].append(time.monotonic())
        self._save()

    def update_config(self, max_submissions: int, window_seconds: float) -> None:
        """Update rate limit parameters from remote config."""
        self.max_submissions = max_submissions
        self.window_seconds = window_seconds

    def _save(self) -> None:
        """Persist rate limiter state to disk.

        Monotonic timestamps are converted to wall-clock offsets (seconds ago)
        for portability across restarts where time.monotonic() epoch resets.
        """
        if not self._persist_path:
            return
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            now_mono = time.monotonic()
            now_wall = time.time()
            cutoff = now_mono - self.window_seconds
            data = {
                uid: [now_wall - (now_mono - t) for t in ts if t > cutoff]
                for uid, ts in self._timestamps.items()
                if any(t > cutoff for t in ts)
            }
            path.write_text(json.dumps(data))
        except OSError as e:
            logger.warning("Failed to persist rate limiter state: %s", e)

    def _load(self) -> None:
        """Load persisted rate limiter state.

        Stored wall-clock timestamps are converted back to monotonic offsets.
        T-8: entries with a negative age (clock jumped backwards since last
        save, e.g. NTP correction) or one beyond the window are discarded,
        and the discard is surfaced at WARN so operators see the anomaly
        instead of a silently-permissive rate limiter.
        """
        if not self._persist_path:
            return
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            now_wall = time.time()
            now_mono = time.monotonic()
            discarded_future = 0  # wall_ts > now_wall — impossible without clock jump
            discarded_stale = 0  # age >= window — legitimately expired, not an anomaly
            for uid, ts_list in data.items():
                valid = []
                for wall_ts in ts_list:
                    age = now_wall - wall_ts
                    if age < 0:
                        discarded_future += 1
                        continue
                    if age >= self.window_seconds:
                        discarded_stale += 1
                        continue
                    valid.append(now_mono - age)
                if valid:
                    self._timestamps[uid] = valid
            if discarded_future:
                logger.warning(
                    "Rate limiter: discarded %d persisted entry(ies) with future "
                    "timestamps — system clock moved backwards since last save "
                    "(NTP correction?). Rate-limit budgets are effectively reset "
                    "for affected hotkeys this cycle.",
                    discarded_future,
                )
            logger.info(
                "Loaded rate limiter state: %d UIDs with active windows (discarded %d stale, %d future-dated)",
                len(self._timestamps),
                discarded_stale,
                discarded_future,
            )
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load rate limiter state from %s: %s", self._persist_path, e)
