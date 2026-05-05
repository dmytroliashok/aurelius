"""Tests that LocalSubmissionQueue resolves limits at call time, not import time.

Regression protection: previously `MAX_QUEUE_SIZE`, `MAX_QUEUE_FILE_SIZE_MB`,
and `DEFAULT_MAX_AGE_SECONDS` were module-level globals captured from Config
at import. They are now resolved per-instance from remote_config or Config.
"""

from types import SimpleNamespace

from aurelius.validator.local_queue import LocalSubmissionQueue, QueuedSubmission


def _make_submission(work_id: str = "w1") -> QueuedSubmission:
    return QueuedSubmission(work_id=work_id, miner_hotkey="hk", scenario_config={"name": "x"})


def _fake_remote(**overrides):
    base = {"queue_max_size": 10, "queue_max_file_size_mb": 1, "queue_max_age_seconds": 60}
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRemoteConfigWiring:
    def test_queue_reads_max_size_from_remote_config(self):
        rc = _fake_remote(queue_max_size=3)
        queue = LocalSubmissionQueue(remote_config=rc)
        for i in range(5):
            queue.enqueue(_make_submission(f"w{i}"))
        # Oldest dropped when over max_size
        assert queue.size == 3

    def test_queue_reads_max_age_seconds_from_remote_config(self):
        rc = _fake_remote(queue_max_age_seconds=120)
        queue = LocalSubmissionQueue(remote_config=rc)
        assert queue._max_age_seconds == 120

    def test_queue_reads_max_file_size_from_remote_config(self):
        rc = _fake_remote(queue_max_file_size_mb=42)
        queue = LocalSubmissionQueue(remote_config=rc)
        assert queue._max_file_size_mb == 42

    def test_explicit_kwargs_override_remote_config(self):
        rc = _fake_remote(queue_max_size=100, queue_max_age_seconds=60)
        queue = LocalSubmissionQueue(remote_config=rc, max_size=5, max_age_seconds=999)
        assert queue._queue.maxlen == 5
        assert queue._max_age_seconds == 999


class TestNoModuleLevelCapture:
    def test_module_has_no_max_queue_globals(self):
        """Old `MAX_QUEUE_SIZE`, `MAX_QUEUE_FILE_SIZE_MB`, `DEFAULT_MAX_AGE_SECONDS`
        module globals must be gone."""
        from aurelius.validator import local_queue

        assert not hasattr(local_queue, "MAX_QUEUE_SIZE")
        assert not hasattr(local_queue, "MAX_QUEUE_FILE_SIZE_MB")
        assert not hasattr(local_queue, "DEFAULT_MAX_AGE_SECONDS")

    def test_different_queues_can_have_different_limits(self):
        q1 = LocalSubmissionQueue(remote_config=_fake_remote(queue_max_size=5))
        q2 = LocalSubmissionQueue(remote_config=_fake_remote(queue_max_size=50))
        assert q1._queue.maxlen == 5
        assert q2._queue.maxlen == 50


class TestFallbackWhenNoRemoteConfig:
    def test_queue_without_remote_config_falls_back_to_config(self):
        from aurelius.config import Config

        queue = LocalSubmissionQueue()
        assert queue._queue.maxlen == Config.QUEUE_MAX_SIZE
        assert queue._max_age_seconds == Config.QUEUE_MAX_AGE_SECONDS
        assert queue._max_file_size_mb == Config.QUEUE_MAX_FILE_SIZE_MB
