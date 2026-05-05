import json

import pytest

from aurelius.validator.local_queue import LocalSubmissionQueue, QueuedSubmission


def _make_submission(work_id: str = "work_1") -> QueuedSubmission:
    return QueuedSubmission(
        work_id=work_id,
        miner_hotkey="miner_abc",
        scenario_config={"name": "test"},
    )


class TestLocalSubmissionQueue:
    def test_enqueue_and_drain(self):
        queue = LocalSubmissionQueue()
        queue.enqueue(_make_submission("w1"))
        queue.enqueue(_make_submission("w2"))
        assert queue.size == 2

        drained = queue.drain(max_count=1)
        assert len(drained) == 1
        assert drained[0].work_id == "w1"
        assert queue.size == 1

    def test_drain_empty(self):
        queue = LocalSubmissionQueue()
        assert queue.drain() == []
        assert queue.is_empty

    def test_max_size(self):
        queue = LocalSubmissionQueue(max_size=3)
        for i in range(5):
            queue.enqueue(_make_submission(f"w{i}"))
        assert queue.size == 3  # Oldest dropped

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "queue.jsonl")

        # Enqueue and persist
        q1 = LocalSubmissionQueue(persist_path=path)
        q1.enqueue(_make_submission("w1"))
        q1.enqueue(_make_submission("w2"))
        assert q1.size == 2

        # Load from disk
        q2 = LocalSubmissionQueue(persist_path=path)
        assert q2.size == 2

        drained = q2.drain()
        assert len(drained) == 2
        assert drained[0].work_id == "w1"

    def test_drain_all(self):
        queue = LocalSubmissionQueue()
        for i in range(10):
            queue.enqueue(_make_submission(f"w{i}"))

        drained = queue.drain(max_count=50)
        assert len(drained) == 10
        assert queue.is_empty

    def test_no_persist_path(self):
        queue = LocalSubmissionQueue(persist_path=None)
        queue.enqueue(_make_submission())


class TestT7SchemaVersioning:
    """T-7: the local queue now writes a schema_version header so a future
    validator binary can detect and migrate (or refuse) older or newer
    file formats, rather than crashing on TypeError during dataclass
    construction.
    """

    def test_saved_file_carries_schema_version_header(self, tmp_path):
        from aurelius.validator.local_queue import SCHEMA_VERSION

        path = str(tmp_path / "queue.jsonl")
        queue = LocalSubmissionQueue(persist_path=path)
        queue.enqueue(_make_submission("w1"))

        with open(path) as f:
            first_line = f.readline()
        header = json.loads(first_line)
        assert header == {"schema_version": SCHEMA_VERSION}

    def test_legacy_file_without_header_still_loads(self, tmp_path, caplog):
        """A queue file written by an older validator has no header —
        every line is a QueuedSubmission. We accept and warn."""
        import logging
        import time as _time

        path = tmp_path / "queue.jsonl"
        legacy_entry = {
            "work_id": "legacy_w1",
            "miner_hotkey": "miner",
            "scenario_config": {"name": "legacy"},
            "classifier_score": None,
            "simulation_transcript": None,
            "queued_at": _time.time() - 60,  # recent, passes the drain age filter
        }
        with open(path, "w") as f:
            f.write(json.dumps(legacy_entry) + "\n")

        with caplog.at_level(logging.WARNING, logger="aurelius.validator.local_queue"):
            queue = LocalSubmissionQueue(persist_path=str(path))

        assert queue.size == 1
        drained = queue.drain()
        assert drained[0].work_id == "legacy_w1"
        assert any("no schema_version header" in r.message for r in caplog.records)

    def test_future_dated_schema_refuses_to_load(self, tmp_path, caplog):
        """If a file claims a version higher than we know, we must refuse
        to load it. Loading partial data would drop entries the newer
        binary knows are mandatory; deferring lets the operator roll
        back or migrate explicitly."""
        import logging

        from aurelius.validator.local_queue import SCHEMA_VERSION

        path = tmp_path / "queue.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"schema_version": SCHEMA_VERSION + 10}) + "\n")
            f.write(json.dumps({"work_id": "future_w", "miner_hotkey": "m", "scenario_config": {}}) + "\n")

        with caplog.at_level(logging.ERROR, logger="aurelius.validator.local_queue"):
            queue = LocalSubmissionQueue(persist_path=str(path))

        assert queue.size == 0
        assert any("exceeds supported" in r.message for r in caplog.records)

    def test_unknown_fields_ignored_on_load(self, tmp_path):
        """Forward compat: a future schema that adds an optional field
        should load cleanly on this version with the extra field dropped."""
        import time as _time

        from aurelius.validator.local_queue import SCHEMA_VERSION

        path = tmp_path / "queue.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"schema_version": SCHEMA_VERSION}) + "\n")
            f.write(
                json.dumps(
                    {
                        "work_id": "w_forward",
                        "miner_hotkey": "m",
                        "scenario_config": {"ok": True},
                        "classifier_score": 0.9,
                        "simulation_transcript": None,
                        "queued_at": _time.time() - 30,
                        "some_future_field": "hello",
                        "another_new_thing": [1, 2, 3],
                    }
                )
                + "\n"
            )

        queue = LocalSubmissionQueue(persist_path=str(path))
        assert queue.size == 1
        assert queue.drain()[0].work_id == "w_forward"

    def test_missing_required_fields_skipped_not_fatal(self, tmp_path, caplog):
        """If an entry is missing a field we now require (broken upgrade,
        manual edit, partial disk write), skip that entry instead of
        failing the whole load."""
        import logging

        from aurelius.validator.local_queue import SCHEMA_VERSION

        path = tmp_path / "queue.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"schema_version": SCHEMA_VERSION}) + "\n")
            f.write(json.dumps({"scenario_config": {}}) + "\n")  # missing work_id, miner_hotkey
            f.write(
                json.dumps({"work_id": "good", "miner_hotkey": "m", "scenario_config": {}})
                + "\n"
            )

        with caplog.at_level(logging.WARNING, logger="aurelius.validator.local_queue"):
            queue = LocalSubmissionQueue(persist_path=str(path))

        assert queue.size == 1
        assert queue.drain()[0].work_id == "good"
        assert any("unparseable" in r.message.lower() or "skipped" in r.message.lower()
                   for r in caplog.records)
