"""Assertion tests: Concordia Simulation (CS-01..CS-07)."""

from aurelius.simulation.coherence import CoherenceResult, validate_coherence
from aurelius.simulation.transcript import EventType, Transcript, TranscriptEvent, TranscriptMetadata


def _make_transcript(completed=True, events=None, agent_names=None):
    if events is None:
        events = [
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="I act.", scene_index=0, step_index=0, metadata={}),
            TranscriptEvent(type=EventType.ACTION, agent="Bob", content="I respond.", scene_index=0, step_index=1, metadata={}),
            TranscriptEvent(type=EventType.FORCED_CHOICE, agent="Alice", content="I choose A.", scene_index=0, step_index=2, metadata={"choices": ["A", "B"]}),
            TranscriptEvent(type=EventType.REFLECTION, agent="Bob", content="I reflect.", scene_index=1, step_index=0, metadata={}),
            TranscriptEvent(type=EventType.REFLECTION, agent="Alice", content="I also reflect.", scene_index=1, step_index=1, metadata={}),
        ]
    return Transcript(
        events=events,
        metadata=TranscriptMetadata(
            llm_tokens_consumed=100,
            wall_clock_seconds=10.0,
            concordia_version="v2.0.0",
            docker_image_tag="v2.0.0",
            container_id="abc123",
            llm_model="test-model",
        ),
        agent_names=agent_names or ["Alice", "Bob"],
        scene_count=2,
        completed=completed,
    )


class TestCS04CoherenceFalseOnIncomplete:
    def test_cs04_incomplete_simulation_fails(self):
        """Coherence must be false if simulation did not complete."""
        transcript = _make_transcript(completed=False)
        result = validate_coherence(transcript)
        assert not result.passed
        assert any("complete" in r.lower() for r in result.reasons)

    def test_cs04_empty_forced_choice_fails(self):
        """Coherence must be false if forced choice has empty content."""
        events = [
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="I act.", scene_index=0, step_index=0, metadata={}),
            TranscriptEvent(type=EventType.ACTION, agent="Bob", content="I act.", scene_index=0, step_index=1, metadata={}),
            TranscriptEvent(type=EventType.FORCED_CHOICE, agent="Alice", content="", scene_index=0, step_index=2, metadata={}),
            TranscriptEvent(type=EventType.REFLECTION, agent="Alice", content="reflect.", scene_index=1, step_index=0, metadata={}),
            TranscriptEvent(type=EventType.REFLECTION, agent="Bob", content="reflect.", scene_index=1, step_index=1, metadata={}),
        ]
        transcript = _make_transcript(events=events)
        result = validate_coherence(transcript)
        assert not result.passed
        assert any("empty forced choice" in r.lower() for r in result.reasons)

    def test_cs04_complete_with_forced_choice_passes(self):
        """Complete simulation with non-empty forced choice passes."""
        transcript = _make_transcript()
        result = validate_coherence(transcript)
        assert result.passed

    def test_cs04_too_few_events_fails(self):
        """Coherence requires minimum events."""
        events = [
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="act.", scene_index=0, step_index=0, metadata={}),
        ]
        transcript = _make_transcript(events=events, agent_names=["Alice"])
        result = validate_coherence(transcript, min_events=5)
        assert not result.passed

    def test_cs04_missing_agent_fails(self):
        """Coherence fails if expected agents didn't participate."""
        events = [
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="act.", scene_index=0, step_index=0, metadata={}),
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="act2.", scene_index=0, step_index=1, metadata={}),
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="act3.", scene_index=0, step_index=2, metadata={}),
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="act4.", scene_index=0, step_index=3, metadata={}),
            TranscriptEvent(type=EventType.ACTION, agent="Alice", content="act5.", scene_index=0, step_index=4, metadata={}),
        ]
        transcript = _make_transcript(events=events, agent_names=["Alice", "Bob"])
        result = validate_coherence(transcript, expected_agents=["Alice", "Bob"])
        assert not result.passed
        assert any("missing" in r.lower() for r in result.reasons)


class TestCS05ResourceLimits:
    def test_cs05_compute_limits_returns_expected_fields(self):
        """Docker resource limits must be enforced by the container runtime."""
        from aurelius.simulation.docker_runner import DockerSimulationRunner

        runner = DockerSimulationRunner.__new__(DockerSimulationRunner)
        runner.base_timeout = 600
        runner.base_ram_mb = 4096
        runner.cpu_count = 2
        limits = runner._compute_limits(agent_count=2)
        assert "mem_limit" in limits
        assert "nano_cpus" in limits
        assert "timeout" in limits
        assert limits["timeout"] > 0
        assert limits["mem_limit"] > 0

    def test_cs05_limits_scale_with_agents(self):
        """Resource limits should scale with agent count."""
        from aurelius.simulation.docker_runner import DockerSimulationRunner

        runner = DockerSimulationRunner.__new__(DockerSimulationRunner)
        runner.base_timeout = 600
        runner.base_ram_mb = 4096
        runner.cpu_count = 2
        limits_2 = runner._compute_limits(agent_count=2)
        limits_4 = runner._compute_limits(agent_count=4)
        assert limits_4["timeout"] > limits_2["timeout"]
        assert limits_4["mem_limit"] > limits_2["mem_limit"]
