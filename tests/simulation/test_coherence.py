from aurelius.simulation.coherence import validate_coherence
from aurelius.simulation.transcript import EventType, Transcript, TranscriptEvent, TranscriptMetadata


def _make_transcript(
    completed: bool = True,
    agents: list[str] | None = None,
    n_actions: int = 6,
    empty_ratio: float = 0.0,
    duplicate_ratio: float = 0.0,
    include_forced_choice: bool = True,
) -> Transcript:
    if agents is None:
        agents = ["Dr. Chen", "Nurse Patel"]

    events = [TranscriptEvent(type=EventType.SCENE_START, content="Scene 1", scene_index=0)]

    for i in range(n_actions):
        agent = agents[i % len(agents)]
        if i < n_actions * empty_ratio:
            content = ""
        elif i < n_actions * (empty_ratio + duplicate_ratio):
            content = "Same repeated action"
        else:
            content = f"Unique action by {agent} at step {i}"
        events.append(TranscriptEvent(type=EventType.ACTION, agent=agent, content=content, scene_index=0, step_index=i))

    if include_forced_choice:
        events.append(
            TranscriptEvent(
                type=EventType.FORCED_CHOICE,
                agent=agents[0],
                content="I choose option A",
                scene_index=0,
                step_index=n_actions,
            )
        )

    events.append(TranscriptEvent(type=EventType.SCENE_END, content="Scene 1 ended", scene_index=0))

    return Transcript(
        events=events,
        metadata=TranscriptMetadata(),
        agent_names=sorted(agents),
        scene_count=1,
        completed=completed,
    )


class TestValidateCoherence:
    def test_valid_transcript_passes(self):
        t = _make_transcript()
        result = validate_coherence(t, expected_agents=["Dr. Chen", "Nurse Patel"])
        assert result.passed

    def test_incomplete_simulation_fails(self):
        t = _make_transcript(completed=False)
        result = validate_coherence(t)
        assert not result.passed
        assert any("complete" in r.lower() for r in result.reasons)

    def test_too_few_events_fails(self):
        t = _make_transcript(n_actions=1, include_forced_choice=False)
        result = validate_coherence(t, min_events=10)
        assert not result.passed
        assert any("few events" in r.lower() for r in result.reasons)

    def test_missing_agent_fails(self):
        t = _make_transcript(agents=["Dr. Chen"])  # Only one agent
        result = validate_coherence(t, expected_agents=["Dr. Chen", "Nurse Patel"])
        assert not result.passed
        assert any("Missing" in r for r in result.reasons)

    def test_unexpected_agent_fails(self):
        t = _make_transcript(agents=["Dr. Chen", "Hallucinated Entity"])
        result = validate_coherence(t, expected_agents=["Dr. Chen", "Nurse Patel"])
        assert not result.passed
        assert any("Unexpected" in r for r in result.reasons)
        assert any("Missing" in r for r in result.reasons)

    def test_extra_agent_with_all_expected_fails(self):
        t = _make_transcript(agents=["Dr. Chen", "Nurse Patel", "Ghost"])
        result = validate_coherence(t, expected_agents=["Dr. Chen", "Nurse Patel"])
        assert not result.passed
        assert any("Unexpected" in r and "Ghost" in r for r in result.reasons)

    def test_too_few_agents_without_expected(self):
        t = _make_transcript(agents=["Solo Agent"])
        result = validate_coherence(t)
        assert not result.passed
        assert any("1 agent" in r for r in result.reasons)

    def test_empty_forced_choice_fails(self):
        t = _make_transcript()
        # Make forced choice empty
        for e in t.events:
            if e.type == EventType.FORCED_CHOICE:
                e.content = ""
        result = validate_coherence(t)
        assert not result.passed
        assert any("Empty forced choice" in r for r in result.reasons)

    def test_excessive_empty_responses_fails(self):
        t = _make_transcript(n_actions=10, empty_ratio=0.5)
        result = validate_coherence(t)
        assert not result.passed
        assert any("empty" in r.lower() for r in result.reasons)

    def test_excessive_repetition_fails(self):
        t = _make_transcript(n_actions=10, duplicate_ratio=0.8)
        result = validate_coherence(t)
        assert not result.passed
        assert any("repetition" in r.lower() for r in result.reasons)

    def test_no_forced_choice_still_passes(self):
        t = _make_transcript(include_forced_choice=False)
        result = validate_coherence(t)
        assert result.passed

    def test_valid_cot_passes(self):
        t = _make_transcript()
        cot = [
            {"step": "situation_perception", "response": "A tense situation"},
            {"step": "self_perception", "response": "I am a doctor"},
            {"step": "theory_of_mind", "response": "Others are worried"},
        ]
        for e in t.events:
            if e.type == EventType.ACTION:
                e.chain_of_thought = cot
        result = validate_coherence(t)
        assert result.passed

    def test_mostly_empty_cot_fails(self):
        t = _make_transcript()
        cot = [
            {"step": "situation_perception", "response": ""},
            {"step": "self_perception", "response": ""},
            {"step": "theory_of_mind", "response": "Valid response"},
            {"step": "self_interest", "response": ""},
        ]
        for e in t.events:
            if e.type == EventType.ACTION:
                e.chain_of_thought = cot
        result = validate_coherence(t)
        assert not result.passed
        assert any("CoT" in r for r in result.reasons)

    def test_no_cot_still_passes(self):
        """Events without chain_of_thought (legacy format) should pass."""
        t = _make_transcript()
        result = validate_coherence(t)
        assert result.passed
