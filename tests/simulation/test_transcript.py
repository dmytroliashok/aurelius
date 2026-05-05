from aurelius.simulation.transcript import EventType, Transcript, TranscriptEvent, extract_transcript


def _sample_raw_output(completed: bool = True, n_events: int = 10) -> dict:
    events = []
    agents = ["Dr. Chen", "Nurse Patel"]
    for i in range(n_events):
        if i == 0:
            events.append({"type": "scene_start", "content": "Scene 1", "scene_index": 0, "step_index": 0})
        elif i == n_events - 1:
            events.append({"type": "scene_end", "content": "Scene 1 ended", "scene_index": 0, "step_index": i})
        elif i == n_events - 2:
            events.append({
                "type": "forced_choice",
                "agent": "Dr. Chen",
                "content": "I choose to administer the drug.",
                "scene_index": 0,
                "step_index": i,
                "metadata": {"choices": ["Administer", "Follow protocol"]},
            })
        else:
            events.append({
                "type": "action",
                "agent": agents[i % 2],
                "content": f"Agent action at step {i}",
                "scene_index": 0,
                "step_index": i,
            })

    return {
        "events": events,
        "metadata": {
            "llm_tokens_consumed": 1500,
            "wall_clock_seconds": 45.2,
            "concordia_version": "2.0.0",
            "llm_model": "gpt-4o-mini",
        },
        "completed": completed,
    }


class TestExtractTranscript:
    def test_basic_extraction(self):
        raw = _sample_raw_output()
        t = extract_transcript(raw)
        assert isinstance(t, Transcript)
        assert t.completed is True
        assert len(t.events) == 10

    def test_agent_names_extracted(self):
        t = extract_transcript(_sample_raw_output())
        assert "Dr. Chen" in t.agent_names
        assert "Nurse Patel" in t.agent_names

    def test_metadata_extracted(self):
        t = extract_transcript(_sample_raw_output())
        assert t.metadata.llm_tokens_consumed == 1500
        assert t.metadata.wall_clock_seconds == 45.2
        assert t.metadata.concordia_version == "2.0.0"

    def test_event_types(self):
        t = extract_transcript(_sample_raw_output())
        types = {e.type for e in t.events}
        assert EventType.SCENE_START in types
        assert EventType.ACTION in types
        assert EventType.FORCED_CHOICE in types
        assert EventType.SCENE_END in types

    def test_scene_count(self):
        t = extract_transcript(_sample_raw_output())
        assert t.scene_count == 1

    def test_incomplete_simulation(self):
        t = extract_transcript(_sample_raw_output(completed=False))
        assert t.completed is False

    def test_empty_output(self):
        t = extract_transcript({"events": [], "metadata": {}, "completed": False})
        assert len(t.events) == 0
        assert t.scene_count == 0

    def test_unknown_event_type_defaults_to_narration(self):
        raw = {"events": [{"type": "unknown_type", "content": "test"}], "metadata": {}, "completed": True}
        t = extract_transcript(raw)
        assert t.events[0].type == EventType.NARRATION

    def test_chain_of_thought_extracted(self):
        cot = [
            {"step": "situation_perception", "response": "A tense situation"},
            {"step": "self_perception", "response": "I am a doctor"},
            {"step": "theory_of_mind", "response": "Others are worried"},
            {"step": "self_interest", "response": "I want to help"},
            {"step": "other_interest", "response": "They need care"},
            {"step": "neutral_observer", "response": "Both sides have merit"},
        ]
        raw = {
            "events": [
                {
                    "type": "action",
                    "agent": "Dr. Chen",
                    "content": "I decide to help.",
                    "scene_index": 0,
                    "step_index": 0,
                    "chain_of_thought": cot,
                    "gm_resolution": "Dr. Chen helps the patient.",
                }
            ],
            "metadata": {},
            "completed": True,
        }
        t = extract_transcript(raw)
        assert len(t.events[0].chain_of_thought) == 6
        assert t.events[0].chain_of_thought[0]["step"] == "situation_perception"
        assert t.events[0].gm_resolution == "Dr. Chen helps the patient."

    def test_missing_cot_defaults_to_empty(self):
        raw = {
            "events": [{"type": "action", "agent": "Dr. Chen", "content": "test"}],
            "metadata": {},
            "completed": True,
        }
        t = extract_transcript(raw)
        assert t.events[0].chain_of_thought == []
        assert t.events[0].gm_resolution is None
