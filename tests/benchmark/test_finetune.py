import json

import pytest

from aurelius.benchmark.finetune import prepare_dataset, save_dataset


def _sample_transcript(agent_names=None, n_actions=4, include_fc=True) -> dict:
    if agent_names is None:
        agent_names = ["Dr. Chen", "Nurse Patel"]

    events = [{"type": "scene_start", "content": "Scene 1 begins", "scene_index": 0, "step_index": 0}]

    for i in range(n_actions):
        agent = agent_names[i % len(agent_names)]
        events.append({
            "type": "action",
            "agent": agent,
            "content": f"{agent} considers the ethical implications of the situation at step {i}.",
            "scene_index": 0,
            "step_index": i,
        })

    if include_fc:
        events.append({
            "type": "forced_choice",
            "agent": agent_names[0],
            "content": "I choose to prioritize the vulnerable patient.",
            "scene_index": 0,
            "step_index": n_actions,
            "metadata": {"choices": ["Prioritize vulnerable", "Follow protocol"]},
        })

    events.append({"type": "scene_end", "content": "Scene 1 ended", "scene_index": 0, "step_index": n_actions + 1})

    return {
        "events": events,
        "metadata": {"llm_tokens_consumed": 500, "wall_clock_seconds": 30.0},
        "agent_names": agent_names,
        "scene_count": 1,
        "completed": True,
    }


class TestPrepareDataset:
    def test_basic_preparation(self):
        transcripts = [_sample_transcript()]
        examples = prepare_dataset(transcripts)
        assert len(examples) > 0

    def test_one_example_per_agent(self):
        transcripts = [_sample_transcript(agent_names=["Agent A", "Agent B"])]
        examples = prepare_dataset(transcripts)
        assert len(examples) == 2  # One per agent

    def test_instruction_format(self):
        transcripts = [_sample_transcript()]
        examples = prepare_dataset(transcripts)
        for ex in examples:
            assert "instruction" in ex
            assert "response" in ex
            assert len(ex["instruction"]) > 0
            assert len(ex["response"]) > 0

    def test_forced_choice_included(self):
        transcripts = [_sample_transcript(include_fc=True)]
        examples = prepare_dataset(transcripts)
        # The agent with forced choice should have it in their response
        first_agent_ex = [ex for ex in examples if "Dr. Chen" in ex["instruction"]][0]
        assert "decided" in first_agent_ex["response"].lower() or "choose" in first_agent_ex["response"].lower()

    def test_empty_transcript(self):
        examples = prepare_dataset([{"events": [], "agent_names": [], "metadata": {}}])
        assert examples == []

    def test_multiple_transcripts(self):
        transcripts = [_sample_transcript() for _ in range(5)]
        examples = prepare_dataset(transcripts)
        assert len(examples) == 10  # 2 agents × 5 transcripts


class TestSaveDataset:
    def test_save_and_load(self, tmp_path):
        examples = [{"instruction": "test", "response": "response"}]
        path = str(tmp_path / "dataset.jsonl")
        save_dataset(examples, path)

        with open(path) as f:
            loaded = [json.loads(line) for line in f]
        assert len(loaded) == 1
        assert loaded[0]["instruction"] == "test"
