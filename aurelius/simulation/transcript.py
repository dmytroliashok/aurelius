"""Transcript models and extraction from Concordia simulation output."""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    ACTION = "action"
    NARRATION = "narration"
    FORCED_CHOICE = "forced_choice"
    REFLECTION = "reflection"
    SCENE_START = "scene_start"
    SCENE_END = "scene_end"


class TranscriptEvent(BaseModel):
    """A single event in the simulation transcript."""

    type: EventType
    agent: str | None = None  # None for narrations / scene markers
    content: str
    scene_index: int = 0
    step_index: int = 0
    metadata: dict = {}
    chain_of_thought: list[dict] = []  # 7-step moral reasoning chain
    gm_resolution: str | None = None  # GM's narrative resolution of the event


class TranscriptMetadata(BaseModel):
    """Metadata about the simulation run."""

    llm_tokens_consumed: int = 0
    wall_clock_seconds: float = 0.0
    concordia_version: str = ""
    docker_image_tag: str = ""
    container_id: str = ""
    llm_model: str = ""


class Transcript(BaseModel):
    """Structured simulation transcript."""

    events: list[TranscriptEvent] = []
    metadata: TranscriptMetadata = TranscriptMetadata()
    agent_names: list[str] = []
    scene_count: int = 0
    completed: bool = False


MAX_TRANSCRIPT_EVENTS = 10_000


def extract_transcript(raw_output: dict) -> Transcript:
    """Parse raw Concordia simulation output into a structured Transcript.

    The raw_output is expected to be a JSON dict from the simulation entrypoint
    with keys: events (list), metadata (dict), completed (bool).

    This function normalizes the output into the Transcript model.
    """
    events = []
    raw_events = raw_output.get("events", [])
    if len(raw_events) > MAX_TRANSCRIPT_EVENTS:
        logger.warning(
            "Transcript has %d events, truncating to %d",
            len(raw_events),
            MAX_TRANSCRIPT_EVENTS,
        )
        raw_events = raw_events[:MAX_TRANSCRIPT_EVENTS]
    for raw_event in raw_events:
        event_type = raw_event.get("type", "narration")
        try:
            etype = EventType(event_type)
        except ValueError:
            etype = EventType.NARRATION

        events.append(
            TranscriptEvent(
                type=etype,
                agent=raw_event.get("agent"),
                content=raw_event.get("content", ""),
                scene_index=raw_event.get("scene_index", 0),
                step_index=raw_event.get("step_index", 0),
                metadata=raw_event.get("metadata", {}),
                chain_of_thought=raw_event.get("chain_of_thought", []),
                gm_resolution=raw_event.get("gm_resolution"),
            )
        )

    raw_meta = raw_output.get("metadata", {})
    metadata = TranscriptMetadata(
        llm_tokens_consumed=raw_meta.get("llm_tokens_consumed", 0),
        wall_clock_seconds=raw_meta.get("wall_clock_seconds", 0.0),
        concordia_version=raw_meta.get("concordia_version", ""),
        docker_image_tag=raw_meta.get("docker_image_tag", ""),
        container_id=raw_meta.get("container_id", ""),
        llm_model=raw_meta.get("llm_model", ""),
    )

    agent_names = list({e.agent for e in events if e.agent is not None})
    scene_indices = {e.scene_index for e in events}
    scene_count = len(scene_indices) if scene_indices else 0

    return Transcript(
        events=events,
        metadata=metadata,
        agent_names=sorted(agent_names),
        scene_count=scene_count,
        completed=raw_output.get("completed", False),
    )
