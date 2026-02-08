"""Structured event types for panorama agent observability.

Defines the semantic event model that drives the panorama dashboard.
Every major perception, matching, and labeling decision emits a typed
``PanoramaEvent`` that the API server and UI can consume directly.

Design principle: the UI should never infer meaning from logs — it
should receive explicit semantic events describing what happened and
why.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# =====================================================================
# Agent state machine states
# =====================================================================

class PanoramaAgentState(str, Enum):
    """Describes the agent's current investigative posture.

    These states determine whether a label request is appropriate and
    what kind of evidence accumulation is happening.
    """

    investigating_unknown = "investigating_unknown"
    """First encounter with unfamiliar scene — gathering evidence."""

    matching_known = "matching_known"
    """Scene resembles a known location — verifying."""

    low_confidence_match = "low_confidence_match"
    """Best candidate is weak — may be new or ambiguous."""

    confident_match = "confident_match"
    """High-confidence match to a known location."""

    novel_location_candidate = "novel_location_candidate"
    """Evidence suggests a genuinely new location after sustained investigation."""

    label_request = "label_request"
    """Agent is requesting a label from the user (evidence bundle ready)."""


# =====================================================================
# Event types
# =====================================================================

class PanoramaEventType(str, Enum):
    """Semantic event types emitted by the panorama pipeline."""

    perception_update = "perception_update"
    match_evaluation = "match_evaluation"
    state_transition = "state_transition"
    investigation_window = "investigation_window"
    label_request = "label_request"
    memory_write = "memory_write"


# =====================================================================
# Typed payloads
# =====================================================================

class MatchCandidate(BaseModel):
    """A single candidate in a location match evaluation."""

    location_id: str
    label: str = "unknown"
    confidence: float = 0.0
    distance: float = 1.0

    model_config = ConfigDict(frozen=True)


class MatchEvaluation(BaseModel):
    """Full ranked evaluation of the current scene against known locations."""

    candidates: list[MatchCandidate] = Field(default_factory=list)
    top_margin: float = 0.0
    """Confidence gap between rank-1 and rank-2 candidates."""

    hysteresis_active: bool = False
    """Whether the hysteresis counter is active (transition pending)."""

    stabilization_frames: int = 0
    """How many consecutive frames have supported the current hypothesis."""

    current_location_id: str | None = None
    current_distance: float = 0.0


class EvidenceBundle(BaseModel):
    """Accumulated evidence supporting a label request.

    Shown to the user *before* they are asked to provide a label,
    so they can see what the agent observed and why it is uncertain.
    """

    viewport_images_b64: list[str] = Field(default_factory=list)
    """Last N viewport JPEGs (base64) from the investigation window."""

    feature_summaries: list[dict[str, Any]] = Field(default_factory=list)
    """Per-step feature summaries (brightness, edges, colours)."""

    match_scores: dict[str, float] = Field(default_factory=dict)
    """location_id → best confidence seen during investigation."""

    confidence_history: list[float] = Field(default_factory=list)
    """Per-step confidence values during the investigation window."""

    investigation_steps: int = 0
    """How many steps the investigation window lasted."""

    margin: float = 0.0
    """Final margin between top-2 candidates."""

    best_candidate_label: str | None = None
    best_candidate_confidence: float = 0.0


class PerceptionPayload(BaseModel):
    """Payload for a perception_update event."""

    confidence: float = 0.0
    feature_summary: dict[str, Any] = Field(default_factory=dict)
    heading_index: int = 0
    total_headings: int = 0
    heading_deg: float = 0.0
    is_panoramic_complete: bool = False
    embedding_norm: float = 0.0
    source_file: str = ""


class MemoryWritePayload(BaseModel):
    """Payload for a memory_write event."""

    location_id: str
    label: str = "unknown"
    is_new: bool = True
    observation_count: int = 1
    embedding_norm: float = 0.0


class StateTransitionPayload(BaseModel):
    """Payload for a state_transition event."""

    previous_state: str
    new_state: str
    reason: str = ""
    confidence: float = 0.0
    steps_in_previous: int = 0


# =====================================================================
# Memory introspection models
# =====================================================================

class MemorySummary(BaseModel):
    """Compact per-location summary for the dashboard memory list."""

    location_id: str
    label: str = "unknown"
    observation_count: int = 0
    embedding_centroid_norm: float = 0.0
    variance: float = 0.0
    stability_score: float = 0.0
    first_seen_step: int = 0
    last_seen_step: int = 0
    confidence_vs_current: float = 0.0
    """Cosine similarity to the active scene embedding."""

    match_history: list[float] = Field(default_factory=list)
    """Recent confidence values when this location was a candidate."""

    aliases: list[str] = Field(default_factory=list)
    entity_cooccurrence: dict[str, int] = Field(default_factory=dict)


class MemoryCard(BaseModel):
    """Full introspection data for a single memory location."""

    location_id: str
    label: str = "unknown"
    embedding_centroid: list[float] = Field(default_factory=list)
    aggregated_features: dict[str, Any] = Field(default_factory=dict)
    variance: float = 0.0
    stability_score: float = 0.0
    observation_count: int = 0
    first_seen_step: int = 0
    last_seen_step: int = 0
    match_confidence_history: list[dict[str, Any]] = Field(default_factory=list)
    """List of {step, confidence} entries."""

    co_occurring_entities: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    transition_positions: list[list[float]] = Field(default_factory=list)


# =====================================================================
# Main event envelope
# =====================================================================

class PanoramaEvent(BaseModel):
    """A single structured event emitted by the panorama pipeline.

    This is the backbone of observability — every major decision is
    wrapped in one of these and pushed to the event bus.
    """

    event_type: PanoramaEventType
    timestamp: datetime = Field(default_factory=datetime.now)
    step: int = 0
    state: PanoramaAgentState = PanoramaAgentState.investigating_unknown

    # Event-specific payload (typed union via discriminator)
    payload: dict[str, Any] = Field(default_factory=dict)

    # Optional evidence bundle (populated for label_request events)
    evidence_bundle: Optional[EvidenceBundle] = None

    model_config = ConfigDict(use_enum_values=True)
