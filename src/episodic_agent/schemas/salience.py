"""Salience and cued recall data contracts.

ARCHITECTURAL INVARIANT: Memory retrieval uses weighted cues with tunable bias.
No predefined salience categories - weights are learned from experience.

This module provides:
- SalienceWeights: Per-link weights for different cue types
- CuedRecallQuery: Multi-cue retrieval specification
- RecallResult: Memory retrieval with confidence and cue provenance
- EntityHypothesis: Same-entity hypotheses across observations
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# CUE TYPE DEFINITIONS
# =============================================================================
# 
# ARCHITECTURAL INVARIANT: Cue types are structural retrieval paths.
# The weights determine how much each cue contributes to retrieval.
# =============================================================================

class CueType(str, Enum):
    """Types of retrieval cues.
    
    Each cue type represents a different retrieval path through memory:
    - LOCATION: "Where was I when..." - spatial context cues
    - ENTITY: "What was involved..." - object/entity cues  
    - TEMPORAL: "When did..." - time-based cues
    - SEMANTIC: "What was similar to..." - label/meaning cues
    - VISUAL: "What did it look like..." - perceptual similarity cues
    - EVENT: "What happened when..." - event-based cues
    """
    LOCATION = "location"
    ENTITY = "entity"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    VISUAL = "visual"
    EVENT = "event"


# Default weights for each cue type (tunable at runtime)
DEFAULT_CUE_WEIGHTS: dict[CueType, float] = {
    CueType.LOCATION: 0.25,
    CueType.ENTITY: 0.20,
    CueType.TEMPORAL: 0.15,
    CueType.SEMANTIC: 0.20,
    CueType.VISUAL: 0.10,
    CueType.EVENT: 0.10,
}


# =============================================================================
# SALIENCE WEIGHTS
# =============================================================================

class SalienceWeights(BaseModel):
    """Per-link salience weights for different cue types.
    
    ARCHITECTURAL INVARIANT: Salience is learned, not predefined.
    Weights determine how much each cue type contributes when this
    link is traversed during memory retrieval.
    
    Example: A strong location-entity link (saw X in kitchen) has high
    location salience, while a strong event-entity link (X moved) has
    high event salience.
    """
    
    # Weights per cue type (0.0 to 1.0)
    location: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly location cues this link",
    )
    entity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly entity cues this link",
    )
    temporal: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly temporal cues this link",
    )
    semantic: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly semantic labels cue this link",
    )
    visual: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly visual similarity cues this link",
    )
    event: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly events cue this link",
    )
    
    # Overall salience (computed from max or weighted combination)
    overall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall salience of this link",
    )
    
    # Track which cue type was strongest (for debugging/analysis)
    dominant_cue: CueType | None = Field(
        default=None,
        description="The cue type with highest weight for this link",
    )
    
    model_config = {"frozen": False}
    
    def compute_overall(self) -> float:
        """Compute overall salience from individual weights."""
        weights = [
            self.location,
            self.entity,
            self.temporal,
            self.semantic,
            self.visual,
            self.event,
        ]
        # Overall is max of individual weights
        self.overall = max(weights) if weights else 0.0
        
        # Find dominant cue
        cue_map = {
            self.location: CueType.LOCATION,
            self.entity: CueType.ENTITY,
            self.temporal: CueType.TEMPORAL,
            self.semantic: CueType.SEMANTIC,
            self.visual: CueType.VISUAL,
            self.event: CueType.EVENT,
        }
        max_weight = max(weights) if weights else 0.0
        if max_weight > 0:
            self.dominant_cue = cue_map.get(max_weight, CueType.LOCATION)
        
        return self.overall
    
    def boost(self, cue_type: CueType, amount: float = 0.1) -> None:
        """Boost a specific cue type's weight (learning)."""
        current = getattr(self, cue_type.value)
        new_value = min(1.0, current + amount)
        setattr(self, cue_type.value, new_value)
        self.compute_overall()
    
    def decay(self, factor: float = 0.95) -> None:
        """Decay all weights over time."""
        self.location *= factor
        self.entity *= factor
        self.temporal *= factor
        self.semantic *= factor
        self.visual *= factor
        self.event *= factor
        self.compute_overall()
    
    def weighted_score(self, query_weights: dict[CueType, float]) -> float:
        """Compute weighted score given query-time cue preferences.
        
        Args:
            query_weights: Dict mapping cue types to query-time weights
            
        Returns:
            Combined score based on link salience × query weights
        """
        score = 0.0
        for cue_type, query_weight in query_weights.items():
            link_weight = getattr(self, cue_type.value, 0.0)
            score += link_weight * query_weight
        return score


# =============================================================================
# CUED RECALL QUERY
# =============================================================================

class CuedRecallQuery(BaseModel):
    """A multi-cue query for retrieving memories.
    
    ARCHITECTURAL INVARIANT: Retrieval combines multiple cue types
    with tunable weights at query time.
    """
    
    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique query identifier",
    )
    
    # Active cues for this query
    location_id: str | None = Field(
        default=None,
        description="Location cue (retrieve memories from this location)",
    )
    location_label: str | None = Field(
        default=None,
        description="Location label cue",
    )
    
    entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity cues (retrieve memories involving these entities)",
    )
    
    time_start: datetime | None = Field(
        default=None,
        description="Temporal cue start",
    )
    time_end: datetime | None = Field(
        default=None,
        description="Temporal cue end",
    )
    
    labels: list[str] = Field(
        default_factory=list,
        description="Semantic label cues",
    )
    
    visual_embedding: list[float] | None = Field(
        default=None,
        description="Visual similarity cue (feature embedding)",
    )
    
    event_type: str | None = Field(
        default=None,
        description="Event type cue",
    )
    
    # Query-time weight adjustments
    cue_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Override default cue weights for this query",
    )
    
    # Retrieval parameters
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    
    min_salience: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum salience threshold",
    )
    
    # Whether to include high-salience moments first
    prioritize_salient: bool = Field(
        default=True,
        description="Sort results by salience (high-salience first)",
    )
    
    model_config = {"frozen": False}
    
    def get_active_cue_types(self) -> list[CueType]:
        """Return list of cue types active in this query."""
        active = []
        if self.location_id or self.location_label:
            active.append(CueType.LOCATION)
        if self.entity_ids:
            active.append(CueType.ENTITY)
        if self.time_start or self.time_end:
            active.append(CueType.TEMPORAL)
        if self.labels:
            active.append(CueType.SEMANTIC)
        if self.visual_embedding:
            active.append(CueType.VISUAL)
        if self.event_type:
            active.append(CueType.EVENT)
        return active
    
    def get_effective_weights(self) -> dict[CueType, float]:
        """Get effective weights for this query (defaults + overrides)."""
        weights = dict(DEFAULT_CUE_WEIGHTS)
        for key, value in self.cue_weights.items():
            try:
                cue_type = CueType(key)
                weights[cue_type] = value
            except ValueError:
                pass  # Ignore invalid cue types
        return weights


# =============================================================================
# RECALL RESULT
# =============================================================================

class RecallResult(BaseModel):
    """Result of a cued recall query.
    
    Contains retrieved memories with salience scores and
    provenance (which cue types contributed).
    """
    
    query_id: str = Field(
        ...,
        description="ID of the query that produced this result",
    )
    
    # Retrieved memories
    node_ids: list[str] = Field(
        default_factory=list,
        description="IDs of retrieved memory nodes",
    )
    
    episode_ids: list[str] = Field(
        default_factory=list,
        description="IDs of retrieved episodes",
    )
    
    # Salience scores per result
    node_salience: list[float] = Field(
        default_factory=list,
        description="Salience score for each node",
    )
    
    episode_salience: list[float] = Field(
        default_factory=list,
        description="Salience score for each episode",
    )
    
    # Cue provenance (which cues contributed to each result)
    cue_contributions: list[dict[str, float]] = Field(
        default_factory=list,
        description="Per-result breakdown of cue type contributions",
    )
    
    # Timing
    query_time_ms: float = Field(
        default=0.0,
        description="Time taken to execute query in milliseconds",
    )
    
    # Debug info
    total_candidates: int = Field(
        default=0,
        description="Total candidates considered before filtering",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# ENTITY HYPOTHESIS
# =============================================================================

class EntityHypothesis(BaseModel):
    """Hypothesis that two observations are the same entity.
    
    ARCHITECTURAL INVARIANT: Same-entity hypotheses are based on
    spatial and visual consistency, not predefined object categories.
    
    Example: Blue mug at position (1, 0, 2) becomes red mug at same
    position → hypothesize they're the same entity (the mug changed color).
    """
    
    hypothesis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique hypothesis identifier",
    )
    
    # The two observations being compared
    observation_a_id: str = Field(
        ...,
        description="ID of first observation",
    )
    observation_b_id: str = Field(
        ...,
        description="ID of second observation",
    )
    
    # Location context
    location_id: str = Field(
        ...,
        description="Common location where both were observed",
    )
    
    # Relative position (agent-centric)
    relative_position_a: tuple[float, float, float] | None = Field(
        default=None,
        description="Relative position of observation A",
    )
    relative_position_b: tuple[float, float, float] | None = Field(
        default=None,
        description="Relative position of observation B",
    )
    
    # Position similarity
    position_distance: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance between positions (0 = same spot)",
    )
    
    # Visual similarity
    visual_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Visual feature similarity (0 = different, 1 = identical)",
    )
    
    # Time gap
    time_gap_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time between observations",
    )
    
    # Hypothesis confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that these are the same entity",
    )
    
    # Hypothesis status
    status: str = Field(
        default="pending",
        description="Status: pending, confirmed, rejected",
    )
    
    # Reason for hypothesis
    reason: str = Field(
        default="",
        description="Explanation for the hypothesis",
    )
    
    # Labels (if different, indicates change)
    label_a: str | None = Field(
        default=None,
        description="Label of observation A",
    )
    label_b: str | None = Field(
        default=None,
        description="Label of observation B",
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this hypothesis was created",
    )
    resolved_at: datetime | None = Field(
        default=None,
        description="When this hypothesis was confirmed/rejected",
    )
    
    model_config = {"frozen": False}
    
    @classmethod
    def from_observations(
        cls,
        obs_a_id: str,
        obs_b_id: str,
        location_id: str,
        pos_a: tuple[float, float, float] | None = None,
        pos_b: tuple[float, float, float] | None = None,
        label_a: str | None = None,
        label_b: str | None = None,
        visual_sim: float = 0.0,
        time_gap: float = 0.0,
    ) -> "EntityHypothesis":
        """Create hypothesis from two observations."""
        # Compute position distance
        pos_dist = 0.0
        if pos_a and pos_b:
            pos_dist = (
                (pos_a[0] - pos_b[0]) ** 2 +
                (pos_a[1] - pos_b[1]) ** 2 +
                (pos_a[2] - pos_b[2]) ** 2
            ) ** 0.5
        
        # Compute initial confidence
        # Higher confidence if close position and similar visuals
        position_score = 1.0 / (1.0 + pos_dist) if pos_dist < 2.0 else 0.0
        confidence = (position_score * 0.6) + (visual_sim * 0.4)
        
        # Determine reason
        reasons = []
        if pos_dist < 0.5:
            reasons.append("same position")
        elif pos_dist < 1.0:
            reasons.append("nearby position")
        if visual_sim > 0.7:
            reasons.append("visually similar")
        if time_gap < 60:
            reasons.append("recent observation")
        
        reason = ", ".join(reasons) if reasons else "spatial proximity"
        
        return cls(
            observation_a_id=obs_a_id,
            observation_b_id=obs_b_id,
            location_id=location_id,
            relative_position_a=pos_a,
            relative_position_b=pos_b,
            position_distance=pos_dist,
            visual_similarity=visual_sim,
            time_gap_seconds=time_gap,
            confidence=confidence,
            reason=reason,
            label_a=label_a,
            label_b=label_b,
        )
    
    def confirm(self, reason: str = "user confirmed") -> None:
        """Confirm this hypothesis."""
        self.status = "confirmed"
        self.reason = reason
        self.resolved_at = datetime.now()
    
    def reject(self, reason: str = "user rejected") -> None:
        """Reject this hypothesis."""
        self.status = "rejected"
        self.reason = reason
        self.resolved_at = datetime.now()


# =============================================================================
# REDUNDANT CUE STORE
# =============================================================================

class RedundantCue(BaseModel):
    """A cue that can trigger recall of a memory.
    
    ARCHITECTURAL INVARIANT: Multiple cues point to the same memory,
    providing redundant retrieval paths (like human memory).
    """
    
    cue_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique cue identifier",
    )
    
    cue_type: CueType = Field(
        ...,
        description="Type of this cue",
    )
    
    # What this cue points to
    target_node_id: str = Field(
        ...,
        description="Memory node this cue retrieves",
    )
    
    target_episode_id: str | None = Field(
        default=None,
        description="Episode this cue retrieves (optional)",
    )
    
    # Cue content (depends on type)
    location_id: str | None = Field(
        default=None,
        description="Location ID (for LOCATION cues)",
    )
    
    entity_id: str | None = Field(
        default=None,
        description="Entity ID (for ENTITY cues)",
    )
    
    label: str | None = Field(
        default=None,
        description="Semantic label (for SEMANTIC cues)",
    )
    
    timestamp: datetime | None = Field(
        default=None,
        description="Timestamp (for TEMPORAL cues)",
    )
    
    visual_embedding: list[float] | None = Field(
        default=None,
        description="Visual features (for VISUAL cues)",
    )
    
    event_id: str | None = Field(
        default=None,
        description="Event ID (for EVENT cues)",
    )
    
    # Cue strength (how likely to trigger recall)
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How strongly this cue triggers recall",
    )
    
    # Access count (strengthens with use)
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this cue triggered recall",
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this cue was created",
    )
    last_accessed: datetime = Field(
        default_factory=datetime.now,
        description="Last time this cue was used",
    )
    
    model_config = {"frozen": False}
    
    def access(self) -> None:
        """Record an access to this cue (strengthens it)."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        # Strengthen slightly with each access
        self.strength = min(1.0, self.strength + 0.01)


# =============================================================================
# LOCATION REVISIT EVENT
# =============================================================================

class LocationRevisit(BaseModel):
    """Record of entering a previously-visited location.
    
    ARCHITECTURAL INVARIANT: Entering a location triggers cued recall
    of prior visits, surfacing salient moments.
    """
    
    revisit_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique revisit identifier",
    )
    
    location_id: str = Field(
        ...,
        description="Location being revisited",
    )
    
    location_label: str | None = Field(
        default=None,
        description="Label of the location",
    )
    
    # Current visit
    current_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this revisit occurred",
    )
    
    # Prior visits
    prior_visit_count: int = Field(
        default=0,
        ge=0,
        description="Number of prior visits",
    )
    
    prior_visit_timestamps: list[datetime] = Field(
        default_factory=list,
        description="Timestamps of prior visits",
    )
    
    # Recalled memories
    recalled_episode_ids: list[str] = Field(
        default_factory=list,
        description="Episodes recalled upon entry",
    )
    
    recalled_salience: list[float] = Field(
        default_factory=list,
        description="Salience of each recalled episode",
    )
    
    # Salient moments from prior visits
    salient_moments: list[str] = Field(
        default_factory=list,
        description="High-salience moments from prior visits",
    )
    
    # Same-entity hypotheses triggered
    entity_hypotheses: list[str] = Field(
        default_factory=list,
        description="Entity hypothesis IDs triggered by revisit",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CueType",
    # Constants
    "DEFAULT_CUE_WEIGHTS",
    # Models
    "SalienceWeights",
    "CuedRecallQuery",
    "RecallResult",
    "EntityHypothesis",
    "RedundantCue",
    "LocationRevisit",
]
