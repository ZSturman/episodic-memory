"""Active Context Frame and Episode data contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from episodic_agent.schemas.frames import ObjectCandidate, Percept


class ActiveContextFrame(BaseModel):
    """Mutable working memory representing current situational context.
    
    The ACF accumulates information about the current episode:
    location, entities present, ongoing events, and their relationships.
    """

    acf_id: str = Field(..., description="Unique identifier for this ACF instance")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this ACF was created",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last modification timestamp",
    )
    
    # Current step count within this ACF
    step_count: int = Field(
        default=0,
        description="Number of steps since ACF creation",
    )
    
    # Location context
    location_label: str = Field(
        default="unknown",
        description="Current location label",
    )
    location_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in location identification",
    )
    location_embedding: list[float] | None = Field(
        default=None,
        description="Embedding for current location",
    )
    
    # Entity context
    entities: list[ObjectCandidate] = Field(
        default_factory=list,
        description="Currently recognized entities in context",
    )
    
    # Event context (events detected this episode)
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Events detected in current episode",
    )
    
    # Delta context (changes detected this episode) - Phase 5
    deltas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Deltas (changes) detected in current episode",
    )
    
    # Most recent percept
    current_percept: Percept | None = Field(
        default=None,
        description="Most recently processed percept",
    )
    
    # For later conflict resolution
    conflict_id: str | None = Field(
        default=None,
        description="ID linking to an active conflict if present",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = datetime.now()
    
    def get_recent_deltas(self, count: int = 5) -> list[dict[str, Any]]:
        """Get the most recent deltas.
        
        Args:
            count: Maximum number of deltas to return.
            
        Returns:
            List of recent delta dictionaries.
        """
        # Combine deltas from field and extras for backward compatibility
        all_deltas = list(self.deltas)
        if "deltas" in self.extras:
            all_deltas.extend(self.extras["deltas"])
        return all_deltas[-count:] if all_deltas else []
    
    def get_recent_events(self, count: int = 5) -> list[dict[str, Any]]:
        """Get the most recent events.
        
        Args:
            count: Maximum number of events to return.
            
        Returns:
            List of recent event dictionaries.
        """
        return self.events[-count:] if self.events else []


class Episode(BaseModel):
    """Frozen snapshot of an ACF - append-only episodic memory.
    
    Episodes are immutable records of past contextual states,
    created when a boundary is detected (location change, significant
    event, or time-based threshold).
    """

    episode_id: str = Field(..., description="Unique identifier for this episode")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this episode was frozen",
    )
    
    # Duration information
    start_time: datetime = Field(..., description="When the episode began")
    end_time: datetime = Field(..., description="When the episode ended")
    step_count: int = Field(..., description="Number of steps in this episode")
    
    # Frozen context
    location_label: str = Field(..., description="Location label for this episode")
    location_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in location at freeze time",
    )
    location_embedding: list[float] | None = Field(
        default=None,
        description="Location embedding at freeze time",
    )
    
    # Entities present during episode
    entities: list[ObjectCandidate] = Field(
        default_factory=list,
        description="Entities present during this episode",
    )
    
    # Events that occurred
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Events that occurred during this episode",
    )
    
    # Deltas detected during episode (Phase 5)
    deltas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Deltas (changes) detected during this episode",
    )
    
    # Episode-level embedding for retrieval
    episode_embedding: list[float] | None = Field(
        default=None,
        description="Aggregate embedding for similarity search",
    )
    
    # Source ACF ID for provenance
    source_acf_id: str = Field(..., description="ID of the ACF this was frozen from")
    
    # Boundary information
    boundary_reason: str = Field(
        default="unknown",
        description="Why this episode boundary was triggered",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": True}  # Episodes are immutable
