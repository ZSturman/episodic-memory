"""Event and delta schemas for change detection and event memory.

ARCHITECTURAL INVARIANT: No predefined semantic categories.
All event types and delta types are learned from user interaction.
These are string-based and extensible, not fixed enums.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# EXTENSIBLE TYPE DEFINITIONS (No Fixed Enums)
# =============================================================================
# 
# ARCHITECTURAL INVARIANT: The system does not have built-in knowledge.
# All semantic categories emerge from user interaction and stored memory.
# 
# These string constants are provided ONLY for:
#   1. Internal system states (unknown, error)
#   2. Structural change types (new, missing, moved, changed)
# 
# Semantic labels like "opened", "closed", "turned_on" must be LEARNED
# from user interaction, not predefined.
# =============================================================================

# Structural delta types (not semantic - describe the structure of change)
DELTA_TYPE_NEW = "new"              # Something new appeared
DELTA_TYPE_MISSING = "missing"      # Something disappeared
DELTA_TYPE_MOVED = "moved"          # Something changed position
DELTA_TYPE_CHANGED = "changed"      # Something changed state

# System states only (not semantic labels)
EVENT_TYPE_UNKNOWN = "unknown"      # Event not yet labeled by user


# Legacy enum aliases for backward compatibility during migration
# TODO: Remove after all code migrated to string-based types
class DeltaType:
    """Legacy delta type constants for backward compatibility.
    
    DEPRECATED: Use string values directly. These will be removed.
    """
    NEW_ENTITY = DELTA_TYPE_NEW
    MISSING_ENTITY = DELTA_TYPE_MISSING
    MOVED_ENTITY = DELTA_TYPE_MOVED
    STATE_CHANGED = DELTA_TYPE_CHANGED


class EventType:
    """Legacy event type constants for backward compatibility.
    
    DEPRECATED: Use string values directly. These will be removed.
    All semantic event types (opened, closed, etc.) should be learned
    from user interaction, not predefined here.
    """
    UNKNOWN = EVENT_TYPE_UNKNOWN
    # All other event types are LEARNED, not predefined


class Delta(BaseModel):
    """A detected change (delta) between steps.
    
    Represents a single change observed in the environment:
    entity appeared, disappeared, moved, or changed state.
    
    ARCHITECTURAL INVARIANT: delta_type is a string, not a fixed enum.
    Structural types (new, missing, moved, changed) are system-defined.
    Semantic interpretation is handled by the backend recognition layer.
    """
    
    delta_id: str = Field(..., description="Unique identifier for this delta")
    delta_type: str = Field(
        ..., 
        description="Type of structural change detected (new/missing/moved/changed)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this delta was detected",
    )
    
    # Entity information
    entity_id: str | None = Field(
        default=None,
        description="GUID or node_id of the involved entity",
    )
    entity_label: str | None = Field(
        default=None,
        description="Human-readable label of the entity (learned, not predefined)",
    )
    # REMOVED: entity_category - backend learns categories from user interaction
    
    # State information for state changes
    pre_state: str | None = Field(
        default=None,
        description="State before the change",
    )
    post_state: str | None = Field(
        default=None,
        description="State after the change",
    )
    
    # Position information for movement (raw coordinates for internal use)
    pre_position: tuple[float, float, float] | None = Field(
        default=None,
        description="Position before movement (internal use)",
    )
    post_position: tuple[float, float, float] | None = Field(
        default=None,
        description="Position after movement (internal use)",
    )
    position_delta: float = Field(
        default=0.0,
        description="Distance moved (for moved deltas)",
    )
    
    # Relative position information (Phase 2 - landmark-based)
    pre_relative_position: dict[str, Any] | None = Field(
        default=None,
        description="Position relative to landmarks before movement",
    )
    post_relative_position: dict[str, Any] | None = Field(
        default=None,
        description="Position relative to landmarks after movement",
    )
    movement_description: str | None = Field(
        default=None,
        description="Human-readable movement description (e.g., 'moved from near the table to near the door')",
    )
    
    # Context
    location_label: str | None = Field(
        default=None,
        description="Location where this change occurred",
    )
    step_number: int = Field(
        default=0,
        description="Step number when delta was detected",
    )
    
    # Confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this delta detection",
    )
    
    # Evidence supporting this delta
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence strings supporting this delta",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class EventCandidate(BaseModel):
    """A candidate event detected from deltas.
    
    Represents a recognized pattern of change that may warrant a label.
    Can be learned (labeled) or recognized against existing event patterns.
    
    ARCHITECTURAL INVARIANT: event_type is a string, not a fixed enum.
    All semantic event types are learned from user interaction.
    """
    
    event_id: str = Field(..., description="Unique identifier for this event")
    event_type: str = Field(
        default=EVENT_TYPE_UNKNOWN,
        description="Event type (learned from user, not predefined)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this event was detected",
    )
    
    # Label information
    label: str = Field(
        default="unknown",
        description="Human-readable event label (learned from user)",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Alternative labels/aliases for this event",
    )
    is_learned: bool = Field(
        default=False,
        description="Whether this event pattern was learned from user",
    )
    
    # Involved entities
    involved_entity_ids: list[str] = Field(
        default_factory=list,
        description="IDs of entities involved in this event",
    )
    involved_entity_labels: list[str] = Field(
        default_factory=list,
        description="Labels of entities involved in this event",
    )
    
    # State signature for matching
    pre_state_signature: str | None = Field(
        default=None,
        description="Signature of state before event (e.g., 'drawer:closed')",
    )
    post_state_signature: str | None = Field(
        default=None,
        description="Signature of state after event (e.g., 'drawer:open')",
    )
    
    # Source delta(s)
    source_delta_ids: list[str] = Field(
        default_factory=list,
        description="IDs of deltas that triggered this event",
    )
    
    # Context
    location_label: str | None = Field(
        default=None,
        description="Location where this event occurred",
    )
    step_number: int = Field(
        default=0,
        description="Step number when event was detected",
    )
    
    # Confidence and evidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this event detection",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence strings supporting this event",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage in ACF events list."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,  # Now a string, not enum
            "timestamp": self.timestamp.isoformat(),
            "label": self.label,
            "labels": self.labels,
            "is_learned": self.is_learned,
            "involved_entity_ids": self.involved_entity_ids,
            "involved_entity_labels": self.involved_entity_labels,
            "pre_state_signature": self.pre_state_signature,
            "post_state_signature": self.post_state_signature,
            "source_delta_ids": self.source_delta_ids,
            "location_label": self.location_label,
            "step_number": self.step_number,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "extras": self.extras,
        }


class StateSignature(BaseModel):
    """A signature representing entity state for event matching.
    
    Used to identify and match recurring event patterns.
    """
    
    entity_category: str = Field(..., description="Category of entity (drawer, light, etc.)")
    state_key: str = Field(..., description="State key (e.g., 'open', 'on')")
    state_value: Any = Field(..., description="State value (bool, str, etc.)")
    
    def to_string(self) -> str:
        """Convert to string signature for matching."""
        return f"{self.entity_category}:{self.state_key}={self.state_value}"
    
    @classmethod
    def from_string(cls, signature: str) -> "StateSignature":
        """Parse from string signature."""
        parts = signature.split(":")
        if len(parts) != 2:
            return cls(entity_category="unknown", state_key="unknown", state_value=signature)
        
        category = parts[0]
        key_value = parts[1].split("=")
        if len(key_value) != 2:
            return cls(entity_category=category, state_key=parts[1], state_value="")
        
        return cls(
            entity_category=category,
            state_key=key_value[0],
            state_value=key_value[1],
        )
    
    model_config = {"frozen": True}
