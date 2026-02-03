"""Event and delta schemas for change detection and event memory."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DeltaType(str, Enum):
    """Types of deltas (changes) detected between steps."""
    
    NEW_ENTITY = "new_entity"           # Entity appeared (not in recent context)
    MISSING_ENTITY = "missing_entity"   # Entity disappeared (was in recent context)
    MOVED_ENTITY = "moved_entity"       # Entity position changed significantly
    STATE_CHANGED = "state_changed"     # Entity state changed (open/closed, on/off)


class EventType(str, Enum):
    """Types of events that can be detected from deltas."""
    
    # State change events
    OPENED = "opened"           # closed -> open (drawer, door, etc.)
    CLOSED = "closed"           # open -> closed
    TURNED_ON = "turned_on"     # off -> on (light, switch, etc.)
    TURNED_OFF = "turned_off"   # on -> off
    
    # Appearance/disappearance events  
    APPEARED = "appeared"       # Entity appeared
    DISAPPEARED = "disappeared" # Entity disappeared
    MOVED = "moved"             # Entity moved significantly
    
    # Interaction events
    PICKED_UP = "picked_up"     # Entity was picked up
    PUT_DOWN = "put_down"       # Entity was put down
    
    # Generic
    STATE_CHANGE = "state_change"  # Generic state change
    UNKNOWN = "unknown"            # Unknown event type


class Delta(BaseModel):
    """A detected change (delta) between steps.
    
    Represents a single change observed in the environment:
    entity appeared, disappeared, moved, or changed state.
    """
    
    delta_id: str = Field(..., description="Unique identifier for this delta")
    delta_type: DeltaType = Field(..., description="Type of change detected")
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
        description="Human-readable label of the entity",
    )
    entity_category: str | None = Field(
        default=None,
        description="Category of the entity (furniture, light, ball, etc.)",
    )
    
    # State information for state changes
    pre_state: str | None = Field(
        default=None,
        description="State before the change",
    )
    post_state: str | None = Field(
        default=None,
        description="State after the change",
    )
    
    # Position information for movement
    pre_position: tuple[float, float, float] | None = Field(
        default=None,
        description="Position before movement",
    )
    post_position: tuple[float, float, float] | None = Field(
        default=None,
        description="Position after movement",
    )
    position_delta: float = Field(
        default=0.0,
        description="Distance moved (for moved_entity deltas)",
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
    Can be learned (labeled) or recognized against existing event types.
    """
    
    event_id: str = Field(..., description="Unique identifier for this event")
    event_type: EventType = Field(
        default=EventType.UNKNOWN,
        description="Recognized event type",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this event was detected",
    )
    
    # Label information
    label: str = Field(
        default="unknown_event",
        description="Human-readable event label",
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
            "event_type": self.event_type.value,
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
