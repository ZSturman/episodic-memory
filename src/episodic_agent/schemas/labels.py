"""Label management and conflict resolution data contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConflictResolutionType(str, Enum):
    """Types of conflict resolution actions."""
    
    MERGE = "merge"           # Treat as same thing (create alias)
    DISAMBIGUATE = "disambiguate"  # Different things with same name
    RENAME = "rename"         # Rename one of the nodes
    KEEP_BOTH = "keep_both"   # Keep both with same label (rare)
    PENDING = "pending"       # Not yet resolved


class LabelConflict(BaseModel):
    """A conflict when assigning a label that exists on another node.
    
    Generated when attempting to assign a label to a node, but that
    label already exists on a different node.
    """

    conflict_id: str = Field(..., description="Unique identifier for this conflict")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this conflict was created",
    )
    
    # The label in question
    label: str = Field(..., description="The conflicting label")
    
    # Competing nodes
    existing_node_id: str = Field(
        ...,
        description="ID of the node that already has this label",
    )
    new_node_id: str = Field(
        ...,
        description="ID of the node trying to claim this label",
    )
    
    # Context for resolution
    existing_node_type: str = Field(
        default="unknown",
        description="Type of the existing node",
    )
    new_node_type: str = Field(
        default="unknown",
        description="Type of the new node",
    )
    
    # Resolution
    resolution: ConflictResolutionType = Field(
        default=ConflictResolutionType.PENDING,
        description="How the conflict was resolved",
    )
    resolved_at: datetime | None = Field(
        default=None,
        description="When the conflict was resolved",
    )
    resolution_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the resolution",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}


class LabelAssignment(BaseModel):
    """Record of a label being assigned to a node.
    
    Used for logging and audit trail.
    """

    assignment_id: str = Field(..., description="Unique identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the assignment occurred",
    )
    
    node_id: str = Field(..., description="ID of the node")
    label: str = Field(..., description="The assigned label")
    is_primary: bool = Field(
        default=False,
        description="Whether this is the primary label",
    )
    
    # Source of the assignment
    source: str = Field(
        default="system",
        description="How the label was assigned (user, system, merge)",
    )
    
    # If this resulted from conflict resolution
    conflict_id: str | None = Field(
        default=None,
        description="ID of resolved conflict if applicable",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": True}
