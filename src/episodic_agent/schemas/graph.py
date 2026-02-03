"""Graph memory data contracts for associative retrieval."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the associative graph."""
    
    CONTEXT = "context"    # Context/episode hub
    CUE = "cue"            # Retrieval cue
    LOCATION = "location"
    ENTITY = "entity"
    EVENT = "event"
    EPISODE = "episode"
    GOAL = "goal"
    CONCEPT = "concept"


class EdgeType(str, Enum):
    """Types of edges in the associative graph."""
    
    # Core relationships
    CUE_OF = "cue_of"           # cue -> context it retrieves
    IN_CONTEXT = "in_context"   # entity/event -> context
    TYPICAL_IN = "typical_in"   # entity -> location (typical)
    INVOLVES = "involves"       # event involves entity
    
    # Event relationships (Phase 5)
    IN_EVENT = "in_event"       # entity -> event (entity participated in event)
    TRIGGERED_BY = "triggered_by"  # event -> delta that triggered it
    OCCURRED_IN = "occurred_in" # event -> location where it occurred
    
    # Label/identity relationships  
    ALIAS_OF = "alias_of"       # node is alias of another
    MERGED_INTO = "merged_into" # node was merged into another
    
    # Legacy/compatibility
    CONTAINS = "contains"       # location contains entity
    OCCURRED_AT = "occurred_at" # event occurred at location
    PART_OF = "part_of"         # entity/event part of episode
    SIMILAR_TO = "similar_to"   # similarity link
    TEMPORAL = "temporal"       # temporal sequence
    CAUSAL = "causal"           # causal relationship


class GraphNode(BaseModel):
    """A node in the associative graph memory.
    
    Nodes represent locations, entities, events, episodes, or concepts
    and carry activation levels for spreading activation retrieval.
    """

    node_id: str = Field(..., description="Unique identifier for this node")
    node_type: NodeType = Field(..., description="Type of this node")
    
    # Human-readable label
    label: str = Field(
        default="unknown",
        description="Display label for this node",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Alternative labels",
    )
    
    # Embedding for similarity
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding for similarity search",
    )
    
    # Activation for spreading activation retrieval
    activation: float = Field(
        default=0.0,
        ge=0.0,
        description="Current activation level",
    )
    base_activation: float = Field(
        default=0.0,
        ge=0.0,
        description="Base-level activation (recency/frequency)",
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this node was created",
    )
    last_accessed: datetime = Field(
        default_factory=datetime.now,
        description="Last time this node was activated",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this node has been accessed",
    )
    
    # Reference to source object (episode_id, entity_id, etc.)
    source_id: str | None = Field(
        default=None,
        description="ID of the source object this node represents",
    )
    
    # Confidence in node identity
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this node's identity",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}


class GraphEdge(BaseModel):
    """An edge connecting two nodes in the associative graph.
    
    Edges carry weight for spreading activation and can be typed
    to support different kinds of relationships.
    """

    edge_id: str = Field(..., description="Unique identifier for this edge")
    edge_type: EdgeType = Field(..., description="Type of relationship")
    
    # Connected nodes
    source_node_id: str = Field(..., description="ID of the source node")
    target_node_id: str = Field(..., description="ID of the target node")
    
    # Weight for spreading activation
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Edge weight for activation spreading",
    )
    
    # Confidence in this relationship
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this relationship",
    )
    
    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this edge was created",
    )
    last_accessed: datetime = Field(
        default_factory=datetime.now,
        description="Last time this edge was traversed",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}
