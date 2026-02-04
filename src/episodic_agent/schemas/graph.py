"""Graph memory data contracts for associative retrieval.

ARCHITECTURAL INVARIANT: No predefined semantic categories.
Node types and edge types are structural, not semantic.
All semantic labels are learned from user interaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from episodic_agent.schemas.salience import SalienceWeights

from pydantic import BaseModel, Field


# =============================================================================
# STRUCTURAL TYPE DEFINITIONS (Extensible Strings)
# =============================================================================
# 
# ARCHITECTURAL INVARIANT: These define graph STRUCTURE, not semantics.
# Semantic meaning is learned from user interaction and stored in labels.
# 
# Node types describe what kind of memory node this is (structural).
# Edge types describe how nodes relate (structural relationships).
# =============================================================================

# Structural node types (what kind of memory node)
NODE_TYPE_CONTEXT = "context"      # Context/episode hub
NODE_TYPE_CUE = "cue"              # Retrieval cue
NODE_TYPE_LOCATION = "location"    # Spatial region
NODE_TYPE_ENTITY = "entity"        # Observable thing
NODE_TYPE_EVENT = "event"          # Temporal occurrence
NODE_TYPE_EPISODE = "episode"      # Memory episode
NODE_TYPE_GOAL = "goal"            # Agent goal
NODE_TYPE_CONCEPT = "concept"      # Abstract concept (learned)

# Structural edge types (how nodes relate)
EDGE_TYPE_CUE_OF = "cue_of"              # Cue retrieves context
EDGE_TYPE_IN_CONTEXT = "in_context"      # Entity/event in context
EDGE_TYPE_TYPICAL_IN = "typical_in"      # Entity typical in location
EDGE_TYPE_INVOLVES = "involves"          # Event involves entity
EDGE_TYPE_IN_EVENT = "in_event"          # Entity participated in event
EDGE_TYPE_TRIGGERED_BY = "triggered_by"  # Event triggered by delta
EDGE_TYPE_OCCURRED_IN = "occurred_in"    # Event occurred in location
EDGE_TYPE_ALIAS_OF = "alias_of"          # Label alias
EDGE_TYPE_MERGED_INTO = "merged_into"    # Node merged
EDGE_TYPE_CONTAINS = "contains"          # Location contains entity
EDGE_TYPE_OCCURRED_AT = "occurred_at"    # Legacy
EDGE_TYPE_PART_OF = "part_of"            # Part of episode
EDGE_TYPE_SIMILAR_TO = "similar_to"      # Similarity link
EDGE_TYPE_TEMPORAL = "temporal"          # Temporal sequence
EDGE_TYPE_CAUSAL = "causal"              # Causal relationship
EDGE_TYPE_REVISIT = "revisit"            # Revisit link for cued recall
EDGE_TYPE_SPATIAL = "spatial"            # Spatial relationship (relative position)


# Legacy class aliases for backward compatibility
class NodeType:
    """Legacy node type constants for backward compatibility."""
    CONTEXT = NODE_TYPE_CONTEXT
    CUE = NODE_TYPE_CUE
    LOCATION = NODE_TYPE_LOCATION
    ENTITY = NODE_TYPE_ENTITY
    EVENT = NODE_TYPE_EVENT
    EPISODE = NODE_TYPE_EPISODE
    GOAL = NODE_TYPE_GOAL
    CONCEPT = NODE_TYPE_CONCEPT


class EdgeType:
    """Legacy edge type constants for backward compatibility."""
    CUE_OF = EDGE_TYPE_CUE_OF
    IN_CONTEXT = EDGE_TYPE_IN_CONTEXT
    TYPICAL_IN = EDGE_TYPE_TYPICAL_IN
    INVOLVES = EDGE_TYPE_INVOLVES
    IN_EVENT = EDGE_TYPE_IN_EVENT
    TRIGGERED_BY = EDGE_TYPE_TRIGGERED_BY
    OCCURRED_IN = EDGE_TYPE_OCCURRED_IN
    ALIAS_OF = EDGE_TYPE_ALIAS_OF
    MERGED_INTO = EDGE_TYPE_MERGED_INTO
    CONTAINS = EDGE_TYPE_CONTAINS
    OCCURRED_AT = EDGE_TYPE_OCCURRED_AT
    PART_OF = EDGE_TYPE_PART_OF
    SIMILAR_TO = EDGE_TYPE_SIMILAR_TO
    TEMPORAL = EDGE_TYPE_TEMPORAL
    CAUSAL = EDGE_TYPE_CAUSAL


class GraphNode(BaseModel):
    """A node in the associative graph memory.
    
    Nodes represent locations, entities, events, episodes, or concepts
    and carry activation levels for spreading activation retrieval.
    
    ARCHITECTURAL INVARIANT: node_type is structural, not semantic.
    Semantic meaning is stored in label/labels fields, learned from user.
    """

    node_id: str = Field(..., description="Unique identifier for this node")
    node_type: str = Field(..., description="Structural type of this node")
    
    # Human-readable labels (learned from user, not predefined)
    label: str = Field(
        default="unknown",
        description="Primary label for this node (learned from user)",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Alternative/hierarchical labels (e.g., ['bedroom', 'room'])",
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
    
    ARCHITECTURAL INVARIANT: edge_type is structural, not semantic.
    Salience weights are learned from experience, not predefined.
    """

    edge_id: str = Field(..., description="Unique identifier for this edge")
    edge_type: str = Field(..., description="Structural type of relationship")
    
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
    
    # Salience weights for cued recall (Phase 6)
    # Stored as dict to avoid circular import; use SalienceWeights for manipulation
    salience: dict[str, float] = Field(
        default_factory=lambda: {
            "location": 0.0,
            "entity": 0.0,
            "temporal": 0.0,
            "semantic": 0.0,
            "visual": 0.0,
            "event": 0.0,
            "overall": 0.0,
        },
        description="Per-cue-type salience weights for this edge",
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
    
    def get_salience_score(self, cue_type: str) -> float:
        """Get salience score for a specific cue type."""
        return self.salience.get(cue_type, 0.0)
    
    def boost_salience(self, cue_type: str, amount: float = 0.1) -> None:
        """Boost salience for a cue type (learning)."""
        if cue_type in self.salience:
            self.salience[cue_type] = min(1.0, self.salience[cue_type] + amount)
            self._update_overall_salience()
    
    def _update_overall_salience(self) -> None:
        """Update overall salience from individual cue weights."""
        weights = [
            self.salience.get("location", 0.0),
            self.salience.get("entity", 0.0),
            self.salience.get("temporal", 0.0),
            self.salience.get("semantic", 0.0),
            self.salience.get("visual", 0.0),
            self.salience.get("event", 0.0),
        ]
        self.salience["overall"] = max(weights) if weights else 0.0
