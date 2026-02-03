"""Retrieval and step result data contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from episodic_agent.schemas.context import Episode
from episodic_agent.schemas.graph import GraphNode
from episodic_agent.utils.config import LOG_VERSION


class RetrievalResult(BaseModel):
    """Result from querying episodic or graph memory.
    
    Contains ranked results with relevance scores from various
    retrieval methods (spreading activation, vector similarity, etc.).
    """

    query_id: str = Field(..., description="Unique identifier for this query")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this query was executed",
    )
    
    # Retrieved episodes ranked by relevance
    episodes: list[Episode] = Field(
        default_factory=list,
        description="Retrieved episodes ranked by relevance",
    )
    episode_scores: list[float] = Field(
        default_factory=list,
        description="Relevance scores for retrieved episodes",
    )
    
    # Retrieved graph nodes (for spreading activation)
    nodes: list[GraphNode] = Field(
        default_factory=list,
        description="Retrieved graph nodes ranked by activation",
    )
    node_scores: list[float] = Field(
        default_factory=list,
        description="Activation scores for retrieved nodes",
    )
    
    # Query metadata
    retrieval_method: str = Field(
        default="unknown",
        description="Method used for retrieval",
    )
    query_embedding: list[float] | None = Field(
        default=None,
        description="Query embedding if vector search was used",
    )
    
    # Performance metrics
    retrieval_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for retrieval in milliseconds",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}


class MemoryCounts(BaseModel):
    """Counts of memory structures for logging."""
    
    episodes: int = Field(default=0, ge=0, description="Number of stored episodes")
    nodes: int = Field(default=0, ge=0, description="Number of graph nodes")
    edges: int = Field(default=0, ge=0, description="Number of graph edges")

    model_config = {"frozen": True}


class StepResult(BaseModel):
    """Result from a single orchestrator step - used for logging.
    
    Contains all information needed for JSONL logging with a stable
    record schema. This is the single source of truth for log records.
    """

    # Log metadata
    log_version: str = Field(
        default=LOG_VERSION,
        description="Version of the log schema",
    )
    run_id: str = Field(..., description="Identifier for this run")
    
    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this step completed",
    )
    step_number: int = Field(..., description="Step number in the run")
    
    # Frame information
    frame_id: int = Field(..., description="ID of the processed sensor frame")
    
    # ACF state
    acf_id: str = Field(..., description="ID of the current ACF")
    location_label: str = Field(..., description="Current location label")
    location_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in location",
    )
    
    # Counts
    entity_count: int = Field(
        default=0,
        ge=0,
        description="Number of entities in context",
    )
    event_count: int = Field(
        default=0,
        ge=0,
        description="Number of events detected this episode",
    )
    episode_count: int = Field(
        default=0,
        ge=0,
        description="Total episodes frozen so far",
    )
    
    # Boundary information
    boundary_triggered: bool = Field(
        default=False,
        description="Whether an episode boundary was triggered",
    )
    boundary_reason: str | None = Field(
        default=None,
        description="Reason for boundary if triggered",
    )
    
    # Memory counts (Phase 2+)
    memory_counts: MemoryCounts | None = Field(
        default=None,
        description="Counts of memory structures",
    )
    
    # Label events this step (Phase 2+)
    label_assignments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Label assignments made this step",
    )
    conflicts_created: list[str] = Field(
        default_factory=list,
        description="IDs of label conflicts created this step",
    )
    conflicts_resolved: list[str] = Field(
        default_factory=list,
        description="IDs of label conflicts resolved this step",
    )
    
    # Forward-compatible extras (for Unity fields, etc.)
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": True}  # StepResults are immutable records

    def to_log_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for JSONL logging.
        
        Ensures consistent field ordering and ISO timestamp format.
        """
        result = {
            "log_version": self.log_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "step_number": self.step_number,
            "frame_id": self.frame_id,
            "acf_id": self.acf_id,
            "location_label": self.location_label,
            "location_confidence": self.location_confidence,
            "entity_count": self.entity_count,
            "event_count": self.event_count,
            "episode_count": self.episode_count,
            "boundary_triggered": self.boundary_triggered,
            "boundary_reason": self.boundary_reason,
        }
        
        # Add memory counts if present
        if self.memory_counts:
            result["memory_counts"] = {
                "episodes": self.memory_counts.episodes,
                "nodes": self.memory_counts.nodes,
                "edges": self.memory_counts.edges,
            }
        
        # Add label events if any
        if self.label_assignments:
            result["label_assignments"] = self.label_assignments
        if self.conflicts_created:
            result["conflicts_created"] = self.conflicts_created
        if self.conflicts_resolved:
            result["conflicts_resolved"] = self.conflicts_resolved
            
        result["extras"] = self.extras
        return result
