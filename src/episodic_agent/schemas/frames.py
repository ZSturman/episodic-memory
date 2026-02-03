"""Sensor frame, percept, and object candidate data contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SensorFrame(BaseModel):
    """Raw sensor input from any source (Unity, file, synthetic).
    
    Represents a single timestep of sensor data before perception processing.
    """

    frame_id: int = Field(..., description="Monotonically increasing frame identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this frame was captured/generated",
    )
    
    # Raw sensor data - structure depends on sensor type
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw sensor readings (images, depth, audio, etc.)",
    )
    
    # Sensor metadata
    sensor_type: str = Field(
        default="unknown",
        description="Type of sensor that produced this frame",
    )
    
    # Forward-compatible extras for Unity cheat fields, etc.
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}


class ObjectCandidate(BaseModel):
    """A recognized object with identification confidence.
    
    Represents a potential object detected in perception, before
    full entity resolution.
    """

    candidate_id: str = Field(..., description="Unique identifier for this candidate")
    
    # Recognition
    label: str = Field(
        default="unknown",
        description="Best-guess label for this object",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Alternative label candidates",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Recognition confidence [0, 1]",
    )
    
    # Embedding for similarity matching
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding for similarity search",
    )
    
    # Spatial information (if available)
    position: tuple[float, float, float] | None = Field(
        default=None,
        description="3D position (x, y, z) if known",
    )
    bounding_box: dict[str, float] | None = Field(
        default=None,
        description="Bounding box coordinates if available",
    )
    
    # For later conflict resolution
    conflict_id: str | None = Field(
        default=None,
        description="ID linking to a label conflict if present",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}


class Percept(BaseModel):
    """Processed perception output from a sensor frame.
    
    Contains recognized objects and a scene-level embedding.
    """

    percept_id: str = Field(..., description="Unique identifier for this percept")
    source_frame_id: int = Field(..., description="ID of the source SensorFrame")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this percept was generated",
    )
    
    # Scene-level embedding
    scene_embedding: list[float] | None = Field(
        default=None,
        description="Overall scene embedding for similarity search",
    )
    
    # Detected objects
    candidates: list[ObjectCandidate] = Field(
        default_factory=list,
        description="Object candidates detected in this percept",
    )
    
    # Scene-level confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall perception confidence [0, 1]",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )

    model_config = {"frozen": False}
