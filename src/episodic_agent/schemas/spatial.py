"""Spatial schemas for relative coordinate system.

ARCHITECTURAL INVARIANT: The agent has no predefined knowledge of world coordinates.
All spatial understanding emerges from learned landmarks and relative observations.

This module provides:
- RelativePosition: Position expressed relative to a known landmark
- SpatialRelation: Qualitative spatial relationship between entities
- LandmarkReference: A learned spatial reference point

The system converts raw sensor coordinates to relative positions
using landmarks learned through user interaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# SPATIAL RELATION TYPES
# =============================================================================
# 
# These are structural descriptions of spatial relationships.
# More semantic relations (e.g., "on the table", "inside the cabinet")
# are learned from user interaction.
# =============================================================================

# Distance-based relations
RELATION_NEAR = "near"              # Within close proximity
RELATION_FAR = "far"                # Beyond close proximity
RELATION_AT = "at"                  # Essentially at the same location

# Direction-based relations (from observer's perspective)
RELATION_LEFT = "left"
RELATION_RIGHT = "right"
RELATION_FRONT = "front"
RELATION_BEHIND = "behind"
RELATION_ABOVE = "above"
RELATION_BELOW = "below"

# Containment relations
RELATION_INSIDE = "inside"
RELATION_OUTSIDE = "outside"


class RelativePosition(BaseModel):
    """Position expressed relative to a known landmark.
    
    Instead of absolute world coordinates, positions are stored as:
    - A reference to a learned landmark
    - Distance from that landmark
    - Direction/bearing from that landmark
    
    This allows spatial understanding to emerge from experience
    rather than relying on predefined coordinate systems.
    """
    
    # Reference landmark
    landmark_id: str = Field(
        ..., 
        description="Node ID of the reference landmark in the graph",
    )
    landmark_label: str | None = Field(
        default=None,
        description="Human-readable label of the landmark (learned)",
    )
    
    # Distance from landmark
    distance: float = Field(
        default=0.0,
        ge=0.0,
        description="Distance from the landmark in world units",
    )
    
    # Direction from landmark (relative to landmark's forward or world axes)
    bearing: float | None = Field(
        default=None,
        description="Horizontal angle from landmark in degrees (0=forward, 90=right)",
    )
    elevation: float | None = Field(
        default=None,
        description="Vertical angle from landmark in degrees (positive=up)",
    )
    
    # Alternative: direction vector (normalized)
    direction: tuple[float, float, float] | None = Field(
        default=None,
        description="Unit direction vector from landmark to this position",
    )
    
    # Qualitative relation as fallback
    relation: str | None = Field(
        default=None,
        description="Qualitative spatial relation (near, far, left, right, etc.)",
    )
    
    # Confidence in this relative position
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the relative position measurement",
    )
    
    # Timestamp of measurement
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this relative position was computed",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class LandmarkReference(BaseModel):
    """A learned spatial reference point (landmark).
    
    Landmarks are entities or locations that serve as reference points
    for relative positioning. They are learned through:
    - User labeling ("this is the kitchen table")
    - Repeated observation (stable entities become implicit landmarks)
    - Explicit marking during exploration
    
    ARCHITECTURAL INVARIANT: Landmark semantics come from user interaction.
    The system only stores observable properties and learned labels.
    """
    
    landmark_id: str = Field(
        ..., 
        description="Unique identifier (typically a graph node ID)",
    )
    
    # Learned identity
    label: str = Field(
        default="unknown",
        description="User-provided or learned label for this landmark",
    )
    
    # The raw coordinates are stored but ONLY used internally
    # for computing relative positions between landmarks
    internal_position: tuple[float, float, float] | None = Field(
        default=None,
        description="Internal use only: raw position for relative calculations",
    )
    
    # Location context (which room/zone this landmark is in)
    location_id: str | None = Field(
        default=None,
        description="Room/zone GUID where this landmark is located",
    )
    location_label: str | None = Field(
        default=None,
        description="Learned label for the location",
    )
    
    # Stability metrics (more stable = better landmark)
    observation_count: int = Field(
        default=0,
        ge=0,
        description="How many times this landmark has been observed",
    )
    last_observed: datetime | None = Field(
        default=None,
        description="When this landmark was last seen",
    )
    is_static: bool = Field(
        default=True,
        description="Whether this landmark is expected to stay in place",
    )
    
    # User verification
    user_verified: bool = Field(
        default=False,
        description="Whether a user has confirmed this as a landmark",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class SpatialRelation(BaseModel):
    """A qualitative spatial relationship between two entities.
    
    Used when precise distance isn't meaningful or available.
    Can be learned from user descriptions or inferred from observations.
    """
    
    # The two entities in the relation
    subject_id: str = Field(..., description="Entity being described")
    subject_label: str | None = Field(default=None)
    
    reference_id: str = Field(..., description="Reference entity")
    reference_label: str | None = Field(default=None)
    
    # The relation itself
    relation: str = Field(
        ...,
        description="Spatial relation (near, far, left, inside, on, etc.)",
    )
    
    # Optional quantification
    distance: float | None = Field(
        default=None,
        description="Distance if measurable",
    )
    
    # How this relation was determined
    source: str = Field(
        default="inferred",
        description="How this relation was determined (user, inferred, observed)",
    )
    
    # Confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this spatial relation",
    )
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this relation was established",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class LocationFingerprint(BaseModel):
    """Fingerprint for a discovered location boundary.
    
    Represents a spatial region discovered through perception-based
    scene fingerprinting. Unlike cheat resolvers that use Unity GUIDs,
    this schema captures what the agent *actually observes* at a location.
    
    ARCHITECTURAL INVARIANT: No predefined semantics. Location boundaries
    are discovered from statistical regularities in the scene embedding
    stream. Labels come from user interaction only.
    """
    
    location_id: str = Field(
        ..., 
        description="Unique identifier for this discovered location",
    )
    
    # Labels — structured parent + variant
    parent_label: str = Field(
        default="unknown",
        description="Primary location label (e.g. 'office', 'kitchen')",
    )
    variant_label: str = Field(
        default="",
        description="Sub-label variant (e.g. 'corner with windows', 'near fridge')",
    )
    
    # Centroid fingerprint (average scene embedding across observations)
    centroid_embedding: list[float] = Field(
        default_factory=list,
        description="Average scene embedding for this location (centroid)",
    )
    
    # Hex reconstruction data — saved per-cell features for visual rebuild
    hex_reconstruction_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-hex-cell feature data for location reconstruction",
    )
    
    # Observation statistics
    observation_count: int = Field(
        default=0,
        ge=0,
        description="Number of frames observed at this location",
    )
    embedding_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Variance of scene embeddings (measures visual diversity)",
    )
    
    # Transition tracking
    transition_positions: list[tuple[float, float, float]] = Field(
        default_factory=list,
        description="Agent positions (raw) at detected transitions into/out of this location",
    )
    approximate_center: tuple[float, float, float] | None = Field(
        default=None,
        description="Approximate center of this region (internal use for relative coords)",
    )
    approximate_radius: float = Field(
        default=0.0,
        ge=0.0,
        description="Approximate extent of this region in world units",
    )
    
    # Entity co-occurrence (what entities have been seen here)
    entity_guids_seen: list[str] = Field(
        default_factory=list,
        description="GUIDs of entities observed at this location",
    )
    entity_cooccurrence_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of observations per entity at this location",
    )
    
    # Timestamps
    first_visited: datetime = Field(
        default_factory=datetime.now,
        description="When this location was first discovered",
    )
    last_visited: datetime = Field(
        default_factory=datetime.now,
        description="Last time this location was observed",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class PositionObservation(BaseModel):
    """Raw position observation before conversion to relative coordinates.
    
    This stores the sensor-provided coordinates temporarily while
    computing relative positions against known landmarks.
    """
    
    # Raw observation
    raw_position: tuple[float, float, float] | None = Field(
        default=None,
        description="Raw sensor coordinates (internal use only)",
    )
    
    # Computed relative positions to known landmarks
    relative_positions: list[RelativePosition] = Field(
        default_factory=list,
        description="Positions relative to known landmarks",
    )
    
    # Primary landmark (closest or most relevant)
    primary_landmark_id: str | None = Field(
        default=None,
        description="ID of the primary reference landmark",
    )
    primary_landmark_label: str | None = Field(
        default=None,
        description="Label of the primary reference landmark",
    )
    
    # Qualitative description for when landmarks aren't available
    qualitative_position: str | None = Field(
        default=None,
        description="Qualitative position description (e.g., 'in the kitchen, near the door')",
    )
    
    # Context
    location_id: str | None = Field(
        default=None,
        description="Room/zone GUID if known",
    )
    location_label: str | None = Field(
        default=None,
        description="Room/zone label if known",
    )
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this observation was made",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_distance(
    pos1: tuple[float, float, float],
    pos2: tuple[float, float, float],
) -> float:
    """Compute Euclidean distance between two 3D positions."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def compute_direction(
    from_pos: tuple[float, float, float],
    to_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compute normalized direction vector from one position to another."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dz = to_pos[2] - from_pos[2]
    
    magnitude = (dx * dx + dy * dy + dz * dz) ** 0.5
    if magnitude < 1e-6:
        return (0.0, 0.0, 0.0)
    
    return (dx / magnitude, dy / magnitude, dz / magnitude)


def compute_bearing(
    from_pos: tuple[float, float, float],
    to_pos: tuple[float, float, float],
    forward_dir: tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    """Compute bearing (horizontal angle) and elevation (vertical angle).
    
    Args:
        from_pos: Origin position
        to_pos: Target position
        forward_dir: Forward direction vector (default is +Z)
        
    Returns:
        Tuple of (bearing_degrees, elevation_degrees)
    """
    import math
    
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dz = to_pos[2] - from_pos[2]
    
    # Horizontal distance
    horizontal_dist = (dx * dx + dz * dz) ** 0.5
    
    # Bearing (angle in XZ plane, 0 = forward/+Z)
    if forward_dir is None:
        # Default: +Z is forward
        bearing = math.degrees(math.atan2(dx, dz))
    else:
        # Compute relative to provided forward direction
        # This is a simplification - full implementation would use quaternions
        fwd_angle = math.atan2(forward_dir[0], forward_dir[2])
        target_angle = math.atan2(dx, dz)
        bearing = math.degrees(target_angle - fwd_angle)
    
    # Normalize to [-180, 180]
    while bearing > 180:
        bearing -= 360
    while bearing < -180:
        bearing += 360
    
    # Elevation (angle from horizontal)
    if horizontal_dist > 1e-6:
        elevation = math.degrees(math.atan2(dy, horizontal_dist))
    else:
        elevation = 90.0 if dy > 0 else -90.0 if dy < 0 else 0.0
    
    return (bearing, elevation)


def classify_distance(distance: float, thresholds: dict[str, float] | None = None) -> str:
    """Classify a distance into a qualitative relation.
    
    Args:
        distance: Distance in world units
        thresholds: Optional custom thresholds (default: near < 2.0, far > 10.0)
        
    Returns:
        Relation string: "at", "near", or "far"
    """
    if thresholds is None:
        thresholds = {"at": 0.5, "near": 2.0, "far": 10.0}
    
    if distance < thresholds.get("at", 0.5):
        return RELATION_AT
    elif distance < thresholds.get("near", 2.0):
        return RELATION_NEAR
    else:
        return RELATION_FAR


def classify_direction(
    bearing: float,
    elevation: float | None = None,
) -> str:
    """Classify bearing/elevation into a qualitative direction.
    
    Args:
        bearing: Horizontal angle in degrees (-180 to 180, 0 = forward)
        elevation: Vertical angle in degrees (optional)
        
    Returns:
        Relation string: front, behind, left, right, above, below
    """
    # Check elevation first if significant
    if elevation is not None:
        if elevation > 30:
            return RELATION_ABOVE
        elif elevation < -30:
            return RELATION_BELOW
    
    # Classify horizontal direction
    # Front: -45 to 45 degrees
    # Right: 45 to 135 degrees
    # Behind: 135 to 180 or -180 to -135 degrees
    # Left: -135 to -45 degrees
    
    abs_bearing = abs(bearing)
    if abs_bearing < 45:
        return RELATION_FRONT
    elif abs_bearing > 135:
        return RELATION_BEHIND
    elif bearing > 0:
        return RELATION_RIGHT
    else:
        return RELATION_LEFT
