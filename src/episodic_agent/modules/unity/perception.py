"""Cheat perception module for Unity integration.

Uses Unity's ground-truth data (GUIDs, room IDs) to generate
deterministic embeddings without any actual perception processing.
This enables testing the memory system with perfect perception.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime
from typing import Any

from episodic_agent.core.interfaces import PerceptionModule
from episodic_agent.schemas import ObjectCandidate, Percept, SensorFrame
from episodic_agent.utils.config import DEFAULT_EMBEDDING_DIM


def guid_to_embedding(guid: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Convert a GUID/string to a deterministic embedding vector.
    
    Uses SHA-256 hash to generate pseudo-random but repeatable values,
    ensuring the same GUID always maps to the same embedding.
    
    Args:
        guid: The GUID or string to convert.
        dim: Dimensionality of the output embedding.
        
    Returns:
        Normalized embedding vector of length `dim`.
    """
    # Hash the GUID
    hash_bytes = hashlib.sha256(guid.encode("utf-8")).digest()
    
    # Extend hash if needed for larger dimensions
    while len(hash_bytes) < dim * 4:  # 4 bytes per float seed
        hash_bytes += hashlib.sha256(hash_bytes).digest()
    
    # Convert bytes to floats in [-1, 1]
    embedding = []
    for i in range(dim):
        # Use 4 bytes per value
        value_bytes = hash_bytes[i * 4:(i + 1) * 4]
        # Convert to unsigned int, then to float in [0, 1]
        uint_value = int.from_bytes(value_bytes, "big")
        float_value = (uint_value / (2**32 - 1)) * 2 - 1  # Map to [-1, 1]
        embedding.append(float_value)
    
    # Normalize to unit length
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding


def compute_motion_features(
    current_pose: dict[str, Any] | None,
    previous_pose: dict[str, Any] | None,
) -> dict[str, float]:
    """Compute minimal motion features from camera pose delta.
    
    Args:
        current_pose: Current camera pose dict with position/rotation.
        previous_pose: Previous camera pose dict.
        
    Returns:
        Dict with motion features (linear_speed, angular_speed, etc.).
    """
    if not current_pose or not previous_pose:
        return {
            "linear_speed": 0.0,
            "angular_speed": 0.0,
            "moving": False,
        }
    
    # Extract positions
    curr_pos = current_pose.get("position", {})
    prev_pos = previous_pose.get("position", {})
    
    # Compute linear delta
    dx = curr_pos.get("x", 0) - prev_pos.get("x", 0)
    dy = curr_pos.get("y", 0) - prev_pos.get("y", 0)
    dz = curr_pos.get("z", 0) - prev_pos.get("z", 0)
    linear_speed = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Extract rotations
    curr_rot = current_pose.get("rotation", {})
    prev_rot = previous_pose.get("rotation", {})
    
    # Compute angular delta (Euler angles, simplified)
    drx = curr_rot.get("x", 0) - prev_rot.get("x", 0)
    dry = curr_rot.get("y", 0) - prev_rot.get("y", 0)
    drz = curr_rot.get("z", 0) - prev_rot.get("z", 0)
    angular_speed = math.sqrt(drx*drx + dry*dry + drz*drz)
    
    return {
        "linear_speed": linear_speed,
        "angular_speed": angular_speed,
        "moving": linear_speed > 0.01 or angular_speed > 0.1,
        "position_delta": {"x": dx, "y": dy, "z": dz},
        "rotation_delta": {"x": drx, "y": dry, "z": drz},
    }


class PerceptionUnityCheat(PerceptionModule):
    """Cheat perception module that uses Unity's ground-truth data.
    
    Instead of actual perception (computer vision, etc.), this module:
    - Uses room GUIDs to generate deterministic scene embeddings
    - Uses entity GUIDs to generate deterministic object embeddings
    - Passes through position/size/state as extras
    
    This enables testing the full memory system without real perception,
    with perfect object identity based on Unity's GUIDs.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        include_invisible: bool = False,
        min_confidence: float = 0.95,  # High confidence for cheat mode
    ) -> None:
        """Initialize the cheat perception module.
        
        Args:
            embedding_dim: Dimensionality of generated embeddings.
            include_invisible: Whether to include non-visible entities.
            min_confidence: Base confidence for cheat-detected objects.
        """
        self._embedding_dim = embedding_dim
        self._include_invisible = include_invisible
        self._min_confidence = min_confidence
        
        # Track previous frame for motion features
        self._previous_pose: dict[str, Any] | None = None
        self._previous_frame_id: int | None = None

    def process(self, frame: SensorFrame) -> Percept:
        """Process a sensor frame into a percept using Unity cheat data.
        
        Extracts room GUID and entity data from frame.extras (populated by
        UnityWebSocketSensorProvider) and generates deterministic embeddings.
        
        Args:
            frame: Raw sensor frame with Unity data in extras.
            
        Returns:
            Percept with scene embedding and object candidates.
        """
        extras = frame.extras or {}
        
        # Get room GUID for scene embedding
        room_guid = extras.get("current_room")
        room_label = extras.get("current_room_label", "unknown")
        
        # Generate scene embedding from room GUID
        if room_guid:
            scene_embedding = guid_to_embedding(room_guid, self._embedding_dim)
        else:
            # No room - generate neutral embedding
            scene_embedding = guid_to_embedding("__no_room__", self._embedding_dim)
        
        # Process entities into object candidates
        entities = extras.get("entities", [])
        candidates = self._process_entities(entities)
        
        # Compute motion features
        current_pose = extras.get("camera_pose")
        motion_features = compute_motion_features(current_pose, self._previous_pose)
        
        # Update previous pose for next frame
        self._previous_pose = current_pose
        self._previous_frame_id = frame.frame_id
        
        # Build percept
        percept = Percept(
            percept_id=f"perc_{uuid.uuid4().hex[:8]}",
            source_frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            scene_embedding=scene_embedding,
            candidates=candidates,
            confidence=self._min_confidence,  # High confidence in cheat mode
            extras={
                "room_guid": room_guid,
                "room_label": room_label,
                "motion_features": motion_features,
                "entity_count_total": len(entities),
                "entity_count_visible": sum(1 for e in entities if e.get("visible", True)),
                "state_changes": extras.get("state_changes", []),
            },
        )
        
        return percept

    def _process_entities(self, entities: list[dict[str, Any]]) -> list[ObjectCandidate]:
        """Convert Unity entity data to ObjectCandidates.
        
        Args:
            entities: List of entity dicts from Unity frame.
            
        Returns:
            List of ObjectCandidate instances.
        """
        candidates = []
        
        for entity in entities:
            # Skip invisible entities unless configured to include them
            if not self._include_invisible and not entity.get("visible", True):
                continue
            
            guid = entity.get("guid", "")
            if not guid:
                continue
            
            # Generate deterministic embedding from GUID
            embedding = guid_to_embedding(guid, self._embedding_dim)
            
            # Extract position
            pos = entity.get("position", {})
            position = (
                pos.get("x", 0.0),
                pos.get("y", 0.0),
                pos.get("z", 0.0),
            ) if pos else None
            
            # Compute confidence based on distance and visibility
            distance = entity.get("distance", 0.0)
            visible = entity.get("visible", True)
            
            # Higher confidence for closer, visible objects
            confidence = self._min_confidence
            if distance > 0:
                # Reduce confidence slightly with distance
                confidence = max(0.5, self._min_confidence - (distance * 0.02))
            if not visible:
                confidence *= 0.5
            
            candidate = ObjectCandidate(
                candidate_id=guid,  # Use GUID as candidate ID for easy matching
                label=entity.get("label", "unknown"),
                labels=[entity.get("label", "unknown")] if entity.get("label") else [],
                confidence=confidence,
                embedding=embedding,
                position=position,
                bounding_box=None,  # Unity doesn't provide bounding boxes
                extras={
                    "guid": guid,
                    "category": entity.get("category", "unknown"),
                    "state": entity.get("state", "default"),
                    "distance": distance,
                    "visible": visible,
                    "room_guid": entity.get("room_guid"),
                    "rotation": entity.get("rotation"),
                },
            )
            
            candidates.append(candidate)
        
        return candidates
