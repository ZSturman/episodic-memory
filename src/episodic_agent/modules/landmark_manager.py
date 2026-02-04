"""Landmark Manager for relative coordinate system.

ARCHITECTURAL INVARIANT: The agent has no predefined knowledge of world coordinates.
Spatial understanding emerges from landmarks learned through user interaction.

This module provides:
- LandmarkManager: Manages learned landmarks and computes relative positions
- Automatic landmark suggestion based on stable observations
- User verification workflow for landmark confirmation
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from episodic_agent.schemas.spatial import (
    LandmarkReference,
    PositionObservation,
    RelativePosition,
    SpatialRelation,
    compute_bearing,
    compute_direction,
    compute_distance,
    classify_direction,
    classify_distance,
)

if TYPE_CHECKING:
    from episodic_agent.core.interfaces import GraphStore, DialogManager


logger = logging.getLogger(__name__)


# Node type for landmarks in the graph
NODE_TYPE_LANDMARK = "landmark"


class LandmarkManager:
    """Manages spatial landmarks and relative position computation.
    
    The LandmarkManager:
    1. Stores and retrieves learned landmarks
    2. Converts absolute positions to relative coordinates
    3. Suggests new landmarks based on stable observations
    4. Handles user verification of landmark labels
    
    ARCHITECTURAL INVARIANT: Landmark labels come from user interaction.
    The system tracks observable properties and prompts for labels when needed.
    """
    
    def __init__(
        self,
        graph_store: "GraphStore",
        dialog_manager: "DialogManager | None" = None,
        *,
        min_observations_for_suggestion: int = 5,
        max_landmarks_per_location: int = 10,
        stability_threshold_seconds: float = 300.0,  # 5 minutes
    ) -> None:
        """Initialize the landmark manager.
        
        Args:
            graph_store: Graph store for persisting landmarks.
            dialog_manager: Optional dialog manager for user interaction.
            min_observations_for_suggestion: How many times an entity must be
                seen before suggesting it as a landmark.
            max_landmarks_per_location: Maximum landmarks to track per location.
            stability_threshold_seconds: How long an entity must be stable to
                be considered a potential landmark.
        """
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        self._min_observations = min_observations_for_suggestion
        self._max_per_location = max_landmarks_per_location
        self._stability_threshold = timedelta(seconds=stability_threshold_seconds)
        
        # In-memory cache of landmarks (refreshed from graph store)
        self._landmarks: dict[str, LandmarkReference] = {}
        
        # Candidate landmarks (entities that might become landmarks)
        self._candidates: dict[str, _LandmarkCandidate] = {}
        
        # Load existing landmarks from graph
        self._load_landmarks()
    
    def _load_landmarks(self) -> None:
        """Load landmarks from the graph store."""
        try:
            nodes = self._graph_store.get_nodes_by_type(NODE_TYPE_LANDMARK)
            for node in nodes:
                landmark = LandmarkReference(
                    landmark_id=node.node_id,
                    label=node.label,
                    internal_position=node.extras.get("_internal_position"),
                    location_id=node.extras.get("location_id"),
                    location_label=node.extras.get("location_label"),
                    observation_count=node.extras.get("observation_count", 0),
                    is_static=node.extras.get("is_static", True),
                    user_verified=node.extras.get("user_verified", False),
                )
                self._landmarks[node.node_id] = landmark
            
            logger.info(f"Loaded {len(self._landmarks)} landmarks from graph store")
        except Exception as e:
            logger.warning(f"Failed to load landmarks: {e}")
    
    def get_landmark(self, landmark_id: str) -> LandmarkReference | None:
        """Get a landmark by ID."""
        return self._landmarks.get(landmark_id)
    
    def get_landmarks_in_location(self, location_id: str) -> list[LandmarkReference]:
        """Get all landmarks in a specific location."""
        return [
            lm for lm in self._landmarks.values()
            if lm.location_id == location_id
        ]
    
    def get_all_landmarks(self) -> list[LandmarkReference]:
        """Get all known landmarks."""
        return list(self._landmarks.values())
    
    def add_landmark(
        self,
        landmark_id: str,
        label: str,
        position: tuple[float, float, float] | None = None,
        location_id: str | None = None,
        location_label: str | None = None,
        user_verified: bool = False,
    ) -> LandmarkReference:
        """Add a new landmark.
        
        Args:
            landmark_id: Unique identifier for the landmark.
            label: Human-readable label (from user or learned).
            position: Optional raw position (for internal calculations).
            location_id: Optional room/zone GUID.
            location_label: Optional room/zone label.
            user_verified: Whether the user has verified this landmark.
            
        Returns:
            The created LandmarkReference.
        """
        landmark = LandmarkReference(
            landmark_id=landmark_id,
            label=label,
            internal_position=position,
            location_id=location_id,
            location_label=location_label,
            observation_count=1,
            last_observed=datetime.now(),
            is_static=True,
            user_verified=user_verified,
        )
        
        self._landmarks[landmark_id] = landmark
        
        # Persist to graph store
        self._persist_landmark(landmark)
        
        logger.info(f"Added landmark: {label} ({landmark_id})")
        return landmark
    
    def _persist_landmark(self, landmark: LandmarkReference) -> None:
        """Persist a landmark to the graph store."""
        from episodic_agent.schemas import GraphNode
        
        node = GraphNode(
            node_id=landmark.landmark_id,
            node_type=NODE_TYPE_LANDMARK,
            label=landmark.label,
            extras={
                "internal_position": landmark.internal_position,
                "location_id": landmark.location_id,
                "location_label": landmark.location_label,
                "observation_count": landmark.observation_count,
                "is_static": landmark.is_static,
                "user_verified": landmark.user_verified,
                "last_observed": (
                    landmark.last_observed.isoformat()
                    if landmark.last_observed else None
                ),
            },
        )
        
        try:
            self._graph_store.add_node(node)
        except Exception as e:
            logger.warning(f"Failed to persist landmark: {e}")
    
    def update_landmark_observation(
        self,
        landmark_id: str,
        position: tuple[float, float, float] | None = None,
    ) -> None:
        """Update a landmark's observation count and last seen time.
        
        Args:
            landmark_id: ID of the landmark.
            position: Optional updated position.
        """
        landmark = self._landmarks.get(landmark_id)
        if not landmark:
            return
        
        landmark.observation_count += 1
        landmark.last_observed = datetime.now()
        
        if position is not None:
            landmark.internal_position = position
        
        # Persist update
        self._persist_landmark(landmark)
    
    def compute_relative_position(
        self,
        raw_position: tuple[float, float, float],
        location_id: str | None = None,
        max_landmarks: int = 3,
    ) -> PositionObservation:
        """Convert raw coordinates to relative positions.
        
        Computes position relative to known landmarks in the same location.
        
        Args:
            raw_position: Raw (x, y, z) coordinates from sensor.
            location_id: Optional location to limit landmark search.
            max_landmarks: Maximum number of landmarks to compute relations for.
            
        Returns:
            PositionObservation with relative positions to landmarks.
        """
        # Get relevant landmarks
        if location_id:
            landmarks = self.get_landmarks_in_location(location_id)
        else:
            landmarks = self.get_all_landmarks()
        
        # Filter to landmarks with known positions
        landmarks_with_pos = [
            lm for lm in landmarks
            if lm.internal_position is not None
        ]
        
        # Sort by distance
        landmarks_by_distance = sorted(
            landmarks_with_pos,
            key=lambda lm: compute_distance(raw_position, lm.internal_position),
        )
        
        # Compute relative positions to nearest landmarks
        relative_positions: list[RelativePosition] = []
        
        for landmark in landmarks_by_distance[:max_landmarks]:
            lm_pos = landmark.internal_position
            
            distance = compute_distance(raw_position, lm_pos)
            direction = compute_direction(lm_pos, raw_position)
            bearing, elevation = compute_bearing(lm_pos, raw_position)
            relation = classify_distance(distance)
            
            rel_pos = RelativePosition(
                landmark_id=landmark.landmark_id,
                landmark_label=landmark.label,
                distance=distance,
                bearing=bearing,
                elevation=elevation,
                direction=direction,
                relation=relation,
                confidence=0.9 if landmark.user_verified else 0.7,
            )
            relative_positions.append(rel_pos)
        
        # Determine primary landmark (nearest verified, or just nearest)
        primary_landmark = None
        for lm in landmarks_by_distance:
            if lm.user_verified:
                primary_landmark = lm
                break
        if not primary_landmark and landmarks_by_distance:
            primary_landmark = landmarks_by_distance[0]
        
        # Build qualitative description
        qualitative = None
        if relative_positions:
            first = relative_positions[0]
            direction_word = classify_direction(first.bearing or 0, first.elevation)
            qualitative = f"{first.relation} the {first.landmark_label}, {direction_word}"
        
        return PositionObservation(
            raw_position=raw_position,
            relative_positions=relative_positions,
            primary_landmark_id=primary_landmark.landmark_id if primary_landmark else None,
            primary_landmark_label=primary_landmark.label if primary_landmark else None,
            qualitative_position=qualitative,
            location_id=location_id,
        )
    
    def compute_spatial_relation(
        self,
        subject_id: str,
        subject_position: tuple[float, float, float],
        reference_id: str,
        reference_position: tuple[float, float, float],
        subject_label: str | None = None,
        reference_label: str | None = None,
    ) -> SpatialRelation:
        """Compute the spatial relation between two entities.
        
        Args:
            subject_id: ID of the entity being described.
            subject_position: Position of the subject.
            reference_id: ID of the reference entity.
            reference_position: Position of the reference.
            subject_label: Optional label for the subject.
            reference_label: Optional label for the reference.
            
        Returns:
            SpatialRelation describing how subject relates to reference.
        """
        distance = compute_distance(subject_position, reference_position)
        bearing, elevation = compute_bearing(reference_position, subject_position)
        
        # Combine distance and direction into a relation
        dist_relation = classify_distance(distance)
        dir_relation = classify_direction(bearing, elevation)
        
        # Choose primary relation based on distance
        if distance < 2.0:
            relation = f"{dist_relation}_{dir_relation}"  # e.g., "near_left"
        else:
            relation = dir_relation  # Just use direction for far objects
        
        return SpatialRelation(
            subject_id=subject_id,
            subject_label=subject_label,
            reference_id=reference_id,
            reference_label=reference_label,
            relation=relation,
            distance=distance,
            source="computed",
            confidence=0.85,
        )
    
    def record_entity_observation(
        self,
        entity_id: str,
        position: tuple[float, float, float],
        label: str | None = None,
        location_id: str | None = None,
    ) -> bool:
        """Record an observation of an entity for potential landmark candidacy.
        
        Tracks entities that are observed repeatedly in stable positions,
        suggesting them as landmarks when threshold is met.
        
        Args:
            entity_id: ID of the observed entity.
            position: Position where entity was observed.
            label: Optional label if known.
            location_id: Optional location context.
            
        Returns:
            True if this entity should be suggested as a landmark.
        """
        now = datetime.now()
        
        if entity_id not in self._candidates:
            self._candidates[entity_id] = _LandmarkCandidate(
                entity_id=entity_id,
                first_position=position,
                last_position=position,
                label=label,
                location_id=location_id,
                first_seen=now,
                last_seen=now,
                observation_count=1,
            )
            return False
        
        candidate = self._candidates[entity_id]
        candidate.observation_count += 1
        candidate.last_seen = now
        
        # Check if position has remained stable
        movement = compute_distance(candidate.first_position, position)
        if movement > 1.0:  # Moved too much, reset
            candidate.first_position = position
            candidate.first_seen = now
            return False
        
        candidate.last_position = position
        
        # Check if ready to suggest as landmark
        time_stable = now - candidate.first_seen
        if (
            candidate.observation_count >= self._min_observations
            and time_stable >= self._stability_threshold
            and entity_id not in self._landmarks
        ):
            return True
        
        return False
    
    def suggest_landmark(
        self,
        entity_id: str,
        default_label: str | None = None,
    ) -> LandmarkReference | None:
        """Suggest an entity as a landmark and prompt user for label.
        
        Args:
            entity_id: ID of the entity to suggest.
            default_label: Default label to suggest to user.
            
        Returns:
            The created landmark if user confirms, None otherwise.
        """
        candidate = self._candidates.get(entity_id)
        if not candidate:
            return None
        
        # If we have a dialog manager, ask the user
        if self._dialog_manager:
            label = self._dialog_manager.request_label(
                f"Would you like to mark this as a landmark? (seen {candidate.observation_count} times)",
                [default_label or candidate.label or "unknown", "skip"],
            )
            
            if label is None or label == "skip":
                # User declined
                del self._candidates[entity_id]
                return None
        else:
            # No dialog manager, use default label
            label = default_label or candidate.label or f"landmark_{entity_id[:8]}"
        
        # Create the landmark
        landmark = self.add_landmark(
            landmark_id=entity_id,
            label=label,
            position=candidate.last_position,
            location_id=candidate.location_id,
            user_verified=self._dialog_manager is not None,
        )
        
        # Remove from candidates
        del self._candidates[entity_id]
        
        return landmark
    
    def verify_landmark(self, landmark_id: str, label: str | None = None) -> bool:
        """Mark a landmark as user-verified.
        
        Args:
            landmark_id: ID of the landmark.
            label: Optional new label from user.
            
        Returns:
            True if landmark was found and updated.
        """
        landmark = self._landmarks.get(landmark_id)
        if not landmark:
            return False
        
        landmark.user_verified = True
        if label:
            landmark.label = label
        
        self._persist_landmark(landmark)
        return True
    
    def remove_landmark(self, landmark_id: str) -> bool:
        """Remove a landmark.
        
        Args:
            landmark_id: ID of the landmark to remove.
            
        Returns:
            True if landmark was found and removed.
        """
        if landmark_id not in self._landmarks:
            return False
        
        del self._landmarks[landmark_id]
        
        # Note: We don't remove from graph store to maintain history
        # Could mark as "removed" in extras if needed
        
        logger.info(f"Removed landmark: {landmark_id}")
        return True


class _LandmarkCandidate:
    """Internal tracking for potential landmarks."""
    
    def __init__(
        self,
        entity_id: str,
        first_position: tuple[float, float, float],
        last_position: tuple[float, float, float],
        label: str | None,
        location_id: str | None,
        first_seen: datetime,
        last_seen: datetime,
        observation_count: int,
    ) -> None:
        self.entity_id = entity_id
        self.first_position = first_position
        self.last_position = last_position
        self.label = label
        self.location_id = location_id
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.observation_count = observation_count
