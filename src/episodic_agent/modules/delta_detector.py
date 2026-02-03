"""Delta detection module for tracking changes between steps.

Computes deltas (changes) by comparing current perception state
to recent context history:
- new_entity: Entity present now, not in recent context
- missing_entity: Entity was in recent context, now absent
- moved_entity: Entity position changed significantly
- state_changed: Entity interactable state changed
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from episodic_agent.schemas.events import Delta, DeltaType
from episodic_agent.utils.config import DEFAULT_MOVE_THRESHOLD

if TYPE_CHECKING:
    from episodic_agent.schemas import ActiveContextFrame, ObjectCandidate, Percept

logger = logging.getLogger(__name__)


class DeltaDetector:
    """Detects changes (deltas) between consecutive perception steps.
    
    Tracks entity state over time and identifies:
    - Entities appearing/disappearing
    - Entity movement beyond threshold
    - State changes (open/closed, on/off)
    """
    
    def __init__(
        self,
        move_threshold: float = DEFAULT_MOVE_THRESHOLD,
        missing_window: int = 3,  # Steps before declaring missing
    ) -> None:
        """Initialize the delta detector.
        
        Args:
            move_threshold: Distance threshold for movement detection.
            missing_window: Steps entity must be absent to be "missing".
        """
        self._move_threshold = move_threshold
        self._missing_window = missing_window
        
        # Track entity history by GUID
        self._entity_history: dict[str, list[EntitySnapshot]] = {}
        
        # Track which entities were visible last step
        self._last_visible_guids: set[str] = set()
        
        # Track pending missing detections (guid -> steps absent)
        self._pending_missing: dict[str, int] = {}
        
        # Statistics
        self._total_deltas_detected = 0

    def detect(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[Delta]:
        """Detect deltas from current perception compared to context.
        
        Args:
            percept: Current perception with entity candidates.
            acf: Current active context frame.
            
        Returns:
            List of detected deltas.
        """
        deltas: list[Delta] = []
        step_number = acf.step_count
        location_label = acf.location_label
        
        # Extract current visible entities
        current_entities = self._extract_entities(percept)
        current_guids = set(current_entities.keys())
        
        # Detect new entities
        new_guids = current_guids - self._last_visible_guids
        for guid in new_guids:
            delta = self._create_new_entity_delta(
                guid, current_entities[guid], step_number, location_label
            )
            if delta:
                deltas.append(delta)
                # Remove from pending missing if it was there
                self._pending_missing.pop(guid, None)
        
        # Detect potentially missing entities
        # Check both: entities that just left AND entities that are still pending
        absent_guids = self._last_visible_guids - current_guids
        
        # Add entities that just left to pending
        for guid in absent_guids:
            self._pending_missing[guid] = self._pending_missing.get(guid, 0) + 1
        
        # Increment counter for already-pending entities that are still absent
        for guid in list(self._pending_missing.keys()):
            if guid not in current_guids and guid not in absent_guids:
                # Was already pending and still absent
                self._pending_missing[guid] += 1
        
        # Check which pending entities have reached the window
        for guid in list(self._pending_missing.keys()):
            if self._pending_missing[guid] >= self._missing_window:
                delta = self._create_missing_entity_delta(
                    guid, step_number, location_label
                )
                if delta:
                    deltas.append(delta)
                    # Remove from tracking after reporting
                    self._pending_missing.pop(guid, None)
        
        # Clear pending for entities that reappeared
        for guid in current_guids:
            self._pending_missing.pop(guid, None)
        
        # Detect movement and state changes for visible entities
        for guid, snapshot in current_entities.items():
            if guid in self._entity_history and self._entity_history[guid]:
                prev_snapshot = self._entity_history[guid][-1]
                
                # Check for movement
                move_delta = self._detect_movement(
                    guid, prev_snapshot, snapshot, step_number, location_label
                )
                if move_delta:
                    deltas.append(move_delta)
                
                # Check for state change
                state_delta = self._detect_state_change(
                    guid, prev_snapshot, snapshot, step_number, location_label
                )
                if state_delta:
                    deltas.append(state_delta)
        
        # Update history
        for guid, snapshot in current_entities.items():
            if guid not in self._entity_history:
                self._entity_history[guid] = []
            self._entity_history[guid].append(snapshot)
            # Keep limited history
            if len(self._entity_history[guid]) > 10:
                self._entity_history[guid] = self._entity_history[guid][-10:]
        
        self._last_visible_guids = current_guids
        self._total_deltas_detected += len(deltas)
        
        return deltas

    def _extract_entities(self, percept: Percept) -> dict[str, "EntitySnapshot"]:
        """Extract entity snapshots from percept.
        
        Args:
            percept: Current perception data.
            
        Returns:
            Dict mapping GUID to EntitySnapshot.
        """
        entities: dict[str, EntitySnapshot] = {}
        
        for candidate in percept.candidates:
            extras = candidate.extras or {}
            guid = extras.get("guid") or candidate.candidate_id
            
            snapshot = EntitySnapshot(
                guid=guid,
                label=candidate.label,
                category=extras.get("category", "unknown"),
                position=candidate.position,
                state=extras.get("state"),
                is_interactable=extras.get("interactable", False),
                visible=extras.get("visible", True),
                timestamp=datetime.now(),
            )
            entities[guid] = snapshot
        
        return entities

    def _create_new_entity_delta(
        self,
        guid: str,
        snapshot: "EntitySnapshot",
        step_number: int,
        location_label: str,
    ) -> Delta | None:
        """Create a delta for a newly appeared entity.
        
        Args:
            guid: Entity GUID.
            snapshot: Current entity snapshot.
            step_number: Current step number.
            location_label: Current location.
            
        Returns:
            Delta if this is genuinely new, None otherwise.
        """
        # Check if we've ever seen this entity before
        has_history = guid in self._entity_history and len(self._entity_history[guid]) > 0
        
        delta = Delta(
            delta_id=f"delta_{uuid.uuid4().hex[:12]}",
            delta_type=DeltaType.NEW_ENTITY,
            entity_id=guid,
            entity_label=snapshot.label,
            entity_category=snapshot.category,
            post_position=snapshot.position,
            post_state=snapshot.state,
            location_label=location_label,
            step_number=step_number,
            confidence=0.9 if not has_history else 0.7,
            evidence=[
                f"Entity {snapshot.label} appeared at step {step_number}",
                "First time seen" if not has_history else "Re-appeared after absence",
            ],
        )
        
        logger.debug(f"New entity delta: {snapshot.label} ({guid[:8]}...)")
        return delta

    def _create_missing_entity_delta(
        self,
        guid: str,
        step_number: int,
        location_label: str,
    ) -> Delta | None:
        """Create a delta for a missing entity.
        
        Args:
            guid: Entity GUID.
            step_number: Current step number.
            location_label: Current location.
            
        Returns:
            Delta for missing entity.
        """
        # Get last known info
        if guid not in self._entity_history or not self._entity_history[guid]:
            return None
        
        last_snapshot = self._entity_history[guid][-1]
        
        delta = Delta(
            delta_id=f"delta_{uuid.uuid4().hex[:12]}",
            delta_type=DeltaType.MISSING_ENTITY,
            entity_id=guid,
            entity_label=last_snapshot.label,
            entity_category=last_snapshot.category,
            pre_position=last_snapshot.position,
            pre_state=last_snapshot.state,
            location_label=location_label,
            step_number=step_number,
            confidence=0.8,
            evidence=[
                f"Entity {last_snapshot.label} not visible for {self._missing_window} steps",
                f"Last seen at step {step_number - self._missing_window}",
            ],
        )
        
        logger.debug(f"Missing entity delta: {last_snapshot.label} ({guid[:8]}...)")
        return delta

    def _detect_movement(
        self,
        guid: str,
        prev: "EntitySnapshot",
        curr: "EntitySnapshot",
        step_number: int,
        location_label: str,
    ) -> Delta | None:
        """Detect if entity has moved significantly.
        
        Args:
            guid: Entity GUID.
            prev: Previous snapshot.
            curr: Current snapshot.
            step_number: Current step number.
            location_label: Current location.
            
        Returns:
            Delta if movement detected, None otherwise.
        """
        if not prev.position or not curr.position:
            return None
        
        # Calculate distance
        distance = self._calculate_distance(prev.position, curr.position)
        
        if distance < self._move_threshold:
            return None
        
        delta = Delta(
            delta_id=f"delta_{uuid.uuid4().hex[:12]}",
            delta_type=DeltaType.MOVED_ENTITY,
            entity_id=guid,
            entity_label=curr.label,
            entity_category=curr.category,
            pre_position=prev.position,
            post_position=curr.position,
            position_delta=distance,
            location_label=location_label,
            step_number=step_number,
            confidence=0.95,
            evidence=[
                f"Entity {curr.label} moved {distance:.2f} units",
                f"From {prev.position} to {curr.position}",
            ],
        )
        
        logger.debug(f"Movement delta: {curr.label} moved {distance:.2f} units")
        return delta

    def _detect_state_change(
        self,
        guid: str,
        prev: "EntitySnapshot",
        curr: "EntitySnapshot",
        step_number: int,
        location_label: str,
    ) -> Delta | None:
        """Detect if entity state has changed.
        
        Args:
            guid: Entity GUID.
            prev: Previous snapshot.
            curr: Current snapshot.
            step_number: Current step number.
            location_label: Current location.
            
        Returns:
            Delta if state change detected, None otherwise.
        """
        # Only check interactable entities
        if not curr.is_interactable:
            return None
        
        # Check if state changed
        if prev.state == curr.state:
            return None
        
        if prev.state is None or curr.state is None:
            return None
        
        delta = Delta(
            delta_id=f"delta_{uuid.uuid4().hex[:12]}",
            delta_type=DeltaType.STATE_CHANGED,
            entity_id=guid,
            entity_label=curr.label,
            entity_category=curr.category,
            pre_state=prev.state,
            post_state=curr.state,
            post_position=curr.position,
            location_label=location_label,
            step_number=step_number,
            confidence=0.98,
            evidence=[
                f"Entity {curr.label} state changed: {prev.state} → {curr.state}",
            ],
        )
        
        logger.info(f"State change delta: {curr.label} {prev.state} → {curr.state}")
        return delta

    def _calculate_distance(
        self,
        pos1: tuple[float, float, float],
        pos2: tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean distance between two 3D positions."""
        return (
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        ) ** 0.5

    def reset(self) -> None:
        """Reset the delta detector state."""
        self._entity_history.clear()
        self._last_visible_guids.clear()
        self._pending_missing.clear()

    @property
    def total_deltas_detected(self) -> int:
        """Get total number of deltas detected."""
        return self._total_deltas_detected


class EntitySnapshot:
    """Snapshot of entity state at a point in time."""
    
    __slots__ = [
        "guid", "label", "category", "position", "state",
        "is_interactable", "visible", "timestamp",
    ]
    
    def __init__(
        self,
        guid: str,
        label: str,
        category: str,
        position: tuple[float, float, float] | None,
        state: str | None,
        is_interactable: bool,
        visible: bool,
        timestamp: datetime,
    ) -> None:
        self.guid = guid
        self.label = label
        self.category = category
        self.position = position
        self.state = state
        self.is_interactable = is_interactable
        self.visible = visible
        self.timestamp = timestamp
