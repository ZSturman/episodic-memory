"""Boundary detector with hysteresis for episode segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from episodic_agent.core.interfaces import BoundaryDetector
from episodic_agent.schemas import ActiveContextFrame
from episodic_agent.utils.config import (
    BOUNDARY_HYSTERESIS_MIN_FRAMES,
    BOUNDARY_TIMEOUT_FRAMES,
    CONFIDENCE_T_HIGH,
    LOCATION_CHANGE_CONFIDENCE_DELTA,
)


class BoundaryReason(str, Enum):
    """Reason codes for episode boundaries."""
    
    TIMEOUT = "timeout"                    # Maximum frames exceeded
    INTERVAL = "interval"                  # Periodic interval
    LOCATION_CHANGE = "location_change"    # Location changed with confidence
    GOAL_COMPLETE = "goal_complete"        # Goal was completed
    USER_TRIGGERED = "user_triggered"      # User manually triggered
    PREDICTION_ERROR = "prediction_error"  # High prediction error (future)


@dataclass
class BoundaryState:
    """Internal state for hysteresis tracking."""
    
    # Previous location for change detection
    prev_location_label: str = "unknown"
    prev_location_confidence: float = 0.0
    
    # Frames since last boundary
    frames_since_boundary: int = 0
    
    # Hysteresis counters
    location_change_frames: int = 0  # Consecutive frames with different location
    stable_frames: int = 0           # Consecutive frames with same location


class HysteresisBoundaryDetector(BoundaryDetector):
    """Boundary detector with hysteresis to prevent episode churn.
    
    Supports multiple boundary triggers:
    - Periodic timeout (maximum frames per episode)
    - Location change (with confidence and hysteresis)
    - Periodic interval (optional, for compatibility with stub)
    
    Hysteresis prevents rapid boundary triggering by requiring:
    - Minimum frames before location-based boundary
    - High confidence in the new location
    - Sustained change across multiple frames
    """

    def __init__(
        self,
        timeout_frames: int = BOUNDARY_TIMEOUT_FRAMES,
        min_frames_hysteresis: int = BOUNDARY_HYSTERESIS_MIN_FRAMES,
        location_confidence_threshold: float = CONFIDENCE_T_HIGH,
        location_confidence_delta: float = LOCATION_CHANGE_CONFIDENCE_DELTA,
        interval_frames: int | None = None,  # Optional periodic interval
    ) -> None:
        """Initialize the boundary detector.
        
        Args:
            timeout_frames: Maximum frames before forced boundary.
            min_frames_hysteresis: Minimum frames before allowing boundary.
            location_confidence_threshold: Confidence needed for location boundary.
            location_confidence_delta: Min confidence increase to consider change.
            interval_frames: Optional periodic interval (like stub behavior).
        """
        self._timeout_frames = timeout_frames
        self._min_frames_hysteresis = min_frames_hysteresis
        self._location_confidence_threshold = location_confidence_threshold
        self._location_confidence_delta = location_confidence_delta
        self._interval_frames = interval_frames
        
        # Internal state
        self._state = BoundaryState()

    def check(self, acf: ActiveContextFrame) -> tuple[bool, str | None]:
        """Check if an episode boundary should be triggered.
        
        Args:
            acf: Current active context frame.
            
        Returns:
            Tuple of (should_freeze, reason_code).
        """
        self._state.frames_since_boundary += 1
        
        # Check timeout (highest priority)
        if self._state.frames_since_boundary >= self._timeout_frames:
            reason = f"{BoundaryReason.TIMEOUT.value}_{self._timeout_frames}_frames"
            self._reset_state(acf)
            return (True, reason)
        
        # Check periodic interval if configured
        if self._interval_frames and acf.step_count > 0:
            if acf.step_count % self._interval_frames == 0:
                reason = f"{BoundaryReason.INTERVAL.value}_{self._interval_frames}_steps"
                self._reset_state(acf)
                return (True, reason)
        
        # Check location change with hysteresis
        location_changed = self._check_location_change(acf)
        if location_changed:
            reason = f"{BoundaryReason.LOCATION_CHANGE.value}:{acf.location_label}"
            self._reset_state(acf)
            return (True, reason)
        
        # Update tracking state
        self._update_state(acf)
        
        return (False, None)

    def _check_location_change(self, acf: ActiveContextFrame) -> bool:
        """Check if location has changed with sufficient confidence.
        
        Uses hysteresis to avoid triggering on noisy location estimates.
        """
        # Need minimum frames before considering location change
        if self._state.frames_since_boundary < self._min_frames_hysteresis:
            return False
        
        # Check if location label changed from the established location
        label_changed = (
            acf.location_label != self._state.prev_location_label and
            acf.location_label != "unknown" and
            self._state.prev_location_label != "unknown"
        )
        
        if not label_changed:
            # Back to established location - reset change counter
            self._state.location_change_frames = 0
            return False
        
        # Location is different - check confidence
        confidence_high = acf.location_confidence >= self._location_confidence_threshold
        confidence_increased = (
            acf.location_confidence - self._state.prev_location_confidence
        ) >= self._location_confidence_delta
        
        if not (confidence_high or confidence_increased):
            # Not confident enough - reset change counter
            self._state.location_change_frames = 0
            return False
        
        # Require sustained change (at least 2 frames)
        self._state.location_change_frames += 1
        if self._state.location_change_frames >= 2:
            return True
        
        return False

    def _update_state(self, acf: ActiveContextFrame) -> None:
        """Update tracking state after a non-boundary check.
        
        Only updates prev_location if we're NOT tracking a potential change,
        or if the previous location was unknown (establishing initial location).
        """
        # If previous location was unknown, establish the first known location
        if self._state.prev_location_label == "unknown" and acf.location_label != "unknown":
            self._state.prev_location_label = acf.location_label
            self._state.prev_location_confidence = acf.location_confidence
            self._state.stable_frames = 1
            return
        
        # Track if location is stable
        if acf.location_label == self._state.prev_location_label:
            self._state.stable_frames += 1
            # Only update confidence when stable - don't update during potential change tracking
            self._state.prev_location_confidence = acf.location_confidence
        else:
            self._state.stable_frames = 0
            # Don't update prev_location here - we want to track change from established location

    def _reset_state(self, acf: ActiveContextFrame) -> None:
        """Reset state after a boundary trigger."""
        self._state = BoundaryState(
            prev_location_label=acf.location_label,
            prev_location_confidence=acf.location_confidence,
            frames_since_boundary=0,
            location_change_frames=0,
            stable_frames=0,
        )

    def reset(self) -> None:
        """Fully reset the detector state."""
        self._state = BoundaryState()

    @property
    def frames_since_boundary(self) -> int:
        """Get frames since last boundary."""
        return self._state.frames_since_boundary

    @property
    def current_location(self) -> str:
        """Get the tracked current location."""
        return self._state.prev_location_label
