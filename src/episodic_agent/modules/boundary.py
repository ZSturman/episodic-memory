"""Boundary detector with hysteresis for episode segmentation.

Enhanced in Phase 6 to support:
- Prediction error spike boundary triggers
- Salient event boundary triggers
- Multiple reason codes per boundary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
    PREDICTION_ERROR = "prediction_error"  # High prediction error
    SALIENT_EVENT = "salient_event"        # Salient labeled event detected


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
    
    # Prediction error tracking (Phase 6)
    prediction_error_history: list[float] = field(default_factory=list)
    prediction_error_spike_frames: int = 0  # Consecutive frames with high error


class HysteresisBoundaryDetector(BoundaryDetector):
    """Boundary detector with hysteresis to prevent episode churn.
    
    Supports multiple boundary triggers:
    - Periodic timeout (maximum frames per episode)
    - Location change (with confidence and hysteresis)
    - Periodic interval (optional, for compatibility with stub)
    - Prediction error spike (Phase 6)
    - Salient event detection (Phase 6)
    
    Hysteresis prevents rapid boundary triggering by requiring:
    - Minimum frames before location-based boundary
    - High confidence in the new location
    - Sustained change across multiple frames
    - Sustained prediction error spike (multiple frames)
    """

    def __init__(
        self,
        timeout_frames: int = BOUNDARY_TIMEOUT_FRAMES,
        min_frames_hysteresis: int = BOUNDARY_HYSTERESIS_MIN_FRAMES,
        location_confidence_threshold: float = CONFIDENCE_T_HIGH,
        location_confidence_delta: float = LOCATION_CHANGE_CONFIDENCE_DELTA,
        interval_frames: int | None = None,  # Optional periodic interval
        # Phase 6: Prediction error settings
        prediction_error_threshold: float = 0.6,
        prediction_error_spike_frames: int = 2,
        prediction_error_history_size: int = 10,
        # Phase 6: Salient event settings
        salient_event_confidence: float = 0.8,
        salient_event_types: list[str] | None = None,
    ) -> None:
        """Initialize the boundary detector.
        
        Args:
            timeout_frames: Maximum frames before forced boundary.
            min_frames_hysteresis: Minimum frames before allowing boundary.
            location_confidence_threshold: Confidence needed for location boundary.
            location_confidence_delta: Min confidence increase to consider change.
            interval_frames: Optional periodic interval (like stub behavior).
            prediction_error_threshold: Threshold for prediction error spike.
            prediction_error_spike_frames: Frames needed for error spike trigger.
            prediction_error_history_size: Size of error history to track.
            salient_event_confidence: Min confidence for salient event trigger.
            salient_event_types: Event types considered salient (None = all).
        """
        self._timeout_frames = timeout_frames
        self._min_frames_hysteresis = min_frames_hysteresis
        self._location_confidence_threshold = location_confidence_threshold
        self._location_confidence_delta = location_confidence_delta
        self._interval_frames = interval_frames
        
        # Phase 6: Prediction error settings
        self._prediction_error_threshold = prediction_error_threshold
        self._prediction_error_spike_frames = prediction_error_spike_frames
        self._prediction_error_history_size = prediction_error_history_size
        
        # Phase 6: Salient event settings
        self._salient_event_confidence = salient_event_confidence
        self._salient_event_types = salient_event_types or [
            "opened", "closed", "turned_on", "turned_off",
            "appeared", "disappeared", "picked_up", "put_down"
        ]
        
        # Internal state
        self._state = BoundaryState()
        
        # Statistics (Phase 6)
        self._boundary_counts: dict[str, int] = {
            BoundaryReason.TIMEOUT.value: 0,
            BoundaryReason.INTERVAL.value: 0,
            BoundaryReason.LOCATION_CHANGE.value: 0,
            BoundaryReason.PREDICTION_ERROR.value: 0,
            BoundaryReason.SALIENT_EVENT.value: 0,
        }

    def check(self, acf: ActiveContextFrame) -> tuple[bool, str | None]:
        """Check if an episode boundary should be triggered.
        
        Args:
            acf: Current active context frame.
            
        Returns:
            Tuple of (should_freeze, reason_code).
        """
        self._state.frames_since_boundary += 1
        reasons: list[str] = []
        
        # Check timeout (highest priority)
        if self._state.frames_since_boundary >= self._timeout_frames:
            reasons.append(f"{BoundaryReason.TIMEOUT.value}_{self._timeout_frames}_frames")
        
        # Check periodic interval if configured
        if self._interval_frames and acf.step_count > 0:
            if acf.step_count % self._interval_frames == 0:
                reasons.append(f"{BoundaryReason.INTERVAL.value}_{self._interval_frames}_steps")
        
        # Check location change with hysteresis
        location_changed, loc_reason = self._check_location_change(acf)
        if location_changed:
            reasons.append(loc_reason)
        
        # Check prediction error spike (Phase 6)
        error_spike, error_reason = self._check_prediction_error(acf)
        if error_spike:
            reasons.append(error_reason)
        
        # Check salient events (Phase 6)
        salient_event, event_reason = self._check_salient_events(acf)
        if salient_event:
            reasons.append(event_reason)
        
        # Update tracking state
        self._update_state(acf)
        
        # Trigger boundary if any reason found
        if reasons:
            combined_reason = "; ".join(reasons)
            self._update_boundary_counts(reasons)
            self._reset_state(acf)
            return (True, combined_reason)
        
        return (False, None)

    def _check_location_change(self, acf: ActiveContextFrame) -> tuple[bool, str]:
        """Check if location has changed with sufficient confidence.
        
        Uses hysteresis to avoid triggering on noisy location estimates.
        
        Returns:
            Tuple of (triggered, reason_string).
        """
        # Need minimum frames before considering location change
        if self._state.frames_since_boundary < self._min_frames_hysteresis:
            return (False, "")
        
        # Check if location label changed from the established location
        label_changed = (
            acf.location_label != self._state.prev_location_label and
            acf.location_label != "unknown" and
            self._state.prev_location_label != "unknown"
        )
        
        if not label_changed:
            # Back to established location - reset change counter
            self._state.location_change_frames = 0
            return (False, "")
        
        # Location is different - check confidence
        confidence_high = acf.location_confidence >= self._location_confidence_threshold
        confidence_increased = (
            acf.location_confidence - self._state.prev_location_confidence
        ) >= self._location_confidence_delta
        
        if not (confidence_high or confidence_increased):
            # Not confident enough - reset change counter
            self._state.location_change_frames = 0
            return (False, "")
        
        # Require sustained change (at least 2 frames)
        self._state.location_change_frames += 1
        if self._state.location_change_frames >= 2:
            return (True, f"{BoundaryReason.LOCATION_CHANGE.value}:{acf.location_label}")
        
        return (False, "")

    def _check_prediction_error(self, acf: ActiveContextFrame) -> tuple[bool, str]:
        """Check if prediction error spike should trigger boundary.
        
        Uses hysteresis to require sustained high error.
        
        Returns:
            Tuple of (triggered, reason_string).
        """
        # Get prediction error magnitude from ACF extras
        error_magnitude = acf.extras.get("prediction_error_magnitude", 0.0)
        error_count = acf.extras.get("prediction_error_count", 0)
        
        # Track error history
        self._state.prediction_error_history.append(error_magnitude)
        if len(self._state.prediction_error_history) > self._prediction_error_history_size:
            self._state.prediction_error_history.pop(0)
        
        # Need minimum frames before considering prediction error
        if self._state.frames_since_boundary < self._min_frames_hysteresis:
            return (False, "")
        
        # Check if error exceeds threshold
        if error_magnitude >= self._prediction_error_threshold:
            self._state.prediction_error_spike_frames += 1
            
            if self._state.prediction_error_spike_frames >= self._prediction_error_spike_frames:
                return (
                    True,
                    f"{BoundaryReason.PREDICTION_ERROR.value}:mag={error_magnitude:.2f},cnt={error_count}"
                )
        else:
            # Reset spike counter (with hysteresis - allow one low frame)
            if self._state.prediction_error_spike_frames > 0:
                self._state.prediction_error_spike_frames -= 1
        
        return (False, "")

    def _check_salient_events(self, acf: ActiveContextFrame) -> tuple[bool, str]:
        """Check if a salient event should trigger boundary.
        
        Returns:
            Tuple of (triggered, reason_string).
        """
        # Need minimum frames before considering salient events
        if self._state.frames_since_boundary < self._min_frames_hysteresis:
            return (False, "")
        
        # Check recent events in ACF
        for event in acf.events[-3:]:  # Check last 3 events
            event_type = event.get("event_type", event.get("type", ""))
            event_confidence = event.get("confidence", 0.0)
            event_label = event.get("label", "unknown")
            
            # Check if event type is salient and confident enough
            if event_type in self._salient_event_types:
                if event_confidence >= self._salient_event_confidence:
                    return (
                        True,
                        f"{BoundaryReason.SALIENT_EVENT.value}:{event_label}({event_type})"
                    )
        
        return (False, "")

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

    def _update_boundary_counts(self, reasons: list[str]) -> None:
        """Update boundary statistics by reason type."""
        for reason in reasons:
            for reason_type in BoundaryReason:
                if reason.startswith(reason_type.value):
                    self._boundary_counts[reason_type.value] = (
                        self._boundary_counts.get(reason_type.value, 0) + 1
                    )
                    break

    def _reset_state(self, acf: ActiveContextFrame) -> None:
        """Reset state after a boundary trigger."""
        self._state = BoundaryState(
            prev_location_label=acf.location_label,
            prev_location_confidence=acf.location_confidence,
            frames_since_boundary=0,
            location_change_frames=0,
            stable_frames=0,
            prediction_error_history=[],
            prediction_error_spike_frames=0,
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

    @property
    def boundary_counts(self) -> dict[str, int]:
        """Get boundary trigger counts by reason type."""
        return dict(self._boundary_counts)

    def get_statistics(self) -> dict[str, Any]:
        """Get boundary detector statistics.
        
        Returns:
            Statistics dictionary.
        """
        return {
            "frames_since_boundary": self._state.frames_since_boundary,
            "current_location": self._state.prev_location_label,
            "stable_frames": self._state.stable_frames,
            "prediction_error_spike_frames": self._state.prediction_error_spike_frames,
            "boundary_counts": dict(self._boundary_counts),
            "error_history_length": len(self._state.prediction_error_history),
        }
