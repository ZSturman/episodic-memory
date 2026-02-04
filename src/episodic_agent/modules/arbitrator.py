"""Motion-Perception Arbitrator for resolving conflicting signals.

ARCHITECTURAL INVARIANT: Motion Advisory / Perception Authoritative
If motion implies new location but visual ACF matches previous location,
enter uncertainty state and investigate. Never trust motion alone.

This module implements the arbitration logic between:
- Motion signals (advisory): suggest agent has moved to new location
- Perception signals (authoritative): visual evidence of current context

When these conflict, perception wins but uncertainty is flagged for
investigation through additional frames or user confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from episodic_agent.schemas import ActiveContextFrame, Percept

logger = logging.getLogger(__name__)


class SignalSource(str, Enum):
    """Source of location evidence."""
    
    MOTION = "motion"              # Motion/IMU/odometry signal
    PERCEPTION = "perception"      # Visual/perceptual evidence
    USER = "user"                  # User-provided information
    MEMORY = "memory"              # Memory-based inference


class ConflictType(str, Enum):
    """Type of signal conflict detected."""
    
    NONE = "none"                  # No conflict
    MOTION_PERCEPTION = "motion_perception"  # Motion says X, perception says Y
    TELEPORT = "teleport"          # Sudden visual change without motion
    DISCONTINUITY = "discontinuity"  # Motion discontinuity (e.g., elevator)


class ArbitrationOutcome(str, Enum):
    """Outcome of arbitration decision."""
    
    TRUST_PERCEPTION = "trust_perception"  # Perception is authoritative
    TRUST_MOTION = "trust_motion"          # Motion was correct (rare)
    UNCERTAIN = "uncertain"                # Conflicting, needs investigation
    INVESTIGATING = "investigating"        # Gathering more evidence


@dataclass
class MotionSignal:
    """Motion-based evidence about location change."""
    
    # Motion metrics
    displacement: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Derived
    total_displacement: float = 0.0
    
    # Motion suggests new location?
    suggests_new_location: bool = False
    suggested_location: str | None = None
    
    # Confidence in motion signal
    confidence: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Compute derived fields."""
        if self.total_displacement == 0.0:
            dx, dy, dz = self.displacement
            self.total_displacement = (dx**2 + dy**2 + dz**2) ** 0.5


@dataclass
class PerceptionSignal:
    """Perception-based evidence about current location."""
    
    # What perception suggests
    perceived_location: str = "unknown"
    location_confidence: float = 0.0
    
    # Feature matching
    acf_match_score: float = 0.0  # How well does this match current ACF?
    known_location_match: str | None = None  # Best matching known location
    known_location_score: float = 0.0
    
    # Anomalies
    is_anomalous: bool = False
    anomaly_reason: str = ""
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArbitrationDecision:
    """Result of motion-perception arbitration."""
    
    # Core decision
    outcome: ArbitrationOutcome
    conflict_type: ConflictType
    
    # What we're going with
    resolved_location: str
    resolved_confidence: float
    
    # Evidence
    motion_evidence: MotionSignal | None = None
    perception_evidence: PerceptionSignal | None = None
    
    # Investigation needed?
    needs_investigation: bool = False
    investigation_frames_remaining: int = 0
    
    # Logging context
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Log the arbitration decision for traceability."""
        if self.conflict_type != ConflictType.NONE:
            motion_loc = (
                self.motion_evidence.suggested_location 
                if self.motion_evidence else "unknown"
            )
            perception_loc = (
                self.perception_evidence.perceived_location 
                if self.perception_evidence else "unknown"
            )
            logger.info(
                f"[ARBITRATION] Motion suggests '{motion_loc}', "
                f"perception suggests '{perception_loc}', "
                f"outcome={self.outcome.value}, reason={self.reason}"
            )


class MotionPerceptionArbitrator:
    """Arbitrates between motion and perception signals.
    
    ARCHITECTURAL INVARIANT: Perception is authoritative, motion is advisory.
    
    When motion and perception conflict:
    1. DO NOT auto-relabel based on motion alone
    2. Enter uncertainty state
    3. Require additional evidence (frames or user confirmation)
    4. Log arbitration decisions clearly
    
    Key scenarios:
    - Elevator: Motion discontinuity + visual match → don't auto-relabel
    - Teleport: Sudden visual change without motion → anomaly, investigate
    - Normal walk: Motion + visual change → monitor, then transition
    """
    
    def __init__(
        self,
        # Motion thresholds
        significant_motion_threshold: float = 0.5,  # Meters
        motion_discontinuity_threshold: float = 5.0,  # Large sudden jump
        
        # Investigation settings
        investigation_frames: int = 5,
        
        # Confidence thresholds
        perception_confidence_for_override: float = 0.8,
        motion_confidence_for_consideration: float = 0.5,
    ) -> None:
        """Initialize the arbitrator.
        
        Args:
            significant_motion_threshold: Motion distance to consider significant.
            motion_discontinuity_threshold: Motion suggesting teleport/discontinuity.
            investigation_frames: Frames to investigate before resolving.
            perception_confidence_for_override: Min perception confidence to trust.
            motion_confidence_for_consideration: Min motion confidence to consider.
        """
        self._significant_motion_threshold = significant_motion_threshold
        self._motion_discontinuity_threshold = motion_discontinuity_threshold
        self._investigation_frames = investigation_frames
        self._perception_confidence_for_override = perception_confidence_for_override
        self._motion_confidence_for_consideration = motion_confidence_for_consideration
        
        # State tracking
        self._current_location: str = "unknown"
        self._investigating: bool = False
        self._investigation_frames_remaining: int = 0
        self._investigation_evidence: list[ArbitrationDecision] = []
        
        # History
        self._previous_motion: MotionSignal | None = None
        self._previous_perception: PerceptionSignal | None = None
        self._decision_history: list[ArbitrationDecision] = []
        
        # Statistics
        self._total_arbitrations = 0
        self._conflicts_detected = 0
        self._perception_wins = 0
        self._investigations_started = 0
    
    @property
    def is_investigating(self) -> bool:
        """Whether currently investigating a conflict."""
        return self._investigating
    
    @property
    def current_location(self) -> str:
        """Current resolved location."""
        return self._current_location
    
    def create_motion_signal(
        self,
        percept: "Percept",
        acf: "ActiveContextFrame",
    ) -> MotionSignal:
        """Extract motion signal from perception and context.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            Motion signal with displacement and derived metrics.
        """
        # Extract motion from percept if available
        displacement = (0.0, 0.0, 0.0)
        velocity = (0.0, 0.0, 0.0)
        
        if percept and hasattr(percept, 'extras') and percept.extras:
            # Check for motion data in extras
            if 'motion' in percept.extras:
                motion_data = percept.extras['motion']
                displacement = tuple(motion_data.get('displacement', [0, 0, 0]))
                velocity = tuple(motion_data.get('velocity', [0, 0, 0]))
            
            # Or camera pose change
            if 'camera_pose' in percept.extras:
                pose = percept.extras['camera_pose']
                if self._previous_perception and hasattr(self._previous_perception, 'extras'):
                    # Compute displacement from pose change
                    # (simplified - would need previous pose)
                    pass
        
        # Compute total displacement
        dx, dy, dz = displacement
        total = (dx**2 + dy**2 + dz**2) ** 0.5
        
        # Does this suggest a new location?
        suggests_new = total > self._significant_motion_threshold
        is_discontinuity = total > self._motion_discontinuity_threshold
        
        return MotionSignal(
            displacement=displacement,
            velocity=velocity,
            total_displacement=total,
            suggests_new_location=suggests_new or is_discontinuity,
            confidence=0.5 if suggests_new else 0.2,
        )
    
    def create_perception_signal(
        self,
        percept: "Percept",
        acf: "ActiveContextFrame",
        acf_match_score: float = 0.0,
    ) -> PerceptionSignal:
        """Extract perception signal from current perception.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            acf_match_score: Score from ACF stability evaluation.
            
        Returns:
            Perception signal with location evidence.
        """
        return PerceptionSignal(
            perceived_location=acf.location_label if acf else "unknown",
            location_confidence=acf.location_confidence if acf else 0.0,
            acf_match_score=acf_match_score,
        )
    
    def arbitrate(
        self,
        motion: MotionSignal,
        perception: PerceptionSignal,
    ) -> ArbitrationDecision:
        """Arbitrate between motion and perception signals.
        
        This is the core arbitration method implementing:
        - Perception authoritative, motion advisory
        - Conflict detection and investigation
        - Never auto-relabel based on motion alone
        
        Args:
            motion: Motion signal evidence.
            perception: Perception signal evidence.
            
        Returns:
            ArbitrationDecision with resolved location and action.
        """
        self._total_arbitrations += 1
        
        # Store evidence
        self._previous_motion = motion
        self._previous_perception = perception
        
        # If investigating, collect evidence
        if self._investigating:
            return self._continue_investigation(motion, perception)
        
        # Check for conflicts
        conflict = self._detect_conflict(motion, perception)
        
        if conflict == ConflictType.NONE:
            # No conflict - trust perception
            decision = ArbitrationDecision(
                outcome=ArbitrationOutcome.TRUST_PERCEPTION,
                conflict_type=ConflictType.NONE,
                resolved_location=perception.perceived_location,
                resolved_confidence=perception.location_confidence,
                motion_evidence=motion,
                perception_evidence=perception,
                reason="No conflict - perception authoritative",
            )
            self._current_location = perception.perceived_location
            self._decision_history.append(decision)
            return decision
        
        # Conflict detected
        self._conflicts_detected += 1
        
        # Case 1: Motion discontinuity (elevator/teleport)
        if conflict == ConflictType.DISCONTINUITY:
            # DON'T auto-relabel - start investigation
            self._start_investigation(motion, perception)
            decision = ArbitrationDecision(
                outcome=ArbitrationOutcome.INVESTIGATING,
                conflict_type=ConflictType.DISCONTINUITY,
                resolved_location=self._current_location,  # Keep current
                resolved_confidence=perception.location_confidence * 0.5,
                motion_evidence=motion,
                perception_evidence=perception,
                needs_investigation=True,
                investigation_frames_remaining=self._investigation_frames,
                reason="Motion discontinuity detected - investigating, NOT auto-relabeling",
            )
            self._decision_history.append(decision)
            return decision
        
        # Case 2: Teleport (visual change without motion)
        if conflict == ConflictType.TELEPORT:
            # Anomalous - start investigation
            self._start_investigation(motion, perception)
            decision = ArbitrationDecision(
                outcome=ArbitrationOutcome.UNCERTAIN,
                conflict_type=ConflictType.TELEPORT,
                resolved_location=self._current_location,  # Keep current
                resolved_confidence=perception.location_confidence * 0.3,
                motion_evidence=motion,
                perception_evidence=perception,
                needs_investigation=True,
                investigation_frames_remaining=self._investigation_frames,
                reason="Visual change without motion - anomaly, investigating",
            )
            self._decision_history.append(decision)
            return decision
        
        # Case 3: Motion-perception conflict
        if conflict == ConflictType.MOTION_PERCEPTION:
            # Perception is authoritative, but flag for investigation
            if perception.location_confidence >= self._perception_confidence_for_override:
                # High confidence perception - trust it but note conflict
                self._perception_wins += 1
                decision = ArbitrationDecision(
                    outcome=ArbitrationOutcome.TRUST_PERCEPTION,
                    conflict_type=ConflictType.MOTION_PERCEPTION,
                    resolved_location=perception.perceived_location,
                    resolved_confidence=perception.location_confidence,
                    motion_evidence=motion,
                    perception_evidence=perception,
                    reason="Motion-perception conflict - high confidence perception wins",
                )
                self._current_location = perception.perceived_location
            else:
                # Lower confidence - investigate
                self._start_investigation(motion, perception)
                decision = ArbitrationDecision(
                    outcome=ArbitrationOutcome.INVESTIGATING,
                    conflict_type=ConflictType.MOTION_PERCEPTION,
                    resolved_location=self._current_location,
                    resolved_confidence=perception.location_confidence * 0.7,
                    motion_evidence=motion,
                    perception_evidence=perception,
                    needs_investigation=True,
                    investigation_frames_remaining=self._investigation_frames,
                    reason="Motion-perception conflict - low confidence, investigating",
                )
            
            self._decision_history.append(decision)
            return decision
        
        # Default - trust perception
        self._perception_wins += 1
        decision = ArbitrationDecision(
            outcome=ArbitrationOutcome.TRUST_PERCEPTION,
            conflict_type=conflict,
            resolved_location=perception.perceived_location,
            resolved_confidence=perception.location_confidence,
            motion_evidence=motion,
            perception_evidence=perception,
            reason="Default - perception authoritative",
        )
        self._current_location = perception.perceived_location
        self._decision_history.append(decision)
        return decision
    
    def _detect_conflict(
        self,
        motion: MotionSignal,
        perception: PerceptionSignal,
    ) -> ConflictType:
        """Detect type of conflict between signals.
        
        Args:
            motion: Motion signal.
            perception: Perception signal.
            
        Returns:
            Type of conflict detected.
        """
        has_significant_motion = motion.total_displacement > self._significant_motion_threshold
        has_discontinuity = motion.total_displacement > self._motion_discontinuity_threshold
        visual_matches_current = perception.acf_match_score > 0.7
        location_changed = perception.perceived_location != self._current_location
        
        # Case 1: Motion discontinuity (e.g., elevator)
        if has_discontinuity and visual_matches_current:
            return ConflictType.DISCONTINUITY
        
        # Case 2: Visual change without motion (teleport)
        if not has_significant_motion and location_changed and not visual_matches_current:
            return ConflictType.TELEPORT
        
        # Case 3: Motion says new location, perception says old
        if has_significant_motion and motion.suggests_new_location:
            if visual_matches_current or perception.perceived_location == self._current_location:
                return ConflictType.MOTION_PERCEPTION
        
        return ConflictType.NONE
    
    def _start_investigation(
        self,
        motion: MotionSignal,
        perception: PerceptionSignal,
    ) -> None:
        """Start an investigation period."""
        self._investigating = True
        self._investigation_frames_remaining = self._investigation_frames
        self._investigation_evidence = []
        self._investigations_started += 1
        
        logger.info(
            f"[ARBITRATION] Starting investigation - "
            f"motion={motion.suggests_new_location}, "
            f"perception='{perception.perceived_location}', "
            f"frames={self._investigation_frames}"
        )
    
    def _continue_investigation(
        self,
        motion: MotionSignal,
        perception: PerceptionSignal,
    ) -> ArbitrationDecision:
        """Continue an ongoing investigation."""
        self._investigation_frames_remaining -= 1
        
        # Collect evidence
        conflict = self._detect_conflict(motion, perception)
        evidence = ArbitrationDecision(
            outcome=ArbitrationOutcome.INVESTIGATING,
            conflict_type=conflict,
            resolved_location=self._current_location,
            resolved_confidence=perception.location_confidence,
            motion_evidence=motion,
            perception_evidence=perception,
        )
        self._investigation_evidence.append(evidence)
        
        if self._investigation_frames_remaining > 0:
            # Still investigating
            return ArbitrationDecision(
                outcome=ArbitrationOutcome.INVESTIGATING,
                conflict_type=conflict,
                resolved_location=self._current_location,
                resolved_confidence=perception.location_confidence * 0.7,
                motion_evidence=motion,
                perception_evidence=perception,
                needs_investigation=True,
                investigation_frames_remaining=self._investigation_frames_remaining,
                reason=f"Investigation ongoing - {self._investigation_frames_remaining} frames remaining",
            )
        
        # Investigation complete - resolve
        return self._resolve_investigation(motion, perception)
    
    def _resolve_investigation(
        self,
        motion: MotionSignal,
        perception: PerceptionSignal,
    ) -> ArbitrationDecision:
        """Resolve an investigation based on collected evidence."""
        self._investigating = False
        
        # Count how many frames supported each outcome
        perception_support = sum(
            1 for e in self._investigation_evidence
            if e.perception_evidence and e.perception_evidence.acf_match_score > 0.5
        )
        
        total_evidence = len(self._investigation_evidence)
        perception_ratio = perception_support / total_evidence if total_evidence > 0 else 0.5
        
        # Perception-majority → trust perception
        if perception_ratio >= 0.6:
            self._perception_wins += 1
            decision = ArbitrationDecision(
                outcome=ArbitrationOutcome.TRUST_PERCEPTION,
                conflict_type=ConflictType.MOTION_PERCEPTION,
                resolved_location=perception.perceived_location,
                resolved_confidence=perception.location_confidence,
                motion_evidence=motion,
                perception_evidence=perception,
                reason=f"Investigation resolved - perception supported by {perception_ratio:.0%} of frames",
            )
            self._current_location = perception.perceived_location
        else:
            # Uncertain - keep current location, flag for user
            decision = ArbitrationDecision(
                outcome=ArbitrationOutcome.UNCERTAIN,
                conflict_type=ConflictType.MOTION_PERCEPTION,
                resolved_location=self._current_location,
                resolved_confidence=0.3,  # Low confidence
                motion_evidence=motion,
                perception_evidence=perception,
                needs_investigation=True,  # Still needs user confirmation
                reason=f"Investigation inconclusive - only {perception_ratio:.0%} perception support, needs user input",
            )
        
        self._investigation_evidence = []
        self._decision_history.append(decision)
        
        logger.info(
            f"[ARBITRATION] Investigation resolved - "
            f"outcome={decision.outcome.value}, "
            f"location='{decision.resolved_location}'"
        )
        
        return decision
    
    def force_location(self, location: str, reason: str = "user_override") -> None:
        """Force a location update (e.g., from user confirmation).
        
        Args:
            location: The confirmed location.
            reason: Reason for the forced update.
        """
        self._current_location = location
        self._investigating = False
        self._investigation_frames_remaining = 0
        self._investigation_evidence = []
        
        logger.info(f"[ARBITRATION] Location forced to '{location}' - reason: {reason}")
    
    def reset(self) -> None:
        """Reset the arbitrator to initial state."""
        self._current_location = "unknown"
        self._investigating = False
        self._investigation_frames_remaining = 0
        self._investigation_evidence = []
        self._previous_motion = None
        self._previous_perception = None
        self._decision_history = []
        
        logger.info("[ARBITRATION] Arbitrator reset to initial state")
    
    def get_statistics(self) -> dict[str, Any]:
        """Get arbitrator statistics for logging/debugging.
        
        Returns:
            Dictionary of statistics.
        """
        return {
            "total_arbitrations": self._total_arbitrations,
            "conflicts_detected": self._conflicts_detected,
            "perception_wins": self._perception_wins,
            "investigations_started": self._investigations_started,
            "currently_investigating": self._investigating,
            "current_location": self._current_location,
            "decision_history_size": len(self._decision_history),
        }
