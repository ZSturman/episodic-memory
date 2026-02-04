"""ACF Stability Guard for protecting contextual identity.

ARCHITECTURAL INVARIANT: ACF Stability
A location label persists across perceptual variation unless strong
contradictory evidence accumulates. Lighting, shadows, temporary clutter
do not fragment identity.

This module implements the core recognition invariants that protect
identity stability:
1. Perceptual variation (lighting, shadows, clutter) does NOT create new location
2. Sudden visual change without motion → candidate anomaly, not new location
3. Only sustained mismatch + spatial contradiction triggers uncertainty
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


class StabilityState(str, Enum):
    """Current stability state of the ACF identity."""
    
    STABLE = "stable"              # Identity is confident
    MONITORING = "monitoring"      # Minor variations detected, watching
    UNCERTAIN = "uncertain"        # Conflicting evidence, needs resolution
    TRANSITIONING = "transitioning"  # Active transition to new identity


class VariationType(str, Enum):
    """Types of perceptual variation that don't fragment identity."""
    
    LIGHTING = "lighting"          # Lighting changes
    SHADOW = "shadow"              # Shadow changes
    CLUTTER = "clutter"            # Temporary objects
    OCCLUSION = "occlusion"        # Partial occlusion
    NOISE = "noise"                # Sensor noise


@dataclass
class StabilityDecision:
    """Result of a stability evaluation."""
    
    # Core decision
    identity_stable: bool
    new_state: StabilityState
    
    # Evidence
    variation_type: VariationType | None = None
    confidence_adjustment: float = 0.0
    
    # Logging context
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Log the stability decision for traceability."""
        logger.info(
            f"[STABILITY] Decision: stable={self.identity_stable}, "
            f"state={self.new_state.value}, reason={self.reason}"
        )


@dataclass
class ACFFingerprint:
    """Fingerprint capturing key identity features of an ACF.
    
    Used to compare whether two perceptual states represent
    the same contextual identity.
    """
    
    # Spatial signature
    entity_count: int = 0
    entity_guids: frozenset[str] = field(default_factory=frozenset)
    
    # Structural signature (positions, not absolute coords)
    entity_relative_positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    
    # Visual signature (learned features, not hardcoded)
    visual_hash: str | None = None
    
    # Location context
    location_label: str = "unknown"
    location_confidence: float = 0.0
    
    def similarity_to(self, other: "ACFFingerprint") -> float:
        """Compute similarity score to another fingerprint.
        
        Returns:
            Similarity score in [0, 1]. Higher = more similar.
        """
        if not other:
            return 0.0
        
        scores = []
        
        # Entity overlap (Jaccard similarity)
        if self.entity_guids or other.entity_guids:
            intersection = len(self.entity_guids & other.entity_guids)
            union = len(self.entity_guids | other.entity_guids)
            entity_sim = intersection / union if union > 0 else 1.0
            scores.append(entity_sim * 0.4)  # 40% weight
        else:
            scores.append(0.4)  # Both empty = same
        
        # Entity count similarity
        if self.entity_count > 0 or other.entity_count > 0:
            max_count = max(self.entity_count, other.entity_count)
            min_count = min(self.entity_count, other.entity_count)
            count_sim = min_count / max_count if max_count > 0 else 1.0
            scores.append(count_sim * 0.2)  # 20% weight
        else:
            scores.append(0.2)
        
        # Location match
        if self.location_label == other.location_label:
            scores.append(0.4)  # 40% weight for same location
        elif self.location_label == "unknown" or other.location_label == "unknown":
            scores.append(0.2)  # Partial credit if unknown
        else:
            scores.append(0.0)  # Different locations
        
        return sum(scores)


class ACFStabilityGuard:
    """Guards ACF identity against inappropriate fragmentation.
    
    ARCHITECTURAL INVARIANT: This class enforces:
    1. Lighting/shadow changes don't fragment identity
    2. Sudden visual change without motion → anomaly, not new location
    3. Only sustained mismatch + spatial contradiction → uncertainty
    
    The guard maintains a history of fingerprints and monitors for
    variations that should NOT trigger identity changes.
    """
    
    def __init__(
        self,
        # Stability thresholds
        similarity_threshold: float = 0.6,
        mismatch_frames_for_uncertainty: int = 5,
        mismatch_frames_for_transition: int = 10,
        
        # Variation tolerance
        lighting_tolerance: float = 0.3,
        entity_churn_tolerance: float = 0.2,
        
        # History settings
        fingerprint_history_size: int = 20,
    ) -> None:
        """Initialize the stability guard.
        
        Args:
            similarity_threshold: Min similarity to consider same identity.
            mismatch_frames_for_uncertainty: Frames of mismatch → uncertain state.
            mismatch_frames_for_transition: Frames of mismatch → allow transition.
            lighting_tolerance: How much lighting variation to tolerate.
            entity_churn_tolerance: How much entity churn to tolerate.
            fingerprint_history_size: How many fingerprints to retain.
        """
        self._similarity_threshold = similarity_threshold
        self._mismatch_frames_for_uncertainty = mismatch_frames_for_uncertainty
        self._mismatch_frames_for_transition = mismatch_frames_for_transition
        self._lighting_tolerance = lighting_tolerance
        self._entity_churn_tolerance = entity_churn_tolerance
        self._fingerprint_history_size = fingerprint_history_size
        
        # State tracking
        self._current_state = StabilityState.STABLE
        self._fingerprint_history: list[ACFFingerprint] = []
        self._baseline_fingerprint: ACFFingerprint | None = None
        
        # Mismatch tracking
        self._consecutive_mismatch_frames = 0
        self._consecutive_stable_frames = 0
        
        # Statistics
        self._total_evaluations = 0
        self._identity_preservations = 0
        self._transitions_allowed = 0
        self._anomalies_detected = 0
    
    @property
    def current_state(self) -> StabilityState:
        """Current stability state."""
        return self._current_state
    
    @property
    def is_stable(self) -> bool:
        """Whether identity is currently stable."""
        return self._current_state == StabilityState.STABLE
    
    @property
    def is_uncertain(self) -> bool:
        """Whether identity is currently uncertain."""
        return self._current_state == StabilityState.UNCERTAIN
    
    def extract_fingerprint(
        self,
        percept: "Percept",
        acf: "ActiveContextFrame",
    ) -> ACFFingerprint:
        """Extract identity fingerprint from current perception.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            Fingerprint capturing identity-relevant features.
        """
        # Extract entity information
        entity_guids = set()
        entity_positions = {}
        
        if percept and percept.candidates:
            for candidate in percept.candidates:
                if hasattr(candidate, 'candidate_id') and candidate.candidate_id:
                    entity_guids.add(candidate.candidate_id)
                    if hasattr(candidate, 'position') and candidate.position:
                        entity_positions[candidate.candidate_id] = candidate.position
        
        # Also include entities from ACF
        if acf and acf.entities:
            for entity in acf.entities:
                if hasattr(entity, 'candidate_id') and entity.candidate_id:
                    entity_guids.add(entity.candidate_id)
        
        return ACFFingerprint(
            entity_count=len(entity_guids),
            entity_guids=frozenset(entity_guids),
            entity_relative_positions=entity_positions,
            location_label=acf.location_label if acf else "unknown",
            location_confidence=acf.location_confidence if acf else 0.0,
        )
    
    def evaluate_stability(
        self,
        percept: "Percept",
        acf: "ActiveContextFrame",
        motion_detected: bool = False,
    ) -> StabilityDecision:
        """Evaluate whether current perception should change identity.
        
        This is the core stability enforcement method. It:
        1. Extracts fingerprint from current perception
        2. Compares to baseline/history
        3. Decides whether variation should fragment identity
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            motion_detected: Whether significant motion was detected.
            
        Returns:
            StabilityDecision indicating whether identity is preserved.
        """
        self._total_evaluations += 1
        
        # Extract current fingerprint
        current_fp = self.extract_fingerprint(percept, acf)
        
        # First frame - establish baseline
        if self._baseline_fingerprint is None:
            self._baseline_fingerprint = current_fp
            self._fingerprint_history.append(current_fp)
            return StabilityDecision(
                identity_stable=True,
                new_state=StabilityState.STABLE,
                reason="Initial baseline established",
                evidence={"fingerprint": current_fp.__dict__},
            )
        
        # Compute similarity to baseline
        similarity = current_fp.similarity_to(self._baseline_fingerprint)
        
        # Add to history
        self._fingerprint_history.append(current_fp)
        if len(self._fingerprint_history) > self._fingerprint_history_size:
            self._fingerprint_history.pop(0)
        
        # Case 1: High similarity - identity stable
        if similarity >= self._similarity_threshold:
            self._consecutive_mismatch_frames = 0
            self._consecutive_stable_frames += 1
            self._identity_preservations += 1
            
            # Reset to stable if we were uncertain
            if self._current_state == StabilityState.UNCERTAIN:
                if self._consecutive_stable_frames >= 3:
                    self._current_state = StabilityState.STABLE
                    return StabilityDecision(
                        identity_stable=True,
                        new_state=StabilityState.STABLE,
                        reason="Stability restored after uncertainty",
                        evidence={"similarity": similarity, "stable_frames": self._consecutive_stable_frames},
                    )
            
            self._current_state = StabilityState.STABLE
            return StabilityDecision(
                identity_stable=True,
                new_state=StabilityState.STABLE,
                confidence_adjustment=0.0,
                reason="High similarity to baseline",
                evidence={"similarity": similarity},
            )
        
        # Case 2: Low similarity WITHOUT motion - likely anomaly/variation
        if not motion_detected:
            self._consecutive_mismatch_frames += 1
            self._consecutive_stable_frames = 0
            
            # Check if this looks like lighting/shadow variation
            variation_type = self._classify_variation(current_fp, self._baseline_fingerprint)
            
            if variation_type in (VariationType.LIGHTING, VariationType.SHADOW, VariationType.NOISE):
                # Tolerate lighting variations - don't fragment identity
                self._anomalies_detected += 1
                logger.info(
                    f"[STABILITY] Tolerating {variation_type.value} variation, "
                    f"preserving identity (similarity={similarity:.2f})"
                )
                return StabilityDecision(
                    identity_stable=True,
                    new_state=self._current_state,
                    variation_type=variation_type,
                    confidence_adjustment=-0.1,  # Slight confidence penalty
                    reason=f"Tolerating {variation_type.value} variation without motion",
                    evidence={"similarity": similarity, "variation": variation_type.value},
                )
            
            # Non-lighting variation without motion = anomaly
            if self._consecutive_mismatch_frames < self._mismatch_frames_for_uncertainty:
                # Not enough frames to be uncertain yet
                self._current_state = StabilityState.MONITORING
                return StabilityDecision(
                    identity_stable=True,
                    new_state=StabilityState.MONITORING,
                    confidence_adjustment=-0.05,
                    reason="Visual change without motion - monitoring",
                    evidence={
                        "similarity": similarity,
                        "mismatch_frames": self._consecutive_mismatch_frames,
                        "threshold": self._mismatch_frames_for_uncertainty,
                    },
                )
            
            # Sustained mismatch without motion = enter uncertainty
            self._current_state = StabilityState.UNCERTAIN
            self._anomalies_detected += 1
            return StabilityDecision(
                identity_stable=True,  # Still preserve identity, but flag uncertainty
                new_state=StabilityState.UNCERTAIN,
                confidence_adjustment=-0.2,
                reason="Sustained visual change without motion - anomaly, entering uncertainty",
                evidence={
                    "similarity": similarity,
                    "mismatch_frames": self._consecutive_mismatch_frames,
                },
            )
        
        # Case 3: Low similarity WITH motion - potential legitimate transition
        self._consecutive_mismatch_frames += 1
        self._consecutive_stable_frames = 0
        
        if self._consecutive_mismatch_frames < self._mismatch_frames_for_transition:
            # Not enough evidence yet
            self._current_state = StabilityState.MONITORING
            return StabilityDecision(
                identity_stable=True,
                new_state=StabilityState.MONITORING,
                confidence_adjustment=-0.1,
                reason="Visual change with motion - monitoring for transition",
                evidence={
                    "similarity": similarity,
                    "mismatch_frames": self._consecutive_mismatch_frames,
                    "threshold": self._mismatch_frames_for_transition,
                },
            )
        
        # Sustained mismatch with motion = allow transition
        self._current_state = StabilityState.TRANSITIONING
        self._transitions_allowed += 1
        return StabilityDecision(
            identity_stable=False,  # Allow identity change
            new_state=StabilityState.TRANSITIONING,
            reason="Sustained visual change with motion - allowing transition",
            evidence={
                "similarity": similarity,
                "mismatch_frames": self._consecutive_mismatch_frames,
            },
        )
    
    def _classify_variation(
        self,
        current: ACFFingerprint,
        baseline: ACFFingerprint,
    ) -> VariationType:
        """Classify the type of perceptual variation.
        
        Args:
            current: Current fingerprint.
            baseline: Baseline fingerprint.
            
        Returns:
            Classification of variation type.
        """
        # If entities are mostly the same, variation is likely lighting/shadow
        entity_overlap = len(current.entity_guids & baseline.entity_guids)
        total_entities = max(len(current.entity_guids | baseline.entity_guids), 1)
        entity_stability = entity_overlap / total_entities
        
        if entity_stability > (1.0 - self._entity_churn_tolerance):
            # Entities stable, visual difference = lighting/shadow
            return VariationType.LIGHTING
        
        # Significant entity change
        if len(current.entity_guids - baseline.entity_guids) > len(baseline.entity_guids) * 0.3:
            # Many new entities = clutter
            return VariationType.CLUTTER
        
        if len(baseline.entity_guids - current.entity_guids) > len(baseline.entity_guids) * 0.3:
            # Many missing entities = occlusion
            return VariationType.OCCLUSION
        
        # Default to noise
        return VariationType.NOISE
    
    def confirm_transition(self, new_location_label: str) -> None:
        """Confirm a location transition and update baseline.
        
        Call this when a legitimate transition has been confirmed
        (e.g., by user or sustained different perception).
        
        Args:
            new_location_label: The confirmed new location label.
        """
        if self._fingerprint_history:
            # Use recent fingerprint as new baseline
            self._baseline_fingerprint = self._fingerprint_history[-1]
            self._baseline_fingerprint.location_label = new_location_label
        
        # Reset state
        self._current_state = StabilityState.STABLE
        self._consecutive_mismatch_frames = 0
        self._consecutive_stable_frames = 0
        
        logger.info(
            f"[STABILITY] Transition confirmed to '{new_location_label}', "
            f"baseline updated"
        )
    
    def reset(self) -> None:
        """Reset the stability guard to initial state."""
        self._current_state = StabilityState.STABLE
        self._fingerprint_history.clear()
        self._baseline_fingerprint = None
        self._consecutive_mismatch_frames = 0
        self._consecutive_stable_frames = 0
        logger.info("[STABILITY] Guard reset to initial state")
    
    def get_statistics(self) -> dict[str, Any]:
        """Get guard statistics for logging/debugging.
        
        Returns:
            Dictionary of statistics.
        """
        return {
            "total_evaluations": self._total_evaluations,
            "identity_preservations": self._identity_preservations,
            "transitions_allowed": self._transitions_allowed,
            "anomalies_detected": self._anomalies_detected,
            "current_state": self._current_state.value,
            "consecutive_mismatch_frames": self._consecutive_mismatch_frames,
            "consecutive_stable_frames": self._consecutive_stable_frames,
            "fingerprint_history_size": len(self._fingerprint_history),
        }
