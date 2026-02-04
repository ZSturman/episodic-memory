"""Tests for Phase 3: ACF Stability + Motion/Perception Arbitration.

These tests verify the core recognition invariants that protect identity stability:

1. ACF Stability Guard:
   - Lighting/shadow changes do NOT fragment identity
   - Sudden visual change without motion → candidate anomaly, not new location
   - Only sustained mismatch + spatial contradiction triggers uncertainty

2. Motion-Perception Arbitrator:
   - Motion is advisory, perception is authoritative
   - Never trust motion alone for location change
   - Investigation required before transitioning

Test Scenarios:
   - Lighting change in same room → location label persists
   - Teleport to visually identical room → uncertainty state triggered
   - Elevator: motion discontinuity + visual match → does not auto-relabel
   - Normal walk with visual change → allows transition after investigation
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from episodic_agent.modules.acf_stability import (
    ACFStabilityGuard,
    ACFFingerprint,
    StabilityState,
    StabilityDecision,
    VariationType,
)
from episodic_agent.modules.arbitrator import (
    MotionPerceptionArbitrator,
    MotionSignal,
    PerceptionSignal,
    ArbitrationDecision,
    ArbitrationOutcome,
    ConflictType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stability_guard():
    """Create an ACF stability guard with default settings."""
    return ACFStabilityGuard(
        similarity_threshold=0.6,
        mismatch_frames_for_uncertainty=3,
        mismatch_frames_for_transition=5,
    )


@pytest.fixture
def arbitrator():
    """Create a motion-perception arbitrator with default settings."""
    return MotionPerceptionArbitrator(
        significant_motion_threshold=0.5,
        motion_discontinuity_threshold=5.0,
        investigation_frames=3,
    )


@pytest.fixture
def mock_percept():
    """Create a mock percept for testing."""
    percept = Mock()
    percept.candidates = []
    percept.extras = {}
    return percept


@pytest.fixture
def mock_acf():
    """Create a mock ACF for testing."""
    acf = Mock()
    acf.location_label = "living_room"
    acf.location_confidence = 0.8
    acf.entities = []
    return acf


def create_percept_with_entities(entity_ids: list[str]) -> Mock:
    """Create a percept with specific entity IDs."""
    percept = Mock()
    percept.candidates = []
    for eid in entity_ids:
        candidate = Mock()
        candidate.candidate_id = eid
        candidate.position = (0, 0, 0)
        percept.candidates.append(candidate)
    percept.extras = {}
    return percept


def create_acf(location: str = "living_room", confidence: float = 0.8, entities: list[str] = None) -> Mock:
    """Create an ACF with specific settings."""
    acf = Mock()
    acf.location_label = location
    acf.location_confidence = confidence
    acf.entities = []
    if entities:
        for eid in entities:
            entity = Mock()
            entity.candidate_id = eid
            acf.entities.append(entity)
    return acf


# =============================================================================
# ACFFingerprint Tests
# =============================================================================


class TestACFFingerprint:
    """Tests for ACF fingerprint extraction and comparison."""
    
    def test_fingerprint_creation(self):
        """Test basic fingerprint creation."""
        fp = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e3"]),
            location_label="kitchen",
            location_confidence=0.9,
        )
        assert fp.entity_count == 3
        assert len(fp.entity_guids) == 3
        assert fp.location_label == "kitchen"
    
    def test_fingerprint_similarity_identical(self):
        """Identical fingerprints should have similarity = 1.0."""
        fp1 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e3"]),
            location_label="kitchen",
            location_confidence=0.9,
        )
        fp2 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e3"]),
            location_label="kitchen",
            location_confidence=0.9,
        )
        assert fp1.similarity_to(fp2) == 1.0
    
    def test_fingerprint_similarity_different_location(self):
        """Different locations should reduce similarity."""
        fp1 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e3"]),
            location_label="kitchen",
        )
        fp2 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e3"]),
            location_label="bedroom",
        )
        similarity = fp1.similarity_to(fp2)
        assert similarity < 1.0  # Different location = lower similarity
        assert similarity >= 0.4  # Same entities = still some similarity
    
    def test_fingerprint_similarity_different_entities(self):
        """Different entity sets should reduce similarity."""
        fp1 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e3"]),
            location_label="kitchen",
        )
        fp2 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e4", "e5", "e6"]),
            location_label="kitchen",
        )
        similarity = fp1.similarity_to(fp2)
        assert similarity < 1.0  # Different entities
    
    def test_fingerprint_similarity_partial_overlap(self):
        """Partial entity overlap should give intermediate similarity."""
        fp1 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e3"]),
            location_label="kitchen",
        )
        fp2 = ACFFingerprint(
            entity_count=3,
            entity_guids=frozenset(["e1", "e2", "e4"]),
            location_label="kitchen",
        )
        similarity = fp1.similarity_to(fp2)
        # 2/4 entity overlap (Jaccard), same location
        assert 0.5 < similarity < 1.0


# =============================================================================
# ACFStabilityGuard Tests
# =============================================================================


class TestACFStabilityGuardBasics:
    """Basic functionality tests for ACFStabilityGuard."""
    
    def test_initial_state(self, stability_guard):
        """Guard should start in stable state."""
        assert stability_guard.current_state == StabilityState.STABLE
        assert stability_guard.is_stable
        assert not stability_guard.is_uncertain
    
    def test_first_frame_establishes_baseline(self, stability_guard, mock_percept, mock_acf):
        """First evaluation should establish baseline."""
        decision = stability_guard.evaluate_stability(mock_percept, mock_acf)
        
        assert decision.identity_stable
        assert decision.new_state == StabilityState.STABLE
        assert "baseline" in decision.reason.lower()
    
    def test_stable_perception_maintains_identity(self, stability_guard):
        """Stable perception should maintain identity."""
        entities = ["e1", "e2", "e3"]
        
        # Establish baseline
        percept1 = create_percept_with_entities(entities)
        acf1 = create_acf("living_room", 0.8, entities)
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Same perception
        percept2 = create_percept_with_entities(entities)
        acf2 = create_acf("living_room", 0.8, entities)
        decision = stability_guard.evaluate_stability(percept2, acf2)
        
        assert decision.identity_stable
        assert decision.new_state == StabilityState.STABLE
    
    def test_extract_fingerprint(self, stability_guard):
        """Should correctly extract fingerprint from percept/ACF."""
        entities = ["e1", "e2", "e3"]
        percept = create_percept_with_entities(entities)
        acf = create_acf("kitchen", 0.9, entities)
        
        fp = stability_guard.extract_fingerprint(percept, acf)
        
        assert fp.entity_count == 3
        assert fp.location_label == "kitchen"
        assert fp.location_confidence == 0.9


class TestACFStabilityLightingInvariant:
    """Tests for: Lighting/shadow changes do NOT fragment identity.
    
    ARCHITECTURAL INVARIANT: ACF Stability
    Lighting, shadows, temporary clutter do not fragment identity.
    """
    
    def test_lighting_change_preserves_identity(self, stability_guard):
        """INVARIANT: Lighting change in same room → location label persists."""
        entities = ["e1", "e2", "e3"]
        
        # Establish baseline in bright lighting
        percept1 = create_percept_with_entities(entities)
        acf1 = create_acf("living_room", 0.9, entities)
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Same entities, but visual difference (simulated by slightly different fingerprint)
        # In reality, this would come from different visual features
        percept2 = create_percept_with_entities(entities)  # Same entities
        acf2 = create_acf("living_room", 0.7, entities)  # Lower confidence due to lighting
        
        # No motion detected
        decision = stability_guard.evaluate_stability(percept2, acf2, motion_detected=False)
        
        # INVARIANT: Identity should persist despite visual variation
        assert decision.identity_stable, "Lighting change should NOT fragment identity"
        assert decision.new_state in (StabilityState.STABLE, StabilityState.MONITORING)
    
    def test_shadow_variation_preserves_identity(self, stability_guard):
        """Shadow variations should not trigger identity change."""
        entities = ["e1", "e2", "e3"]
        
        # Establish baseline
        percept1 = create_percept_with_entities(entities)
        acf1 = create_acf("hallway", 0.85, entities)
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Same entities, same location (shadow variation)
        percept2 = create_percept_with_entities(entities)
        acf2 = create_acf("hallway", 0.75, entities)
        
        decision = stability_guard.evaluate_stability(percept2, acf2, motion_detected=False)
        
        assert decision.identity_stable
    
    def test_temporary_clutter_preserves_identity(self, stability_guard):
        """Temporary clutter appearing should not fragment identity."""
        base_entities = ["e1", "e2", "e3"]
        
        # Establish baseline
        percept1 = create_percept_with_entities(base_entities)
        acf1 = create_acf("kitchen", 0.9, base_entities)
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Additional entity appears (temporary clutter)
        cluttered_entities = base_entities + ["clutter_1"]
        percept2 = create_percept_with_entities(cluttered_entities)
        acf2 = create_acf("kitchen", 0.85, cluttered_entities)
        
        decision = stability_guard.evaluate_stability(percept2, acf2, motion_detected=False)
        
        # Identity should persist - clutter doesn't mean new location
        assert decision.identity_stable


class TestACFStabilityAnomalyDetection:
    """Tests for: Sudden visual change without motion → anomaly, not new location.
    
    ARCHITECTURAL INVARIANT: ACF Stability
    Only sustained mismatch + spatial contradiction triggers uncertainty.
    """
    
    def test_sudden_visual_change_without_motion_is_anomaly(self, stability_guard):
        """Sudden visual change without motion should NOT auto-relabel."""
        # Establish baseline
        percept1 = create_percept_with_entities(["e1", "e2", "e3"])
        acf1 = create_acf("living_room", 0.9, ["e1", "e2", "e3"])
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Completely different perception WITHOUT motion
        percept2 = create_percept_with_entities(["x1", "x2", "x3"])
        acf2 = create_acf("bedroom", 0.8, ["x1", "x2", "x3"])
        
        # No motion - this is anomalous
        decision = stability_guard.evaluate_stability(percept2, acf2, motion_detected=False)
        
        # Should preserve identity and flag as anomaly/uncertain
        assert decision.identity_stable, "Should NOT auto-relabel without motion"
        assert decision.new_state in (StabilityState.MONITORING, StabilityState.UNCERTAIN)
    
    def test_sustained_mismatch_without_motion_enters_uncertainty(self, stability_guard):
        """Sustained visual mismatch without motion → uncertainty state."""
        # Establish baseline
        percept1 = create_percept_with_entities(["e1", "e2", "e3"])
        acf1 = create_acf("living_room", 0.9, ["e1", "e2", "e3"])
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Sustained different perception
        for i in range(5):  # More than mismatch_frames_for_uncertainty
            percept = create_percept_with_entities(["x1", "x2", "x3"])
            acf = create_acf("bedroom", 0.8, ["x1", "x2", "x3"])
            decision = stability_guard.evaluate_stability(percept, acf, motion_detected=False)
        
        # Should be in uncertain state but still preserving identity
        assert decision.identity_stable
        assert decision.new_state == StabilityState.UNCERTAIN
    
    def test_stability_restored_after_uncertainty(self, stability_guard):
        """Returning to baseline after uncertainty should restore stability."""
        base_entities = ["e1", "e2", "e3"]
        
        # Establish baseline
        percept1 = create_percept_with_entities(base_entities)
        acf1 = create_acf("living_room", 0.9, base_entities)
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Create uncertainty
        for _ in range(5):
            percept = create_percept_with_entities(["x1", "x2"])
            acf = create_acf("unknown", 0.5)
            stability_guard.evaluate_stability(percept, acf, motion_detected=False)
        
        assert stability_guard.is_uncertain
        
        # Return to baseline
        for _ in range(5):
            percept = create_percept_with_entities(base_entities)
            acf = create_acf("living_room", 0.85, base_entities)
            decision = stability_guard.evaluate_stability(percept, acf, motion_detected=False)
        
        # Should restore to stable
        assert decision.identity_stable
        assert decision.new_state == StabilityState.STABLE


class TestACFStabilityTransition:
    """Tests for legitimate identity transitions."""
    
    def test_sustained_mismatch_with_motion_allows_transition(self, stability_guard):
        """Sustained visual change WITH motion should eventually allow transition."""
        # Establish baseline
        percept1 = create_percept_with_entities(["e1", "e2", "e3"])
        acf1 = create_acf("living_room", 0.9, ["e1", "e2", "e3"])
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Sustained different perception WITH motion
        for i in range(10):  # More than mismatch_frames_for_transition
            percept = create_percept_with_entities(["x1", "x2", "x3"])
            acf = create_acf("kitchen", 0.85, ["x1", "x2", "x3"])
            decision = stability_guard.evaluate_stability(percept, acf, motion_detected=True)
        
        # After sustained evidence, should allow transition
        assert not decision.identity_stable
        assert decision.new_state == StabilityState.TRANSITIONING
    
    def test_confirm_transition_updates_baseline(self, stability_guard):
        """Confirming transition should update baseline."""
        # Establish baseline
        percept1 = create_percept_with_entities(["e1", "e2", "e3"])
        acf1 = create_acf("living_room", 0.9, ["e1", "e2", "e3"])
        stability_guard.evaluate_stability(percept1, acf1)
        
        # Confirm transition to kitchen
        stability_guard.confirm_transition("kitchen")
        
        # New perception in kitchen should be stable
        percept2 = create_percept_with_entities(["k1", "k2", "k3"])
        acf2 = create_acf("kitchen", 0.9, ["k1", "k2", "k3"])
        decision = stability_guard.evaluate_stability(percept2, acf2)
        
        assert stability_guard.is_stable


# =============================================================================
# MotionPerceptionArbitrator Tests
# =============================================================================


class TestArbitratorBasics:
    """Basic functionality tests for MotionPerceptionArbitrator."""
    
    def test_initial_state(self, arbitrator):
        """Arbitrator should start with unknown location."""
        assert arbitrator.current_location == "unknown"
        assert not arbitrator.is_investigating
    
    def test_no_conflict_trusts_perception(self, arbitrator):
        """No conflict should trust perception."""
        motion = MotionSignal(
            displacement=(0.1, 0, 0),
            suggests_new_location=False,
        )
        perception = PerceptionSignal(
            perceived_location="living_room",
            location_confidence=0.9,
            acf_match_score=0.95,
        )
        
        decision = arbitrator.arbitrate(motion, perception)
        
        assert decision.outcome == ArbitrationOutcome.TRUST_PERCEPTION
        assert decision.resolved_location == "living_room"
        assert decision.conflict_type == ConflictType.NONE
    
    def test_creates_motion_signal(self, arbitrator, mock_percept, mock_acf):
        """Should create motion signal from percept."""
        signal = arbitrator.create_motion_signal(mock_percept, mock_acf)
        
        assert isinstance(signal, MotionSignal)
        assert signal.total_displacement >= 0
    
    def test_creates_perception_signal(self, arbitrator, mock_percept, mock_acf):
        """Should create perception signal from percept."""
        signal = arbitrator.create_perception_signal(mock_percept, mock_acf, acf_match_score=0.8)
        
        assert isinstance(signal, PerceptionSignal)
        assert signal.perceived_location == "living_room"
        assert signal.acf_match_score == 0.8


class TestArbitratorMotionAdvisory:
    """Tests for: Motion is advisory, perception is authoritative.
    
    ARCHITECTURAL INVARIANT: Motion Advisory / Perception Authoritative
    Never trust motion alone for location change.
    """
    
    def test_motion_alone_does_not_change_location(self, arbitrator):
        """INVARIANT: Motion alone should NOT auto-relabel location."""
        # Set initial location
        arbitrator.force_location("living_room", "initial")
        
        # Motion suggests new location
        motion = MotionSignal(
            displacement=(3.0, 0, 0),  # Significant motion
            total_displacement=3.0,
            suggests_new_location=True,
            suggested_location="kitchen",
            confidence=0.8,
        )
        
        # But perception still matches current location
        perception = PerceptionSignal(
            perceived_location="living_room",
            location_confidence=0.9,
            acf_match_score=0.9,  # High match to current ACF
        )
        
        decision = arbitrator.arbitrate(motion, perception)
        
        # INVARIANT: Should NOT auto-relabel based on motion
        assert decision.resolved_location == "living_room"
        assert decision.outcome != ArbitrationOutcome.TRUST_MOTION
    
    def test_perception_overrides_motion(self, arbitrator):
        """High-confidence perception should override motion suggestion."""
        arbitrator.force_location("living_room", "initial")
        
        motion = MotionSignal(
            displacement=(2.0, 0, 0),
            total_displacement=2.0,
            suggests_new_location=True,
            confidence=0.7,
        )
        
        perception = PerceptionSignal(
            perceived_location="living_room",
            location_confidence=0.95,  # Very high confidence
            acf_match_score=0.9,
        )
        
        decision = arbitrator.arbitrate(motion, perception)
        
        assert decision.outcome == ArbitrationOutcome.TRUST_PERCEPTION
        assert decision.resolved_location == "living_room"


class TestArbitratorElevatorScenario:
    """Tests for: Elevator scenario - motion discontinuity + visual match.
    
    ARCHITECTURAL INVARIANT: Motion discontinuity + visual match → don't auto-relabel.
    """
    
    def test_elevator_discontinuity_does_not_auto_relabel(self, arbitrator):
        """INVARIANT: Elevator (motion discontinuity + visual match) should NOT auto-relabel."""
        arbitrator.force_location("lobby_floor_1", "initial")
        
        # Large motion discontinuity (elevator movement)
        motion = MotionSignal(
            displacement=(0, 10.0, 0),  # Large vertical movement
            total_displacement=10.0,  # Over discontinuity threshold
            suggests_new_location=True,
            suggested_location="lobby_floor_2",
            confidence=0.9,
        )
        
        # But visual ACF still matches (identical lobby on different floor)
        perception = PerceptionSignal(
            perceived_location="lobby_floor_1",  # Looks the same
            location_confidence=0.85,
            acf_match_score=0.95,  # High match - looks identical
        )
        
        decision = arbitrator.arbitrate(motion, perception)
        
        # INVARIANT: Should NOT auto-relabel, should investigate
        assert decision.conflict_type == ConflictType.DISCONTINUITY
        assert decision.resolved_location == "lobby_floor_1"  # Keeps current
        assert decision.outcome == ArbitrationOutcome.INVESTIGATING
        assert decision.needs_investigation
    
    def test_elevator_requires_investigation(self, arbitrator):
        """Elevator scenario should require investigation frames."""
        arbitrator.force_location("lobby", "initial")
        
        motion = MotionSignal(
            displacement=(0, 8.0, 0),
            total_displacement=8.0,
            suggests_new_location=True,
        )
        perception = PerceptionSignal(
            perceived_location="lobby",
            location_confidence=0.8,
            acf_match_score=0.9,
        )
        
        decision = arbitrator.arbitrate(motion, perception)
        
        assert decision.needs_investigation
        assert decision.investigation_frames_remaining > 0


class TestArbitratorTeleportScenario:
    """Tests for: Teleport scenario - visual change without motion.
    
    ARCHITECTURAL INVARIANT: Sudden visual change without motion → uncertainty.
    """
    
    def test_teleport_triggers_uncertainty(self, arbitrator):
        """INVARIANT: Visual change without motion should trigger uncertainty."""
        arbitrator.force_location("living_room", "initial")
        
        # No significant motion
        motion = MotionSignal(
            displacement=(0.1, 0, 0),
            total_displacement=0.1,
            suggests_new_location=False,
        )
        
        # But perception shows completely different location
        perception = PerceptionSignal(
            perceived_location="bedroom",  # Different location
            location_confidence=0.8,
            acf_match_score=0.2,  # Low match - very different
        )
        
        decision = arbitrator.arbitrate(motion, perception)
        
        # Should trigger uncertainty, not auto-relabel
        assert decision.conflict_type == ConflictType.TELEPORT
        assert decision.resolved_location == "living_room"  # Keeps current
        assert decision.outcome == ArbitrationOutcome.UNCERTAIN


class TestArbitratorInvestigation:
    """Tests for investigation and resolution flow."""
    
    def test_investigation_collects_evidence(self, arbitrator):
        """Investigation should collect evidence over multiple frames."""
        arbitrator.force_location("room_a", "initial")
        
        # Trigger investigation
        motion = MotionSignal(
            displacement=(6.0, 0, 0),  # Discontinuity
            total_displacement=6.0,
            suggests_new_location=True,
        )
        perception = PerceptionSignal(
            perceived_location="room_a",
            location_confidence=0.8,
            acf_match_score=0.9,
        )
        
        decision = arbitrator.arbitrate(motion, perception)
        assert decision.outcome == ArbitrationOutcome.INVESTIGATING
        
        # Continue investigation
        for _ in range(2):
            decision = arbitrator.arbitrate(motion, perception)
            assert decision.outcome == ArbitrationOutcome.INVESTIGATING
    
    def test_investigation_resolves_with_perception_majority(self, arbitrator):
        """Investigation should resolve by trusting perception if supported by evidence."""
        arbitrator._investigation_frames = 3
        arbitrator.force_location("room_a", "initial")
        
        # Trigger investigation
        motion = MotionSignal(total_displacement=6.0, suggests_new_location=True)
        perception = PerceptionSignal(
            perceived_location="room_a",
            location_confidence=0.85,
            acf_match_score=0.9,
        )
        
        arbitrator.arbitrate(motion, perception)
        
        # Continue with consistent perception
        for _ in range(3):
            decision = arbitrator.arbitrate(motion, perception)
        
        # Should resolve to trust perception
        assert not arbitrator.is_investigating
        assert decision.outcome == ArbitrationOutcome.TRUST_PERCEPTION
    
    def test_force_location_ends_investigation(self, arbitrator):
        """Forcing location should end any ongoing investigation."""
        arbitrator._investigating = True
        arbitrator._investigation_frames_remaining = 5
        
        arbitrator.force_location("confirmed_room", "user_confirmed")
        
        assert not arbitrator.is_investigating
        assert arbitrator.current_location == "confirmed_room"


class TestArbitratorStatistics:
    """Tests for statistics tracking."""
    
    def test_tracks_arbitration_count(self, arbitrator):
        """Should track total arbitration count."""
        motion = MotionSignal()
        perception = PerceptionSignal(perceived_location="room", location_confidence=0.8)
        
        for _ in range(5):
            arbitrator.arbitrate(motion, perception)
        
        stats = arbitrator.get_statistics()
        assert stats["total_arbitrations"] == 5
    
    def test_tracks_conflicts(self, arbitrator):
        """Should track conflict count."""
        arbitrator.force_location("room_a", "initial")
        
        # Create conflict
        motion = MotionSignal(total_displacement=6.0, suggests_new_location=True)
        perception = PerceptionSignal(
            perceived_location="room_a",
            location_confidence=0.8,
            acf_match_score=0.9,
        )
        
        arbitrator.arbitrate(motion, perception)
        
        stats = arbitrator.get_statistics()
        assert stats["conflicts_detected"] >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestStabilityArbitratorIntegration:
    """Tests for combined stability guard and arbitrator."""
    
    def test_combined_workflow(self, stability_guard, arbitrator):
        """Test typical workflow with both components."""
        # Initial perception
        percept1 = create_percept_with_entities(["e1", "e2", "e3"])
        acf1 = create_acf("living_room", 0.9, ["e1", "e2", "e3"])
        
        # Stability guard establishes baseline
        stability_decision = stability_guard.evaluate_stability(percept1, acf1)
        assert stability_decision.identity_stable
        
        # Arbitrator confirms location
        motion = MotionSignal(suggests_new_location=False)
        perception = PerceptionSignal(
            perceived_location="living_room",
            location_confidence=0.9,
            acf_match_score=stability_decision.evidence.get("similarity", 0.9) if stability_decision.evidence else 0.9,
        )
        arb_decision = arbitrator.arbitrate(motion, perception)
        assert arb_decision.resolved_location == "living_room"
    
    def test_both_components_detect_anomaly(self, stability_guard, arbitrator):
        """Both components should flag anomalous situation."""
        # Establish baseline
        percept1 = create_percept_with_entities(["e1", "e2", "e3"])
        acf1 = create_acf("living_room", 0.9, ["e1", "e2", "e3"])
        stability_guard.evaluate_stability(percept1, acf1)
        arbitrator.force_location("living_room", "initial")
        
        # Anomalous change without motion
        percept2 = create_percept_with_entities(["x1", "x2"])
        acf2 = create_acf("unknown", 0.5)
        
        stability_decision = stability_guard.evaluate_stability(percept2, acf2, motion_detected=False)
        
        motion = MotionSignal(total_displacement=0.1, suggests_new_location=False)
        perception = PerceptionSignal(
            perceived_location="unknown",
            location_confidence=0.5,
            acf_match_score=0.3,
        )
        arb_decision = arbitrator.arbitrate(motion, perception)
        
        # Both should flag this as anomalous
        assert stability_decision.identity_stable  # Preserves identity
        assert stability_decision.new_state in (StabilityState.MONITORING, StabilityState.UNCERTAIN)
        assert arb_decision.conflict_type == ConflictType.TELEPORT


# =============================================================================
# Reset and Statistics Tests
# =============================================================================


class TestResetAndStatistics:
    """Tests for reset and statistics functionality."""
    
    def test_stability_guard_reset(self, stability_guard, mock_percept, mock_acf):
        """Reset should clear all state."""
        # Build up some state
        stability_guard.evaluate_stability(mock_percept, mock_acf)
        stability_guard.evaluate_stability(mock_percept, mock_acf)
        
        stability_guard.reset()
        
        assert stability_guard.current_state == StabilityState.STABLE
        assert stability_guard._baseline_fingerprint is None
    
    def test_arbitrator_reset(self, arbitrator):
        """Reset should clear all state."""
        arbitrator.force_location("some_room", "test")
        arbitrator.reset()
        
        assert arbitrator.current_location == "unknown"
        assert not arbitrator.is_investigating
    
    def test_stability_guard_statistics(self, stability_guard, mock_percept, mock_acf):
        """Statistics should track evaluations."""
        for _ in range(5):
            stability_guard.evaluate_stability(mock_percept, mock_acf)
        
        stats = stability_guard.get_statistics()
        assert stats["total_evaluations"] == 5
        assert "identity_preservations" in stats
        assert "current_state" in stats
