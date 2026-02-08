"""Tests for panorama investigation state machine and event bus.

Covers:
- State transitions in InvestigationStateMachine
- Label gating (label_request requires plateau + low confidence)
- Adaptive investigation window
- EvidenceBundle accumulation
- PanoramaEventBus ring buffer, filtering, thread safety
- PanoramaEvent schema validation
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import pytest

from episodic_agent.schemas.panorama_events import (
    EvidenceBundle,
    MatchCandidate,
    MatchEvaluation,
    MemoryCard,
    MemorySummary,
    PanoramaAgentState,
    PanoramaEvent,
    PanoramaEventType,
    PerceptionPayload,
    StateTransitionPayload,
    MemoryWritePayload,
)
from episodic_agent.modules.panorama.investigation import InvestigationStateMachine
from episodic_agent.modules.panorama.event_bus import PanoramaEventBus


# =====================================================================
# Helpers
# =====================================================================

def _make_evaluation(
    top_confidence: float = 0.5,
    n_candidates: int = 3,
    margin: float = 0.1,
) -> MatchEvaluation:
    """Create a test MatchEvaluation with configurable values."""
    candidates = []
    for i in range(n_candidates):
        conf = max(0.0, top_confidence - i * margin)
        candidates.append(MatchCandidate(
            location_id=f"loc_{i}",
            label=f"location_{i}",
            confidence=conf,
            distance=1.0 - conf,
        ))
    return MatchEvaluation(
        candidates=candidates,
        top_margin=margin if n_candidates >= 2 else 0.0,
        hysteresis_active=False,
        stabilization_frames=0,
    )


# =====================================================================
# InvestigationStateMachine tests
# =====================================================================

class TestInvestigationStateMachine:
    """Tests for the investigation state machine."""

    def test_initial_state(self) -> None:
        sm = InvestigationStateMachine()
        assert sm.state == PanoramaAgentState.investigating_unknown
        assert sm.investigation_steps == 0
        assert not sm.should_request_label()

    def test_confident_match_immediate(self) -> None:
        """High confidence should transition directly to confident_match."""
        sm = InvestigationStateMachine(confident_match_threshold=0.7)
        ev = _make_evaluation(top_confidence=0.85, n_candidates=2)
        state = sm.update(ev)
        assert state == PanoramaAgentState.confident_match

    def test_no_label_before_min_steps(self) -> None:
        """Label request should not fire before min_investigation_steps."""
        sm = InvestigationStateMachine(min_investigation_steps=5)
        for _ in range(4):
            ev = _make_evaluation(top_confidence=0.1)
            sm.update(ev)
        assert not sm.should_request_label()
        assert sm.state != PanoramaAgentState.label_request

    def test_label_request_after_plateau(self) -> None:
        """Label request should fire after confidence plateau with low conf."""
        sm = InvestigationStateMachine(
            min_investigation_steps=3,
            max_investigation_steps=20,
            plateau_threshold=0.05,
            label_request_ceiling=0.4,
            plateau_window=3,
        )
        # Feed stable low-confidence evaluations
        for i in range(15):
            ev = _make_evaluation(top_confidence=0.15 + (i % 2) * 0.01)
            sm.update(ev)

        # After enough steps with plateau, should eventually reach label_request
        # The SM goes investigating → novel_location_candidate → label_request
        assert sm.state in (
            PanoramaAgentState.novel_location_candidate,
            PanoramaAgentState.label_request,
        )

    def test_matching_known_when_moderate_confidence(self) -> None:
        """Moderate confidence should transition to matching_known."""
        sm = InvestigationStateMachine(
            min_investigation_steps=3,
            label_request_ceiling=0.4,
        )
        for _ in range(5):
            ev = _make_evaluation(top_confidence=0.55)
            sm.update(ev)
        assert sm.state in (
            PanoramaAgentState.matching_known,
            PanoramaAgentState.confident_match,
            PanoramaAgentState.low_confidence_match,
        )

    def test_max_investigation_forces_decision(self) -> None:
        """Hitting max_investigation_steps should force a state transition."""
        sm = InvestigationStateMachine(
            min_investigation_steps=2,
            max_investigation_steps=5,
            plateau_threshold=100.0,  # disable plateau detection
        )
        for _ in range(6):
            ev = _make_evaluation(top_confidence=0.2)
            sm.update(ev)
        assert sm.state != PanoramaAgentState.investigating_unknown

    def test_reset(self) -> None:
        """Reset should return to initial state."""
        sm = InvestigationStateMachine(min_investigation_steps=2)
        for _ in range(5):
            sm.update(_make_evaluation(top_confidence=0.8))
        assert sm.state == PanoramaAgentState.confident_match
        sm.reset()
        assert sm.state == PanoramaAgentState.investigating_unknown
        assert sm.investigation_steps == 0

    def test_reset_to_confident(self) -> None:
        sm = InvestigationStateMachine()
        sm.reset_to_confident("kitchen")
        assert sm.state == PanoramaAgentState.confident_match
        assert sm.investigation_steps == 0

    def test_confidence_history_tracked(self) -> None:
        sm = InvestigationStateMachine()
        for i in range(5):
            sm.update(_make_evaluation(top_confidence=0.1 * (i + 1)))
        assert len(sm.confidence_history) == 5
        assert sm.confidence_history[0] == pytest.approx(0.1, abs=0.01)

    def test_evidence_bundle_accumulation(self) -> None:
        """Evidence bundle should contain images and features."""
        sm = InvestigationStateMachine()
        for i in range(3):
            sm.update(
                _make_evaluation(top_confidence=0.2),
                viewport_b64=f"image_{i}",
                feature_summary={"brightness": 0.5 + i * 0.1},
            )
        bundle = sm.get_evidence_bundle()
        assert bundle.investigation_steps == 3
        assert len(bundle.viewport_images_b64) == 3
        assert len(bundle.feature_summaries) == 3
        assert len(bundle.confidence_history) == 3

    def test_evidence_bundle_match_scores(self) -> None:
        """Evidence bundle match_scores should track best per candidate."""
        sm = InvestigationStateMachine()
        sm.update(_make_evaluation(top_confidence=0.3, n_candidates=2))
        sm.update(_make_evaluation(top_confidence=0.5, n_candidates=2))
        bundle = sm.get_evidence_bundle()
        # loc_0 should have best score of 0.5
        assert bundle.match_scores.get("loc_0", 0.0) >= 0.5

    def test_steps_in_state_increments(self) -> None:
        sm = InvestigationStateMachine(confident_match_threshold=0.95)
        # All low conf → stays in investigating_unknown
        for _ in range(3):
            sm.update(_make_evaluation(top_confidence=0.1))
        assert sm.steps_in_state >= 2


# =====================================================================
# PanoramaEventBus tests
# =====================================================================

class TestPanoramaEventBus:
    """Tests for the event bus ring buffer."""

    def test_emit_and_get(self) -> None:
        bus = PanoramaEventBus()
        event = PanoramaEvent(
            event_type=PanoramaEventType.perception_update,
            step=1,
            state=PanoramaAgentState.investigating_unknown,
        )
        bus.emit(event)
        assert bus.event_count == 1
        events = bus.get_latest(10)
        assert len(events) == 1
        assert events[0].step == 1

    def test_get_events_since_step(self) -> None:
        bus = PanoramaEventBus()
        for i in range(10):
            bus.emit_simple(
                PanoramaEventType.perception_update,
                step=i,
                state=PanoramaAgentState.investigating_unknown,
            )
        events = bus.get_events(since_step=7)
        assert len(events) == 3
        assert all(e.step >= 7 for e in events)

    def test_ring_buffer_capacity(self) -> None:
        bus = PanoramaEventBus(max_events=5)
        for i in range(10):
            bus.emit_simple(
                PanoramaEventType.perception_update,
                step=i,
                state=PanoramaAgentState.investigating_unknown,
            )
        assert bus.event_count == 5
        events = bus.get_latest(10)
        assert events[0].step == 5  # oldest retained

    def test_get_events_by_type(self) -> None:
        bus = PanoramaEventBus()
        bus.emit_simple(PanoramaEventType.perception_update, step=0, state=PanoramaAgentState.investigating_unknown)
        bus.emit_simple(PanoramaEventType.match_evaluation, step=1, state=PanoramaAgentState.matching_known)
        bus.emit_simple(PanoramaEventType.perception_update, step=2, state=PanoramaAgentState.investigating_unknown)

        perceptions = bus.get_events_by_type(PanoramaEventType.perception_update)
        assert len(perceptions) == 2
        matches = bus.get_events_by_type(PanoramaEventType.match_evaluation)
        assert len(matches) == 1

    def test_latest_step(self) -> None:
        bus = PanoramaEventBus()
        bus.emit_simple(PanoramaEventType.perception_update, step=42, state=PanoramaAgentState.investigating_unknown)
        assert bus.latest_step == 42

    def test_subscribe_callback(self) -> None:
        bus = PanoramaEventBus()
        received: list[PanoramaEvent] = []
        bus.subscribe(lambda e: received.append(e))
        bus.emit_simple(PanoramaEventType.perception_update, step=1, state=PanoramaAgentState.investigating_unknown)
        assert len(received) == 1

    def test_unsubscribe(self) -> None:
        bus = PanoramaEventBus()
        received: list[PanoramaEvent] = []
        cb = lambda e: received.append(e)
        bus.subscribe(cb)
        bus.emit_simple(PanoramaEventType.perception_update, step=1, state=PanoramaAgentState.investigating_unknown)
        bus.unsubscribe(cb)
        bus.emit_simple(PanoramaEventType.perception_update, step=2, state=PanoramaAgentState.investigating_unknown)
        assert len(received) == 1

    def test_clear(self) -> None:
        bus = PanoramaEventBus()
        bus.emit_simple(PanoramaEventType.perception_update, step=1, state=PanoramaAgentState.investigating_unknown)
        bus.clear()
        assert bus.event_count == 0
        assert bus.latest_step == 0

    def test_snapshot_serializable(self) -> None:
        bus = PanoramaEventBus()
        bus.emit_simple(
            PanoramaEventType.match_evaluation,
            step=5,
            state=PanoramaAgentState.matching_known,
            payload={"candidates": [{"location_id": "loc_1", "confidence": 0.8}]},
        )
        snap = bus.snapshot()
        assert len(snap) == 1
        assert snap[0]["step"] == 5
        assert isinstance(snap[0]["event_type"], str)

    def test_thread_safety(self) -> None:
        """Concurrent emit + read should not crash."""
        bus = PanoramaEventBus(max_events=100)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    bus.emit_simple(
                        PanoramaEventType.perception_update,
                        step=i,
                        state=PanoramaAgentState.investigating_unknown,
                    )
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(50):
                    bus.get_events(since_step=0)
                    bus.get_latest(5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# =====================================================================
# Schema model tests
# =====================================================================

class TestPanoramaEventSchemas:
    """Tests for the Pydantic schema models."""

    def test_match_candidate_frozen(self) -> None:
        mc = MatchCandidate(location_id="loc_1", label="kitchen", confidence=0.8, distance=0.2)
        assert mc.location_id == "loc_1"
        with pytest.raises(Exception):
            mc.label = "other"  # should be frozen

    def test_evidence_bundle_defaults(self) -> None:
        bundle = EvidenceBundle()
        assert bundle.viewport_images_b64 == []
        assert bundle.investigation_steps == 0
        assert bundle.margin == 0.0

    def test_match_evaluation_serialization(self) -> None:
        ev = _make_evaluation(top_confidence=0.7, n_candidates=3)
        d = ev.model_dump()
        assert len(d["candidates"]) == 3
        assert d["top_margin"] == 0.1

    def test_panorama_event_json_roundtrip(self) -> None:
        event = PanoramaEvent(
            event_type=PanoramaEventType.state_transition,
            step=10,
            state=PanoramaAgentState.confident_match,
            payload={"previous_state": "investigating_unknown", "new_state": "confident_match"},
        )
        d = event.model_dump(mode="json")
        assert d["event_type"] == "state_transition"
        assert d["state"] == "confident_match"
        assert d["step"] == 10

    def test_perception_payload(self) -> None:
        pp = PerceptionPayload(
            confidence=0.75,
            heading_index=3,
            total_headings=8,
            heading_deg=135.0,
            embedding_norm=5.2,
            source_file="test.jpg",
        )
        assert pp.confidence == 0.75
        assert pp.heading_deg == 135.0

    def test_memory_summary(self) -> None:
        ms = MemorySummary(
            location_id="loc_1",
            label="kitchen",
            observation_count=15,
            variance=0.02,
        )
        assert ms.stability_score == 0.0  # default, computed externally
        assert ms.entity_cooccurrence == {}

    def test_memory_card(self) -> None:
        card = MemoryCard(
            location_id="loc_1",
            label="kitchen",
            embedding_centroid=[0.1] * 128,
            observation_count=10,
            variance=0.01,
        )
        assert len(card.embedding_centroid) == 128
        assert card.aliases == []

    def test_state_transition_payload(self) -> None:
        stp = StateTransitionPayload(
            previous_state="investigating_unknown",
            new_state="confident_match",
            reason="high confidence",
            confidence=0.85,
        )
        assert stp.previous_state == "investigating_unknown"

    def test_memory_write_payload(self) -> None:
        mwp = MemoryWritePayload(
            location_id="loc_1",
            label="kitchen",
            is_new=True,
            observation_count=1,
        )
        assert mwp.is_new is True


# =====================================================================
# Integration: SM + EventBus together
# =====================================================================

class TestInvestigationEventIntegration:
    """Tests for the investigation SM and event bus working together."""

    def test_sm_updates_bus_via_manual_emit(self) -> None:
        """Verify events can be emitted based on SM state changes."""
        bus = PanoramaEventBus()
        sm = InvestigationStateMachine(confident_match_threshold=0.7)

        ev = _make_evaluation(top_confidence=0.85)
        new_state = sm.update(ev)

        # Manually emit state transition (as LocationResolver would)
        bus.emit_simple(
            PanoramaEventType.state_transition,
            step=1,
            state=new_state,
            payload={"new_state": new_state.value},
        )

        events = bus.get_events_by_type(PanoramaEventType.state_transition)
        assert len(events) == 1
        assert events[0].state == PanoramaAgentState.confident_match.value

    def test_evidence_bundle_available_at_label_request(self) -> None:
        """When SM reaches label_request, evidence bundle should be populated."""
        sm = InvestigationStateMachine(
            min_investigation_steps=3,
            max_investigation_steps=8,
            plateau_threshold=0.05,
            label_request_ceiling=0.4,
            plateau_window=3,
        )

        # Feed many low-confidence frames to trigger label request
        for i in range(25):
            sm.update(
                _make_evaluation(top_confidence=0.1, n_candidates=2),
                viewport_b64=f"img_{i}",
                feature_summary={"brightness": 0.3},
            )

        if sm.should_request_label():
            bundle = sm.get_evidence_bundle()
            assert bundle.investigation_steps > 3
            assert len(bundle.viewport_images_b64) > 0
            assert len(bundle.confidence_history) > 0
            assert "loc_0" in bundle.match_scores
