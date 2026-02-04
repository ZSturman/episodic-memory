"""Phase 5 tests for delta detection and event resolution.

Tests:
- Delta detection (new, missing, moved, state_changed)
- State-change event recognition
- Event labeling and learning
- Event pattern recognition
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from episodic_agent.modules.delta_detector import DeltaDetector, EntitySnapshot
from episodic_agent.modules.event_resolver import EventResolverStateChange
from episodic_agent.schemas import (
    ActiveContextFrame,
    ObjectCandidate,
    Percept,
    Delta,
    DeltaType,
    EventCandidate,
    EventType,
    GraphNode,
    NodeType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_store():
    """Create a mock graph store."""
    store = MagicMock()
    store.get_nodes_by_type.return_value = []
    store.get_node.return_value = None
    return store


@pytest.fixture
def mock_dialog_manager():
    """Create a mock dialog manager."""
    manager = MagicMock()
    manager.ask_label.return_value = "test_event"
    manager.notify.return_value = None
    return manager


@pytest.fixture
def delta_detector():
    """Create a fresh delta detector."""
    return DeltaDetector(
        move_threshold=0.5,
        missing_window=2,  # Entity must be absent for 2 steps
    )


@pytest.fixture
def event_resolver(mock_graph_store, mock_dialog_manager):
    """Create an event resolver with mocked dependencies."""
    return EventResolverStateChange(
        graph_store=mock_graph_store,
        dialog_manager=mock_dialog_manager,
        auto_label_events=True,
        prompt_for_unknown_events=False,
    )


@pytest.fixture
def sample_acf():
    """Create a sample active context frame."""
    return ActiveContextFrame(
        acf_id="test_acf_001",
        step_count=0,
        location_label="test_room",
        location_confidence=0.9,
    )


def make_percept(
    candidates: list[dict],
    room_guid: str = "room_001",
) -> Percept:
    """Create a test percept with candidates."""
    objects = []
    for c in candidates:
        obj = ObjectCandidate(
            candidate_id=c.get("guid", f"obj_{len(objects)}"),
            label=c.get("label", "unknown"),
            position=c.get("position"),
            extras={
                "guid": c.get("guid"),
                "category": c.get("category", "object"),
                "state": c.get("state"),
                "interactable": c.get("interactable", False),
                "visible": c.get("visible", True),
            },
        )
        objects.append(obj)
    
    return Percept(
        percept_id="percept_001",
        source_frame_id=1,
        candidates=objects,
        extras={
            "room_guid": room_guid,
            "room_label": "Test Room",
        },
    )


# =============================================================================
# Delta Detector Tests
# =============================================================================


class TestDeltaDetector:
    """Tests for the DeltaDetector class."""
    
    def test_detects_new_entity(self, delta_detector, sample_acf):
        """Test detection of a new entity appearing."""
        # First percept: empty
        percept1 = make_percept([])
        deltas1 = delta_detector.detect(percept1, sample_acf)
        assert len(deltas1) == 0
        
        # Second percept: entity appears
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "category": "ball",
            "position": (1.0, 0.0, 2.0),
        }])
        
        deltas2 = delta_detector.detect(percept2, sample_acf)
        
        assert len(deltas2) == 1
        assert deltas2[0].delta_type == DeltaType.NEW_ENTITY
        assert deltas2[0].entity_label == "Ball"
        # Note: entity_category was removed per emergent-knowledge architecture
    
    def test_detects_missing_entity(self, sample_acf):
        """Test detection of a missing entity (after missing_window steps)."""
        # Use detector with window of 2 (triggers on 2nd missing step)
        detector = DeltaDetector(move_threshold=0.5, missing_window=2)
        
        # First percept: entity present
        percept1 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "category": "ball",
        }])
        detector.detect(percept1, sample_acf)
        
        # Second percept: entity absent (step 1 of missing)
        sample_acf.step_count = 1
        percept2 = make_percept([])
        deltas2 = detector.detect(percept2, sample_acf)
        # First absence - counter goes to 1, not yet at window of 2
        
        # Third percept: entity still absent (step 2 - counter reaches window)
        sample_acf.step_count = 2
        percept3 = make_percept([])
        deltas3 = detector.detect(percept3, sample_acf)
        
        # Should trigger on 2nd absence (counter reaches window of 2)
        assert len(deltas3) == 1, f"Expected missing delta, got: {deltas3}"
        assert deltas3[0].delta_type == DeltaType.MISSING_ENTITY
        assert deltas3[0].entity_label == "Ball"
    
    def test_detects_moved_entity(self, delta_detector, sample_acf):
        """Test detection of entity movement."""
        # First percept: entity at position A
        percept1 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "category": "ball",
            "position": (0.0, 0.0, 0.0),
        }])
        delta_detector.detect(percept1, sample_acf)
        
        # Second percept: entity at position B (moved > threshold)
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "category": "ball",
            "position": (2.0, 0.0, 0.0),  # Moved 2 units > 0.5 threshold
        }])
        
        deltas = delta_detector.detect(percept2, sample_acf)
        
        assert len(deltas) == 1
        assert deltas[0].delta_type == DeltaType.MOVED_ENTITY
        assert deltas[0].position_delta == pytest.approx(2.0, rel=0.01)
    
    def test_detects_state_change(self, delta_detector, sample_acf):
        """Test detection of state change (open/closed)."""
        # First percept: drawer closed
        percept1 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "closed",
            "interactable": True,
        }])
        delta_detector.detect(percept1, sample_acf)
        
        # Second percept: drawer opened
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "open",
            "interactable": True,
        }])
        
        deltas = delta_detector.detect(percept2, sample_acf)
        
        assert len(deltas) == 1
        assert deltas[0].delta_type == DeltaType.STATE_CHANGED
        assert deltas[0].pre_state == "closed"
        assert deltas[0].post_state == "open"
    
    def test_no_delta_for_small_movement(self, delta_detector, sample_acf):
        """Test that small movements don't trigger delta."""
        # First percept
        percept1 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "position": (0.0, 0.0, 0.0),
        }])
        delta_detector.detect(percept1, sample_acf)
        
        # Second percept: tiny movement
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "position": (0.1, 0.0, 0.0),  # Moved 0.1 < 0.5 threshold
        }])
        
        deltas = delta_detector.detect(percept2, sample_acf)
        
        # Should be empty - no significant change
        assert len(deltas) == 0
    
    def test_reappearing_entity_clears_missing(self, delta_detector, sample_acf):
        """Test that entity reappearing clears pending missing."""
        # Entity present
        percept1 = make_percept([{"guid": "ball_001", "label": "Ball"}])
        delta_detector.detect(percept1, sample_acf)
        
        # Entity absent (step 1)
        sample_acf.step_count = 1
        percept2 = make_percept([])
        delta_detector.detect(percept2, sample_acf)
        
        # Entity reappears before missing_window
        sample_acf.step_count = 2
        percept3 = make_percept([{"guid": "ball_001", "label": "Ball"}])
        deltas = delta_detector.detect(percept3, sample_acf)
        
        # Should see it as new (reappeared)
        assert len(deltas) == 1
        assert deltas[0].delta_type == DeltaType.NEW_ENTITY


# =============================================================================
# Event Resolver Tests
# =============================================================================


class TestEventResolverStateChange:
    """Tests for the EventResolverStateChange class."""
    
    def test_creates_opened_event(self, event_resolver, sample_acf):
        """Test that opening a drawer creates a state-change event.
        
        Note: Per emergent-knowledge architecture, event types like 'opened'
        are now LEARNED from user interaction, not predefined. The system
        starts with structural types ('unknown') that get semantic labels from users.
        """
        # First percept: drawer closed
        percept1 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "closed",
            "interactable": True,
        }])
        events1 = event_resolver.resolve(percept1, sample_acf)
        
        # Second percept: drawer opened
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "open",
            "interactable": True,
        }])
        events2 = event_resolver.resolve(percept2, sample_acf)
        
        # Should have a state-change event (type is learned, not predefined)
        assert len(events2) == 1
        event = events2[0]
        # Event type is now structural until user labels it
        assert event["event_type"] in ["unknown", "state_change"]
        # Label should reflect the state change
        assert "open" in event["label"].lower() or "closed" in event["label"].lower() or "unknown" in event["label"].lower()
    
    def test_creates_closed_event(self, event_resolver, sample_acf):
        """Test that closing creates a CLOSED event."""
        # First percept: drawer open
        percept1 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "open",
            "interactable": True,
        }])
        event_resolver.resolve(percept1, sample_acf)
        
        # Second percept: drawer closed
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "closed",
            "interactable": True,
        }])
        events = event_resolver.resolve(percept2, sample_acf)
        
        assert len(events) == 1
        # Event type is learned, not predefined
        assert events[0]["event_type"] in ["unknown", "state_change"]
    
    def test_creates_turned_on_event(self, event_resolver, sample_acf):
        """Test that turning on a light creates a state-change event.
        
        Note: Per emergent-knowledge architecture, 'turned_on' is a LEARNED
        label, not predefined. System uses structural types initially.
        """
        # First percept: light off
        percept1 = make_percept([{
            "guid": "light_001",
            "label": "Light",
            "category": "light",
            "state": "off",
            "interactable": True,
        }])
        event_resolver.resolve(percept1, sample_acf)
        
        # Second percept: light on
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "light_001",
            "label": "Light",
            "category": "light",
            "state": "on",
            "interactable": True,
        }])
        events = event_resolver.resolve(percept2, sample_acf)
        
        assert len(events) == 1
        # Event type is learned, not predefined
        assert events[0]["event_type"] in ["unknown", "state_change"]
    
    def test_creates_appeared_event(self, event_resolver, sample_acf):
        """Test that entity appearing creates an appearance event."""
        # First percept: empty
        percept1 = make_percept([])
        event_resolver.resolve(percept1, sample_acf)
        
        # Second percept: ball appears
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "category": "ball",
        }])
        events = event_resolver.resolve(percept2, sample_acf)
        
        assert len(events) == 1
        # Event type is now structural: 'appeared' not predefined enum
        assert events[0]["event_type"] == "appeared"
    
    def test_creates_moved_event(self, event_resolver, sample_acf):
        """Test that entity movement creates a movement event."""
        # First percept: ball at position A
        percept1 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "category": "ball",
            "position": (0.0, 0.0, 0.0),
        }])
        event_resolver.resolve(percept1, sample_acf)
        
        # Second percept: ball moved
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "ball_001",
            "label": "Ball",
            "category": "ball",
            "position": (5.0, 0.0, 0.0),
        }])
        events = event_resolver.resolve(percept2, sample_acf)
        
        assert len(events) == 1
        # Event type is now structural: 'moved' not predefined enum
        assert events[0]["event_type"] == "moved"
    
    def test_stores_deltas_in_acf(self, event_resolver, sample_acf):
        """Test that deltas are stored in ACF extras."""
        # Create state change
        percept1 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "closed",
            "interactable": True,
        }])
        event_resolver.resolve(percept1, sample_acf)
        
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "open",
            "interactable": True,
        }])
        event_resolver.resolve(percept2, sample_acf)
        
        # Check deltas stored in ACF
        assert "deltas" in sample_acf.extras
        assert len(sample_acf.extras["deltas"]) >= 1
    
    def test_recognizes_learned_event(self, mock_graph_store, mock_dialog_manager, sample_acf):
        """Test that previously learned events are recognized.
        
        Note: Per emergent-knowledge architecture, events are learned from
        user interaction and stored with pattern signatures. When a matching
        pattern is found, the learned label is used.
        """
        # Create a "known" event node with the new pattern signature format
        # IMPORTANT: Pattern signature is now state-only (no category)
        known_event = GraphNode(
            node_id="event_001",
            node_type=NodeType.EVENT,
            label="drawer_opened_known",
            labels=["drawer opened"],
            extras={
                # New format: state:pre->state:post (no category prefix)
                "pattern_signature": "state:closed->state:open",
                "event_type": "opened",
            },
        )
        mock_graph_store.get_nodes_by_type.return_value = [known_event]
        
        resolver = EventResolverStateChange(
            graph_store=mock_graph_store,
            dialog_manager=mock_dialog_manager,
            auto_label_events=True,
        )
        
        # Drawer closed
        percept1 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "closed",
            "interactable": True,
        }])
        resolver.resolve(percept1, sample_acf)
        
        # Drawer opened
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "category": "drawer",
            "state": "open",
            "interactable": True,
        }])
        events = resolver.resolve(percept2, sample_acf)
        
        # Should use the learned label
        assert len(events) == 1
        assert events[0]["label"] == "drawer_opened_known"
        assert events[0]["is_learned"] is True
    
    def test_event_includes_involved_entities(self, event_resolver, sample_acf):
        """Test that events include involved entity information."""
        # State change
        percept1 = make_percept([{
            "guid": "drawer_123",
            "label": "Kitchen Drawer",
            "category": "drawer",
            "state": "closed",
            "interactable": True,
        }])
        event_resolver.resolve(percept1, sample_acf)
        
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_123",
            "label": "Kitchen Drawer",
            "category": "drawer",
            "state": "open",
            "interactable": True,
        }])
        events = event_resolver.resolve(percept2, sample_acf)
        
        event = events[0]
        assert "drawer_123" in event["involved_entity_ids"]
        assert "Kitchen Drawer" in event["involved_entity_labels"]
    
    def test_metrics_tracking(self, event_resolver, sample_acf):
        """Test that event resolver tracks metrics."""
        # Generate some events
        percept1 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "state": "closed",
            "interactable": True,
        }])
        event_resolver.resolve(percept1, sample_acf)
        
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "state": "open",
            "interactable": True,
        }])
        event_resolver.resolve(percept2, sample_acf)
        
        # Check metrics
        assert event_resolver.events_detected >= 1
        assert event_resolver.events_labeled >= 1
        assert event_resolver.deltas_detected >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase5Integration:
    """Integration tests for Phase 5 features."""
    
    def test_full_state_change_cycle(self, mock_graph_store, mock_dialog_manager):
        """Test complete cycle: detect delta -> create event -> learn -> recognize."""
        resolver = EventResolverStateChange(
            graph_store=mock_graph_store,
            dialog_manager=mock_dialog_manager,
            auto_label_events=True,
            prompt_for_unknown_events=False,
        )
        
        acf = ActiveContextFrame(
            acf_id="test_acf",
            step_count=0,
            location_label="kitchen",
        )
        
        # Step 1: Initial state
        percept1 = make_percept([{
            "guid": "light_001",
            "label": "Kitchen Light",
            "category": "light",
            "state": "off",
            "interactable": True,
        }])
        events1 = resolver.resolve(percept1, acf)
        
        # Step 2: Turn on
        acf.step_count = 1
        percept2 = make_percept([{
            "guid": "light_001",
            "label": "Kitchen Light",
            "category": "light",
            "state": "on",
            "interactable": True,
        }])
        events2 = resolver.resolve(percept2, acf)
        
        assert len(events2) == 1
        turned_on_event = events2[0]
        # Event types are now structural strings, not predefined enums
        assert turned_on_event["event_type"] in ["unknown", "state_change"]
        
        # Step 3: Turn off
        acf.step_count = 2
        percept3 = make_percept([{
            "guid": "light_001",
            "label": "Kitchen Light",
            "category": "light",
            "state": "off",
            "interactable": True,
        }])
        events3 = resolver.resolve(percept3, acf)
        
        assert len(events3) == 1
        # Event types are now structural strings, not predefined enums
        assert events3[0]["event_type"] in ["unknown", "state_change"]
        
        # Verify metrics - note: first appearance also generates an event
        assert resolver.events_detected >= 2  # At least the two state changes
        assert resolver.deltas_detected >= 2
    
    def test_acf_delta_and_event_accessors(self, event_resolver, sample_acf):
        """Test ACF helper methods for accessing deltas and events."""
        # Generate events
        percept1 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "state": "closed",
            "interactable": True,
        }])
        event_resolver.resolve(percept1, sample_acf)
        
        sample_acf.step_count = 1
        percept2 = make_percept([{
            "guid": "drawer_001",
            "label": "Drawer",
            "state": "open",
            "interactable": True,
        }])
        
        # Get events using ACF method
        events = event_resolver.resolve(percept2, sample_acf)
        sample_acf.events.extend(events)
        
        # Test accessors
        recent_events = sample_acf.get_recent_events(3)
        recent_deltas = sample_acf.get_recent_deltas(3)
        
        assert len(recent_events) >= 1
        assert len(recent_deltas) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
