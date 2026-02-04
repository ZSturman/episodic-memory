"""Comprehensive tests for schemas and data contracts."""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from episodic_agent.schemas import (
    ActiveContextFrame,
    Episode,
    GraphEdge,
    GraphNode,
    ObjectCandidate,
    Percept,
    RetrievalResult,
    SensorFrame,
    StepResult,
)
from episodic_agent.schemas.graph import EdgeType, NodeType
from episodic_agent.schemas.events import Delta, DeltaType, EventCandidate, EventType


# =============================================================================
# SensorFrame Tests
# =============================================================================

class TestSensorFrame:
    """Tests for SensorFrame schema."""

    def test_minimal_creation(self):
        """Create frame with minimal required fields."""
        frame = SensorFrame(
            frame_id=1,
            sensor_type="test",
        )
        assert frame.frame_id == 1
        assert frame.sensor_type == "test"
        assert frame.raw_data == {}
        assert frame.extras == {}

    def test_full_creation(self):
        """Create frame with all fields."""
        now = datetime.now()
        frame = SensorFrame(
            frame_id=42,
            timestamp=now,
            raw_data={"image": "base64data", "depth": [1.0, 2.0]},
            sensor_type="unity_websocket",
            extras={"current_room": "room-001", "entities": []},
        )
        assert frame.frame_id == 42
        assert frame.timestamp == now
        assert "image" in frame.raw_data
        assert frame.extras["current_room"] == "room-001"

    def test_serialization(self):
        """Frame serializes to JSON and back."""
        frame = SensorFrame(
            frame_id=1,
            sensor_type="test",
            raw_data={"key": "value"},
        )
        
        # To JSON
        json_str = frame.model_dump_json()
        data = json.loads(json_str)
        assert data["frame_id"] == 1
        
        # From JSON
        restored = SensorFrame.model_validate_json(json_str)
        assert restored.frame_id == frame.frame_id

    def test_extras_forward_compatibility(self):
        """Extras field allows arbitrary data."""
        frame = SensorFrame(
            frame_id=1,
            sensor_type="test",
            extras={
                "new_field_v2": "some_value",
                "nested": {"a": 1, "b": 2},
                "list_data": [1, 2, 3],
            },
        )
        assert frame.extras["new_field_v2"] == "some_value"
        assert frame.extras["nested"]["a"] == 1


# =============================================================================
# ObjectCandidate Tests
# =============================================================================

class TestObjectCandidate:
    """Tests for ObjectCandidate schema."""

    def test_minimal_creation(self):
        """Create candidate with minimal fields."""
        obj = ObjectCandidate(candidate_id="obj_001")
        assert obj.candidate_id == "obj_001"
        assert obj.label == "unknown"
        assert obj.confidence == 0.0

    def test_full_creation(self):
        """Create candidate with all fields."""
        obj = ObjectCandidate(
            candidate_id="door_001",
            label="Front Door",
            labels=["Door", "Entrance"],
            confidence=0.95,
            embedding=[0.1, 0.2, 0.3],
            position=(1.0, 2.0, 3.0),
            extras={"guid": "unity-guid-123", "category": "door", "state": "Closed"},
        )
        assert obj.label == "Front Door"
        assert obj.confidence == 0.95
        assert len(obj.embedding) == 3
        assert obj.position == (1.0, 2.0, 3.0)

    def test_confidence_bounds(self):
        """Confidence must be 0-1."""
        # Valid
        obj = ObjectCandidate(candidate_id="test", confidence=0.5)
        assert obj.confidence == 0.5
        
        # Invalid - should raise
        with pytest.raises(ValidationError):
            ObjectCandidate(candidate_id="test", confidence=1.5)
        
        with pytest.raises(ValidationError):
            ObjectCandidate(candidate_id="test", confidence=-0.1)


# =============================================================================
# Percept Tests
# =============================================================================

class TestPercept:
    """Tests for Percept schema."""

    def test_creation(self):
        """Create percept with objects."""
        obj1 = ObjectCandidate(candidate_id="obj_1", label="Table")
        obj2 = ObjectCandidate(candidate_id="obj_2", label="Chair")
        
        percept = Percept(
            percept_id="percept_001",
            scene_embedding=[0.1, 0.2, 0.3, 0.4],
            objects=[obj1, obj2],
        )
        
        assert percept.percept_id == "percept_001"
        assert len(percept.scene_embedding) == 4
        assert len(percept.objects) == 2
        assert percept.objects[0].label == "Table"

    def test_empty_percept(self):
        """Percept can be empty (no objects)."""
        percept = Percept(
            percept_id="empty",
            scene_embedding=[],
        )
        assert len(percept.objects) == 0


# =============================================================================
# ActiveContextFrame Tests
# =============================================================================

class TestActiveContextFrame:
    """Tests for ActiveContextFrame schema."""

    def test_creation(self):
        """Create ACF with default values."""
        acf = ActiveContextFrame(acf_id="acf_001")
        
        assert acf.acf_id == "acf_001"
        assert acf.step_count == 0
        assert acf.location_label == "unknown"
        assert acf.location_confidence == 0.0
        assert acf.entities == []
        assert acf.events == []

    def test_touch_updates_timestamp(self):
        """Touch method updates timestamp."""
        acf = ActiveContextFrame(acf_id="acf_001")
        original = acf.updated_at
        
        import time
        time.sleep(0.01)  # Small delay
        acf.touch()
        
        assert acf.updated_at > original

    def test_mutable_update(self):
        """ACF can be modified in place."""
        acf = ActiveContextFrame(acf_id="acf_001")
        
        acf.step_count = 10
        acf.location_label = "Kitchen"
        acf.location_confidence = 0.95
        acf.events.append({"type": "state_change", "entity": "door"})
        
        assert acf.step_count == 10
        assert acf.location_label == "Kitchen"
        assert len(acf.events) == 1


# =============================================================================
# Episode Tests
# =============================================================================

class TestEpisode:
    """Tests for Episode schema."""

    def test_creation(self):
        """Create episode with required fields."""
        now = datetime.now()
        episode = Episode(
            episode_id="ep_001",
            source_acf_id="acf_001",
            location_label="Kitchen",
            start_time=now,
            end_time=now,
            step_count=50,
        )
        
        assert episode.episode_id == "ep_001"
        assert episode.location_label == "Kitchen"
        assert episode.step_count == 50

    def test_serialization_roundtrip(self):
        """Episode serializes and deserializes correctly."""
        episode = Episode(
            episode_id="ep_001",
            source_acf_id="acf_001",
            location_label="Kitchen",
            start_time=datetime.now(),
            end_time=datetime.now(),
            step_count=50,
            entities=["door_001", "lamp_001"],
            events=[{"type": "state_change"}],
            boundary_reason="location_change",
        )
        
        json_str = episode.model_dump_json()
        restored = Episode.model_validate_json(json_str)
        
        assert restored.episode_id == episode.episode_id
        assert restored.entities == episode.entities
        assert restored.boundary_reason == episode.boundary_reason


# =============================================================================
# GraphNode Tests
# =============================================================================

class TestGraphNode:
    """Tests for GraphNode schema."""

    def test_location_node(self):
        """Create location node."""
        node = GraphNode(
            node_id="loc_kitchen",
            node_type=NodeType.LOCATION,
            label="Kitchen",
            properties={"guid": "room-kitchen-001", "visit_count": 5},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert node.node_type == NodeType.LOCATION
        assert node.properties["visit_count"] == 5

    def test_entity_node(self):
        """Create entity node."""
        node = GraphNode(
            node_id="ent_door",
            node_type=NodeType.ENTITY,
            label="Front Door",
            properties={"category": "door", "typical_state": "Closed"},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert node.node_type == NodeType.ENTITY
        assert node.properties["category"] == "door"

    def test_node_types(self):
        """All node types are valid."""
        for node_type in NodeType:
            node = GraphNode(
                node_id=f"node_{node_type.value}",
                node_type=node_type,
                label="Test",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            assert node.node_type == node_type


# =============================================================================
# GraphEdge Tests
# =============================================================================

class TestGraphEdge:
    """Tests for GraphEdge schema."""

    def test_typical_in_edge(self):
        """Create typical_in edge."""
        edge = GraphEdge(
            edge_id="e_001",
            edge_type=EdgeType.TYPICAL_IN,
            source_id="ent_door",
            target_id="loc_kitchen",
            weight=5.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert edge.edge_type == EdgeType.TYPICAL_IN
        assert edge.weight == 5.0

    def test_edge_types(self):
        """All edge types are valid."""
        for edge_type in EdgeType:
            edge = GraphEdge(
                edge_id=f"edge_{edge_type.value}",
                edge_type=edge_type,
                source_id="src",
                target_id="tgt",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            assert edge.edge_type == edge_type


# =============================================================================
# Delta Tests
# =============================================================================

class TestDelta:
    """Tests for Delta schema."""

    def test_state_change_delta(self):
        """Create state change delta."""
        delta = Delta(
            delta_type=DeltaType.STATE_CHANGED,
            entity_id="door_001",
            entity_label="Front Door",
            old_value="Closed",
            new_value="Open",
        )
        
        assert delta.delta_type == DeltaType.STATE_CHANGED
        assert delta.old_value == "Closed"
        assert delta.new_value == "Open"

    def test_delta_types(self):
        """All delta types work."""
        for delta_type in DeltaType:
            delta = Delta(
                delta_type=delta_type,
                entity_id="test_entity",
            )
            assert delta.delta_type == delta_type


# =============================================================================
# EventCandidate Tests
# =============================================================================

class TestEventCandidate:
    """Tests for EventCandidate schema."""

    def test_state_change_event(self):
        """Create state change event."""
        event = EventCandidate(
            event_type=EventType.STATE_CHANGE,
            entity_id="lamp_001",
            label="Light turned on",
            confidence=1.0,
            timestamp=datetime.now(),
            details={"old_state": "Off", "new_state": "On"},
        )
        
        assert event.event_type == EventType.STATE_CHANGE
        assert event.details["new_state"] == "On"


# =============================================================================
# RetrievalResult Tests
# =============================================================================

class TestRetrievalResult:
    """Tests for RetrievalResult schema."""

    def test_empty_result(self):
        """Create empty retrieval result."""
        result = RetrievalResult(query_id="q_001")
        
        assert result.query_id == "q_001"
        assert result.episodes == []
        assert result.nodes == []
        assert result.scores == {}

    def test_result_with_scores(self):
        """Result with scores."""
        result = RetrievalResult(
            query_id="q_001",
            scores={
                "ep_001": 0.8,
                "ep_002": 0.6,
                "node_001": 0.5,
            },
        )
        
        assert result.scores["ep_001"] == 0.8


# =============================================================================
# StepResult Tests
# =============================================================================

class TestStepResult:
    """Tests for StepResult schema."""

    def test_creation(self):
        """Create step result."""
        result = StepResult(
            step_number=42,
            timestamp=datetime.now(),
            location_label="Kitchen",
            location_confidence=0.95,
            entity_count=5,
            event_count=2,
            episode_count=3,
            boundary_triggered=True,
            boundary_reason="location_change",
        )
        
        assert result.step_number == 42
        assert result.boundary_triggered is True

    def test_to_log_dict(self):
        """Step result converts to log dict."""
        result = StepResult(
            step_number=1,
            timestamp=datetime.now(),
            location_label="Test",
            location_confidence=0.5,
            entity_count=0,
            event_count=0,
            episode_count=0,
            boundary_triggered=False,
        )
        
        log_dict = result.to_log_dict()
        assert "step_number" in log_dict
        assert "timestamp" in log_dict
        assert log_dict["step_number"] == 1
