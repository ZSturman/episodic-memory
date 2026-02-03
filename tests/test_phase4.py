"""Phase 4 tests for Unity integration."""

from __future__ import annotations

import hashlib
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from episodic_agent.schemas import (
    ActiveContextFrame,
    GraphNode,
    NodeType,
    ObjectCandidate,
    Percept,
    SensorFrame,
)


# =============================================================================
# Tests for GUID-to-embedding determinism
# =============================================================================


class TestGuidToEmbedding:
    """Test deterministic GUID to embedding conversion."""

    def test_same_guid_same_embedding(self):
        """Same GUID should always produce the same embedding."""
        from episodic_agent.modules.unity.perception import guid_to_embedding
        
        guid = "test-guid-12345"
        
        emb1 = guid_to_embedding(guid, dim=128)
        emb2 = guid_to_embedding(guid, dim=128)
        
        assert emb1 == emb2
        assert len(emb1) == 128

    def test_different_guids_different_embeddings(self):
        """Different GUIDs should produce different embeddings."""
        from episodic_agent.modules.unity.perception import guid_to_embedding
        
        guid1 = "room-living-001"
        guid2 = "room-kitchen-002"
        
        emb1 = guid_to_embedding(guid1)
        emb2 = guid_to_embedding(guid2)
        
        assert emb1 != emb2

    def test_embedding_normalized(self):
        """Embeddings should be unit normalized."""
        from episodic_agent.modules.unity.perception import guid_to_embedding
        import math
        
        guid = "entity-door-front"
        embedding = guid_to_embedding(guid, dim=64)
        
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 1e-6

    def test_embedding_dimension(self):
        """Embedding dimension should be configurable."""
        from episodic_agent.modules.unity.perception import guid_to_embedding
        
        for dim in [32, 64, 128, 256]:
            embedding = guid_to_embedding("test-guid", dim=dim)
            assert len(embedding) == dim


# =============================================================================
# Tests for PerceptionUnityCheat
# =============================================================================


class TestPerceptionUnityCheat:
    """Test cheat perception module."""

    def test_process_frame_with_room(self):
        """Should extract room GUID and create scene embedding."""
        from episodic_agent.modules.unity.perception import PerceptionUnityCheat
        
        perception = PerceptionUnityCheat()
        
        frame = SensorFrame(
            frame_id=1,
            timestamp=datetime.now(),
            raw_data={},
            sensor_type="unity_websocket",
            extras={
                "current_room": "room-living-001",
                "current_room_label": "Living Room",
                "entities": [],
                "camera_pose": {"position": {"x": 0, "y": 1, "z": 0}},
            },
        )
        
        percept = perception.process(frame)
        
        assert percept.source_frame_id == 1
        assert percept.scene_embedding is not None
        assert len(percept.scene_embedding) == 128  # Default dim
        assert percept.extras.get("room_guid") == "room-living-001"
        assert percept.extras.get("room_label") == "Living Room"

    def test_process_frame_with_entities(self):
        """Should convert entities to object candidates."""
        from episodic_agent.modules.unity.perception import PerceptionUnityCheat
        
        perception = PerceptionUnityCheat()
        
        frame = SensorFrame(
            frame_id=1,
            timestamp=datetime.now(),
            raw_data={},
            sensor_type="unity_websocket",
            extras={
                "current_room": "room-living-001",
                "entities": [
                    {
                        "guid": "door-front-001",
                        "label": "Front Door",
                        "category": "door",
                        "position": {"x": 0, "y": 1, "z": 5},
                        "state": "Closed",
                        "visible": True,
                        "distance": 4.2,
                    },
                    {
                        "guid": "lamp-table-001",
                        "label": "Table Lamp",
                        "category": "item",
                        "position": {"x": 3, "y": 0.8, "z": -2},
                        "state": "On",
                        "visible": True,
                        "distance": 1.5,
                    },
                ],
            },
        )
        
        percept = perception.process(frame)
        
        assert len(percept.candidates) == 2
        
        door = next(c for c in percept.candidates if c.candidate_id == "door-front-001")
        assert door.label == "Front Door"
        assert door.extras.get("category") == "door"
        assert door.extras.get("state") == "Closed"
        assert door.embedding is not None

    def test_invisible_entities_filtered_by_default(self):
        """Invisible entities should be filtered out by default."""
        from episodic_agent.modules.unity.perception import PerceptionUnityCheat
        
        perception = PerceptionUnityCheat(include_invisible=False)
        
        frame = SensorFrame(
            frame_id=1,
            timestamp=datetime.now(),
            raw_data={},
            extras={
                "current_room": "room-living-001",
                "entities": [
                    {"guid": "visible-001", "label": "Visible", "category": "item", "visible": True, "position": {}},
                    {"guid": "invisible-001", "label": "Invisible", "category": "item", "visible": False, "position": {}},
                ],
            },
        )
        
        percept = perception.process(frame)
        
        assert len(percept.candidates) == 1
        assert percept.candidates[0].candidate_id == "visible-001"

    def test_include_invisible_entities(self):
        """Should include invisible entities when configured."""
        from episodic_agent.modules.unity.perception import PerceptionUnityCheat
        
        perception = PerceptionUnityCheat(include_invisible=True)
        
        frame = SensorFrame(
            frame_id=1,
            timestamp=datetime.now(),
            raw_data={},
            extras={
                "current_room": "room-living-001",
                "entities": [
                    {"guid": "visible-001", "label": "Visible", "category": "item", "visible": True, "position": {}},
                    {"guid": "invisible-001", "label": "Invisible", "category": "item", "visible": False, "position": {}},
                ],
            },
        )
        
        percept = perception.process(frame)
        
        assert len(percept.candidates) == 2


# =============================================================================
# Tests for LocationResolverCheat
# =============================================================================


class TestLocationResolverCheat:
    """Test cheat location resolver."""

    def test_unknown_location_created(self):
        """Unknown location should create a new node."""
        from episodic_agent.modules.unity.resolvers import LocationResolverCheat
        from episodic_agent.memory.graph_store import LabeledGraphStore
        from episodic_agent.modules.dialog import AutoAcceptDialogManager
        
        graph_store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager(default_label="test_room")
        resolver = LocationResolverCheat(graph_store, dialog, auto_label=True)
        
        percept = Percept(
            percept_id="test",
            source_frame_id=1,
            extras={
                "room_guid": "room-living-001",
                "room_label": "Living Room",
            },
        )
        acf = ActiveContextFrame(acf_id="test_acf")
        
        label, confidence = resolver.resolve(percept, acf)
        
        assert label == "Living Room"  # From room_label hint
        assert confidence > 0.5
        
        # Verify node was created
        locations = graph_store.get_nodes_by_type(NodeType.LOCATION)
        assert len(locations) == 1
        assert locations[0].source_id == "room-living-001"

    def test_known_location_resolved(self):
        """Known location should be resolved with high confidence."""
        from episodic_agent.modules.unity.resolvers import LocationResolverCheat
        from episodic_agent.memory.graph_store import LabeledGraphStore
        from episodic_agent.modules.dialog import AutoAcceptDialogManager
        
        graph_store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        resolver = LocationResolverCheat(graph_store, dialog, auto_label=True)
        
        # Create location first time
        percept = Percept(
            percept_id="test1",
            source_frame_id=1,
            extras={"room_guid": "room-kitchen-001", "room_label": "Kitchen"},
        )
        acf = ActiveContextFrame(acf_id="test_acf")
        
        resolver.resolve(percept, acf)
        
        # Resolve same location again
        percept2 = Percept(
            percept_id="test2",
            source_frame_id=2,
            extras={"room_guid": "room-kitchen-001", "room_label": "Kitchen"},
        )
        
        label, confidence = resolver.resolve(percept2, acf)
        
        assert label == "Kitchen"
        assert confidence > 0.8  # High confidence for known location

    def test_no_room_returns_unknown(self):
        """No room GUID should return unknown location."""
        from episodic_agent.modules.unity.resolvers import LocationResolverCheat
        from episodic_agent.memory.graph_store import LabeledGraphStore
        from episodic_agent.modules.dialog import AutoAcceptDialogManager
        
        graph_store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        resolver = LocationResolverCheat(graph_store, dialog)
        
        percept = Percept(
            percept_id="test",
            source_frame_id=1,
            extras={"room_guid": None},
        )
        acf = ActiveContextFrame(acf_id="test_acf")
        
        label, confidence = resolver.resolve(percept, acf)
        
        assert label == "unknown"
        assert confidence == 0.0


# =============================================================================
# Tests for EntityResolverCheat
# =============================================================================


class TestEntityResolverCheat:
    """Test cheat entity resolver."""

    def test_new_entity_created(self):
        """New entity should create a node in graph."""
        from episodic_agent.modules.unity.resolvers import (
            EntityResolverCheat,
            LocationResolverCheat,
        )
        from episodic_agent.memory.graph_store import LabeledGraphStore
        from episodic_agent.modules.dialog import AutoAcceptDialogManager
        
        graph_store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        loc_resolver = LocationResolverCheat(graph_store, dialog, auto_label=True)
        entity_resolver = EntityResolverCheat(
            graph_store, dialog, loc_resolver, auto_label=True
        )
        
        candidate = ObjectCandidate(
            candidate_id="door-front-001",
            label="Front Door",
            confidence=0.9,
            extras={
                "guid": "door-front-001",
                "category": "door",
                "visible": True,
            },
        )
        
        percept = Percept(
            percept_id="test",
            source_frame_id=1,
            candidates=[candidate],
            extras={"room_guid": "room-living-001"},
        )
        acf = ActiveContextFrame(acf_id="test_acf")
        
        resolved = entity_resolver.resolve(percept, acf)
        
        assert len(resolved) == 1
        assert resolved[0].label == "Front Door"
        
        # Verify node created
        entities = graph_store.get_nodes_by_type(NodeType.ENTITY)
        assert len(entities) == 1
        assert entities[0].source_id == "door-front-001"

    def test_known_entity_resolved(self):
        """Known entity should be resolved from graph."""
        from episodic_agent.modules.unity.resolvers import (
            EntityResolverCheat,
            LocationResolverCheat,
        )
        from episodic_agent.memory.graph_store import LabeledGraphStore
        from episodic_agent.modules.dialog import AutoAcceptDialogManager
        
        graph_store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        loc_resolver = LocationResolverCheat(graph_store, dialog, auto_label=True)
        entity_resolver = EntityResolverCheat(
            graph_store, dialog, loc_resolver, auto_label=True
        )
        
        acf = ActiveContextFrame(acf_id="test_acf")
        
        # First sighting
        candidate = ObjectCandidate(
            candidate_id="lamp-001",
            label="Table Lamp",
            extras={"guid": "lamp-001", "category": "item", "visible": True},
        )
        percept = Percept(
            percept_id="test1",
            source_frame_id=1,
            candidates=[candidate],
            extras={"room_guid": "room-living-001"},
        )
        entity_resolver.resolve(percept, acf)
        
        # Second sighting
        candidate2 = ObjectCandidate(
            candidate_id="lamp-001",
            label="Table Lamp",
            extras={"guid": "lamp-001", "category": "item", "visible": True},
        )
        percept2 = Percept(
            percept_id="test2",
            source_frame_id=2,
            candidates=[candidate2],
            extras={"room_guid": "room-living-001"},
        )
        resolved = entity_resolver.resolve(percept2, acf)
        
        # Should still be one entity in graph
        entities = graph_store.get_nodes_by_type(NodeType.ENTITY)
        assert len(entities) == 1
        assert entities[0].access_count == 2  # Accessed twice

    def test_typical_in_edge_created(self):
        """Entity should create typical_in edge to location."""
        from episodic_agent.modules.unity.resolvers import (
            EntityResolverCheat,
            LocationResolverCheat,
        )
        from episodic_agent.memory.graph_store import LabeledGraphStore
        from episodic_agent.modules.dialog import AutoAcceptDialogManager
        from episodic_agent.schemas import EdgeType
        
        graph_store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        loc_resolver = LocationResolverCheat(graph_store, dialog, auto_label=True)
        entity_resolver = EntityResolverCheat(
            graph_store, dialog, loc_resolver, auto_label=True
        )
        
        acf = ActiveContextFrame(acf_id="test_acf")
        
        # Create location first
        loc_percept = Percept(
            percept_id="loc",
            source_frame_id=0,
            extras={"room_guid": "room-living-001", "room_label": "Living Room"},
        )
        loc_resolver.resolve(loc_percept, acf)
        
        # Create entity
        candidate = ObjectCandidate(
            candidate_id="sofa-001",
            label="Sofa",
            extras={"guid": "sofa-001", "category": "furniture", "visible": True},
        )
        percept = Percept(
            percept_id="test",
            source_frame_id=1,
            candidates=[candidate],
            extras={"room_guid": "room-living-001"},
        )
        entity_resolver.resolve(percept, acf)
        
        # Check edge exists
        edges = graph_store.get_all_edges()
        typical_in_edges = [e for e in edges if e.edge_type == EdgeType.TYPICAL_IN]
        
        assert len(typical_in_edges) == 1
        assert typical_in_edges[0].weight == 1.0


# =============================================================================
# Tests for Profile System
# =============================================================================


class TestProfiles:
    """Test profile configuration system."""

    def test_get_stub_profile(self):
        """Should return stub profile."""
        from episodic_agent.utils.profiles import get_profile
        
        profile = get_profile("stub")
        
        assert profile.name == "stub"
        assert profile.sensor_provider == "StubSensorProvider"

    def test_get_unity_cheat_profile(self):
        """Should return unity_cheat profile."""
        from episodic_agent.utils.profiles import get_profile
        
        profile = get_profile("unity_cheat")
        
        assert profile.name == "unity_cheat"
        assert profile.sensor_provider == "UnityWebSocketSensorProvider"
        assert profile.perception == "PerceptionUnityCheat"
        assert profile.location_resolver == "LocationResolverCheat"
        assert profile.entity_resolver == "EntityResolverCheat"

    def test_invalid_profile_raises(self):
        """Invalid profile name should raise ValueError."""
        from episodic_agent.utils.profiles import get_profile
        
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("invalid_profile")

    def test_list_profiles(self):
        """Should list available profiles."""
        from episodic_agent.utils.profiles import list_profiles
        
        profiles = list_profiles()
        
        assert len(profiles) >= 2
        names = [p[0] for p in profiles]
        assert "stub" in names
        assert "unity_cheat" in names


class TestModuleFactory:
    """Test module factory creation."""

    def test_create_stub_modules(self, tmp_path):
        """Should create all stub modules."""
        from episodic_agent.utils.profiles import ModuleFactory, get_profile
        
        profile = get_profile("stub")
        factory = ModuleFactory(profile, run_dir=tmp_path, seed=42)
        
        modules = factory.create_modules()
        
        assert "sensor" in modules
        assert "perception" in modules
        assert "acf_builder" in modules
        assert "location_resolver" in modules
        assert "entity_resolver" in modules
        assert "episode_store" in modules
        assert "graph_store" in modules


# =============================================================================
# Tests for SensorFrame Conversion
# =============================================================================


class TestSensorFrameConversion:
    """Test Unity JSON to SensorFrame conversion."""

    def test_convert_unity_frame(self):
        """Should convert Unity JSON to SensorFrame."""
        from episodic_agent.modules.unity.sensor_provider import (
            UnityWebSocketSensorProvider,
        )
        
        provider = UnityWebSocketSensorProvider.__new__(UnityWebSocketSensorProvider)
        provider._embedding_dim = 128
        
        unity_data = {
            "protocol_version": "1.0.0",
            "frame_id": 42,
            "timestamp": 1704067200.123,
            "camera_pose": {
                "position": {"x": 2.5, "y": 1.6, "z": -3.2},
                "rotation": {"x": 15.0, "y": 45.0, "z": 0.0},
                "forward": {"x": 0.65, "y": -0.26, "z": 0.71},
            },
            "current_room": "room-living-001",
            "current_room_label": "Living Room",
            "entities": [
                {
                    "guid": "door-front-001",
                    "label": "Front Door",
                    "category": "door",
                    "position": {"x": 0.0, "y": 1.0, "z": 5.0},
                    "state": "Closed",
                    "visible": True,
                    "distance": 4.2,
                },
            ],
            "state_changes": [],
        }
        
        frame = provider._convert_to_sensor_frame(unity_data)
        
        assert frame.frame_id == 42
        assert frame.sensor_type == "unity_websocket"
        assert frame.extras.get("current_room") == "room-living-001"
        assert frame.extras.get("current_room_label") == "Living Room"
        assert len(frame.extras.get("entities", [])) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase4Integration:
    """Integration tests for Phase 4 components."""

    def test_full_perception_pipeline(self):
        """Test perception → location → entity pipeline."""
        from episodic_agent.modules.unity.perception import PerceptionUnityCheat
        from episodic_agent.modules.unity.resolvers import (
            EntityResolverCheat,
            LocationResolverCheat,
        )
        from episodic_agent.memory.graph_store import LabeledGraphStore
        from episodic_agent.modules.dialog import AutoAcceptDialogManager
        
        # Setup
        graph_store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        perception = PerceptionUnityCheat()
        loc_resolver = LocationResolverCheat(graph_store, dialog, auto_label=True)
        entity_resolver = EntityResolverCheat(
            graph_store, dialog, loc_resolver, auto_label=True
        )
        
        # Create sensor frame
        frame = SensorFrame(
            frame_id=1,
            timestamp=datetime.now(),
            raw_data={},
            sensor_type="unity_websocket",
            extras={
                "current_room": "room-kitchen-001",
                "current_room_label": "Kitchen",
                "entities": [
                    {
                        "guid": "fridge-001",
                        "label": "Refrigerator",
                        "category": "appliance",
                        "position": {"x": 1, "y": 1, "z": 2},
                        "state": "Closed",
                        "visible": True,
                        "distance": 2.0,
                    },
                ],
                "camera_pose": {"position": {"x": 0, "y": 1, "z": 0}},
            },
        )
        
        # Process through pipeline
        percept = perception.process(frame)
        acf = ActiveContextFrame(acf_id="test_acf")
        
        location, loc_conf = loc_resolver.resolve(percept, acf)
        entities = entity_resolver.resolve(percept, acf)
        
        # Verify results
        assert location == "Kitchen"
        assert loc_conf > 0.5
        assert len(entities) == 1
        assert entities[0].label == "Refrigerator"
        
        # Verify graph state
        loc_nodes = graph_store.get_nodes_by_type(NodeType.LOCATION)
        ent_nodes = graph_store.get_nodes_by_type(NodeType.ENTITY)
        
        assert len(loc_nodes) == 1
        assert len(ent_nodes) == 1
        
        # Verify edge
        edges = graph_store.get_all_edges()
        assert len(edges) == 1
        assert edges[0].source_node_id == ent_nodes[0].node_id
        assert edges[0].target_node_id == loc_nodes[0].node_id
