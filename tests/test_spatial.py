"""Tests for Phase 2: Relative Coordinate System.

Tests the spatial schemas and landmark manager for converting
absolute coordinates to relative positions based on learned landmarks.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from episodic_agent.schemas.spatial import (
    RelativePosition,
    LandmarkReference,
    SpatialRelation,
    PositionObservation,
    compute_distance,
    compute_direction,
    compute_bearing,
    classify_distance,
    classify_direction,
    RELATION_NEAR,
    RELATION_FAR,
    RELATION_AT,
    RELATION_LEFT,
    RELATION_RIGHT,
    RELATION_FRONT,
    RELATION_BEHIND,
)


# =============================================================================
# Spatial Utility Function Tests
# =============================================================================

class TestComputeDistance:
    """Tests for distance computation."""
    
    def test_same_point(self):
        """Distance between same point is zero."""
        pos = (1.0, 2.0, 3.0)
        assert compute_distance(pos, pos) == 0.0
    
    def test_horizontal_distance(self):
        """Distance in XZ plane."""
        pos1 = (0.0, 0.0, 0.0)
        pos2 = (3.0, 0.0, 4.0)
        assert compute_distance(pos1, pos2) == pytest.approx(5.0)
    
    def test_vertical_distance(self):
        """Distance along Y axis."""
        pos1 = (0.0, 0.0, 0.0)
        pos2 = (0.0, 5.0, 0.0)
        assert compute_distance(pos1, pos2) == pytest.approx(5.0)
    
    def test_3d_distance(self):
        """Full 3D distance."""
        pos1 = (1.0, 1.0, 1.0)
        pos2 = (2.0, 3.0, 3.0)
        # sqrt(1 + 4 + 4) = sqrt(9) = 3
        assert compute_distance(pos1, pos2) == pytest.approx(3.0)


class TestComputeDirection:
    """Tests for direction vector computation."""
    
    def test_same_point(self):
        """Direction from same point is zero."""
        pos = (1.0, 2.0, 3.0)
        direction = compute_direction(pos, pos)
        assert direction == (0.0, 0.0, 0.0)
    
    def test_positive_x(self):
        """Direction along positive X."""
        direction = compute_direction((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        assert direction[0] == pytest.approx(1.0)
        assert direction[1] == pytest.approx(0.0)
        assert direction[2] == pytest.approx(0.0)
    
    def test_normalized(self):
        """Direction vector is normalized."""
        direction = compute_direction((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))
        magnitude = (direction[0]**2 + direction[1]**2 + direction[2]**2) ** 0.5
        assert magnitude == pytest.approx(1.0)


class TestComputeBearing:
    """Tests for bearing and elevation computation."""
    
    def test_forward_bearing(self):
        """Bearing for point directly ahead (positive Z)."""
        bearing, elevation = compute_bearing((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
        assert bearing == pytest.approx(0.0)
        assert elevation == pytest.approx(0.0)
    
    def test_right_bearing(self):
        """Bearing for point to the right (positive X)."""
        bearing, elevation = compute_bearing((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
        assert bearing == pytest.approx(90.0)
    
    def test_left_bearing(self):
        """Bearing for point to the left (negative X)."""
        bearing, elevation = compute_bearing((0.0, 0.0, 0.0), (-5.0, 0.0, 0.0))
        assert bearing == pytest.approx(-90.0)
    
    def test_behind_bearing(self):
        """Bearing for point behind (negative Z)."""
        bearing, elevation = compute_bearing((0.0, 0.0, 0.0), (0.0, 0.0, -5.0))
        assert abs(bearing) == pytest.approx(180.0)
    
    def test_positive_elevation(self):
        """Elevation for point above."""
        bearing, elevation = compute_bearing((0.0, 0.0, 0.0), (0.0, 5.0, 5.0))
        assert elevation > 0
    
    def test_negative_elevation(self):
        """Elevation for point below."""
        bearing, elevation = compute_bearing((0.0, 0.0, 0.0), (0.0, -5.0, 5.0))
        assert elevation < 0


class TestClassifyDistance:
    """Tests for distance classification."""
    
    def test_at_threshold(self):
        """Very close distance is 'at'."""
        assert classify_distance(0.3) == RELATION_AT
    
    def test_near_threshold(self):
        """Close distance is 'near'."""
        assert classify_distance(1.5) == RELATION_NEAR
    
    def test_far_threshold(self):
        """Far distance is 'far'."""
        assert classify_distance(15.0) == RELATION_FAR
    
    def test_custom_thresholds(self):
        """Custom thresholds work."""
        thresholds = {"at": 1.0, "near": 5.0, "far": 20.0}
        assert classify_distance(0.5, thresholds) == RELATION_AT
        assert classify_distance(3.0, thresholds) == RELATION_NEAR
        assert classify_distance(10.0, thresholds) == RELATION_FAR


class TestClassifyDirection:
    """Tests for direction classification."""
    
    def test_front(self):
        """Direction ahead is 'front'."""
        assert classify_direction(0.0) == RELATION_FRONT
        assert classify_direction(30.0) == RELATION_FRONT
        assert classify_direction(-30.0) == RELATION_FRONT
    
    def test_right(self):
        """Direction to right."""
        assert classify_direction(90.0) == RELATION_RIGHT
        assert classify_direction(60.0) == RELATION_RIGHT
    
    def test_left(self):
        """Direction to left."""
        assert classify_direction(-90.0) == RELATION_LEFT
        assert classify_direction(-60.0) == RELATION_LEFT
    
    def test_behind(self):
        """Direction behind."""
        assert classify_direction(180.0) == RELATION_BEHIND
        assert classify_direction(-180.0) == RELATION_BEHIND
        assert classify_direction(150.0) == RELATION_BEHIND


# =============================================================================
# Schema Tests
# =============================================================================

class TestRelativePosition:
    """Tests for RelativePosition schema."""
    
    def test_minimal_creation(self):
        """Create with minimal fields."""
        pos = RelativePosition(
            landmark_id="lm_001",
        )
        assert pos.landmark_id == "lm_001"
        assert pos.distance == 0.0
        assert pos.confidence == 0.0
    
    def test_full_creation(self):
        """Create with all fields."""
        pos = RelativePosition(
            landmark_id="lm_001",
            landmark_label="Kitchen Table",
            distance=2.5,
            bearing=45.0,
            elevation=10.0,
            direction=(0.7, 0.1, 0.7),
            relation="near",
            confidence=0.9,
        )
        assert pos.landmark_label == "Kitchen Table"
        assert pos.distance == 2.5
        assert pos.bearing == 45.0
        assert pos.relation == "near"


class TestLandmarkReference:
    """Tests for LandmarkReference schema."""
    
    def test_minimal_creation(self):
        """Create with minimal fields."""
        lm = LandmarkReference(
            landmark_id="lm_001",
        )
        assert lm.landmark_id == "lm_001"
        assert lm.label == "unknown"
        assert lm.user_verified is False
    
    def test_full_creation(self):
        """Create with all fields."""
        lm = LandmarkReference(
            landmark_id="lm_001",
            label="Kitchen Table",
            internal_position=(5.0, 0.0, 3.0),
            location_id="room_kitchen",
            location_label="Kitchen",
            observation_count=10,
            is_static=True,
            user_verified=True,
        )
        assert lm.label == "Kitchen Table"
        assert lm.location_id == "room_kitchen"
        assert lm.observation_count == 10
        assert lm.user_verified is True


class TestSpatialRelation:
    """Tests for SpatialRelation schema."""
    
    def test_creation(self):
        """Create spatial relation."""
        rel = SpatialRelation(
            subject_id="ball_001",
            subject_label="red ball",
            reference_id="table_001",
            reference_label="kitchen table",
            relation="near_left",
            distance=1.5,
            source="computed",
            confidence=0.85,
        )
        assert rel.subject_id == "ball_001"
        assert rel.reference_id == "table_001"
        assert rel.relation == "near_left"
        assert rel.distance == 1.5


class TestPositionObservation:
    """Tests for PositionObservation schema."""
    
    def test_minimal_creation(self):
        """Create with minimal fields."""
        obs = PositionObservation()
        assert obs.raw_position is None
        assert obs.relative_positions == []
    
    def test_with_relative_positions(self):
        """Create with relative positions."""
        rel_pos = RelativePosition(
            landmark_id="lm_001",
            landmark_label="Table",
            distance=2.0,
            relation="near",
        )
        obs = PositionObservation(
            raw_position=(5.0, 0.0, 3.0),
            relative_positions=[rel_pos],
            primary_landmark_id="lm_001",
            primary_landmark_label="Table",
            qualitative_position="near the Table, front",
        )
        assert len(obs.relative_positions) == 1
        assert obs.primary_landmark_label == "Table"


# =============================================================================
# Landmark Manager Tests
# =============================================================================

class TestLandmarkManager:
    """Tests for LandmarkManager functionality."""
    
    @pytest.fixture
    def mock_graph_store(self):
        """Create a mock graph store."""
        store = MagicMock()
        store.get_nodes_by_type.return_value = []
        return store
    
    @pytest.fixture
    def manager(self, mock_graph_store):
        """Create a landmark manager with mocked dependencies."""
        from episodic_agent.modules.landmark_manager import LandmarkManager
        return LandmarkManager(
            graph_store=mock_graph_store,
            dialog_manager=None,
            min_observations_for_suggestion=3,
            stability_threshold_seconds=1.0,  # Short for testing
        )
    
    def test_initial_state(self, manager):
        """Manager starts with no landmarks."""
        assert manager.get_all_landmarks() == []
    
    def test_add_landmark(self, manager):
        """Add a landmark."""
        lm = manager.add_landmark(
            landmark_id="lm_001",
            label="Kitchen Table",
            position=(5.0, 0.0, 3.0),
            location_id="room_kitchen",
        )
        
        assert lm.landmark_id == "lm_001"
        assert lm.label == "Kitchen Table"
        assert manager.get_landmark("lm_001") is not None
    
    def test_get_landmarks_in_location(self, manager):
        """Get landmarks filtered by location."""
        manager.add_landmark("lm_001", "Table", (0, 0, 0), "kitchen")
        manager.add_landmark("lm_002", "Chair", (1, 0, 0), "kitchen")
        manager.add_landmark("lm_003", "Sofa", (5, 0, 5), "living_room")
        
        kitchen_landmarks = manager.get_landmarks_in_location("kitchen")
        assert len(kitchen_landmarks) == 2
        
        living_landmarks = manager.get_landmarks_in_location("living_room")
        assert len(living_landmarks) == 1
    
    def test_compute_relative_position(self, manager):
        """Compute position relative to landmarks."""
        manager.add_landmark("lm_001", "Table", (0.0, 0.0, 0.0))
        manager.add_landmark("lm_002", "Door", (10.0, 0.0, 0.0))
        
        # Position near the table
        obs = manager.compute_relative_position((1.0, 0.0, 0.0))
        
        assert len(obs.relative_positions) > 0
        # First should be nearest (the table)
        assert obs.relative_positions[0].landmark_label == "Table"
        assert obs.relative_positions[0].distance == pytest.approx(1.0)
    
    def test_compute_spatial_relation(self, manager):
        """Compute spatial relation between entities."""
        rel = manager.compute_spatial_relation(
            subject_id="ball",
            subject_position=(2.0, 0.0, 0.0),
            reference_id="table",
            reference_position=(0.0, 0.0, 0.0),
            subject_label="Ball",
            reference_label="Table",
        )
        
        assert rel.subject_id == "ball"
        assert rel.reference_id == "table"
        assert rel.distance == pytest.approx(2.0)
        assert "near" in rel.relation or "right" in rel.relation
    
    def test_entity_observation_tracking(self, manager):
        """Track entity observations for landmark candidacy."""
        import time
        
        # First few observations - not yet a candidate
        for i in range(2):
            result = manager.record_entity_observation(
                "entity_001",
                (5.0, 0.0, 3.0),
                label="Chair",
            )
            assert result is False
        
        # After stability threshold and enough observations
        time.sleep(1.1)  # Wait for stability threshold
        result = manager.record_entity_observation(
            "entity_001",
            (5.0, 0.0, 3.0),
            label="Chair",
        )
        assert result is True  # Now ready to suggest as landmark
    
    def test_verify_landmark(self, manager):
        """Verify a landmark with user label."""
        manager.add_landmark("lm_001", "unknown_object", (0, 0, 0))
        
        success = manager.verify_landmark("lm_001", "Kitchen Chair")
        assert success is True
        
        lm = manager.get_landmark("lm_001")
        assert lm.user_verified is True
        assert lm.label == "Kitchen Chair"
    
    def test_remove_landmark(self, manager):
        """Remove a landmark."""
        manager.add_landmark("lm_001", "Table", (0, 0, 0))
        assert manager.get_landmark("lm_001") is not None
        
        success = manager.remove_landmark("lm_001")
        assert success is True
        assert manager.get_landmark("lm_001") is None
    
    def test_qualitative_position_description(self, manager):
        """Generate qualitative position descriptions."""
        manager.add_landmark("lm_001", "Kitchen Table", (0.0, 0.0, 0.0))
        
        # Position to the right of the table
        obs = manager.compute_relative_position((5.0, 0.0, 0.0))
        
        assert obs.qualitative_position is not None
        assert "Kitchen Table" in obs.qualitative_position


# =============================================================================
# Integration Tests
# =============================================================================

class TestRelativePositionIntegration:
    """Integration tests for relative position system."""
    
    def test_delta_with_relative_positions(self):
        """Delta can store relative position information."""
        from episodic_agent.schemas import Delta
        
        delta = Delta(
            delta_id="d_001",
            delta_type="moved",
            entity_id="ball_001",
            entity_label="Red Ball",
            pre_position=(0.0, 0.0, 0.0),
            post_position=(5.0, 0.0, 0.0),
            position_delta=5.0,
            pre_relative_position={
                "landmark_id": "table",
                "relation": "at",
                "distance": 0.5,
            },
            post_relative_position={
                "landmark_id": "table",
                "relation": "far",
                "distance": 5.0,
            },
            movement_description="moved from at the table to far from the table",
        )
        
        assert delta.pre_relative_position["relation"] == "at"
        assert delta.post_relative_position["relation"] == "far"
        assert "table" in delta.movement_description
    
    def test_entity_observation_with_spatial_info(self):
        """EntityObservation can store spatial relation info."""
        from episodic_agent.modules.sensor_gateway.types import EntityObservation
        
        obs = EntityObservation(
            entity_id="ball_001",
            label="Red Ball",
            position=(5.0, 0.0, 3.0),
            spatial_relation="near_left",
            reference_landmark="table_001",
        )
        
        assert obs.spatial_relation == "near_left"
        assert obs.reference_landmark == "table_001"
    
    def test_location_context_with_relative_info(self):
        """LocationContext can store relative position info."""
        from episodic_agent.modules.sensor_gateway.types import LocationContext
        
        ctx = LocationContext(
            position=(5.0, 0.0, 3.0),
            position_confidence=0.9,
            room_id="room_kitchen",
            qualitative_position="near the kitchen table, to the left",
        )
        
        assert ctx.qualitative_position is not None
        assert "kitchen table" in ctx.qualitative_position
