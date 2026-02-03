"""Unit tests for Phase 2 components."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from episodic_agent.memory.episode_store import PersistentEpisodeStore
from episodic_agent.memory.graph_store import LabeledGraphStore
from episodic_agent.modules.boundary import (
    BoundaryReason,
    HysteresisBoundaryDetector,
)
from episodic_agent.modules.dialog import (
    AutoAcceptDialogManager,
    CLIDialogManager,
)
from episodic_agent.modules.label_manager import LabelManager
from episodic_agent.schemas import (
    ActiveContextFrame,
    Episode,
    GraphEdge,
    GraphNode,
)
from episodic_agent.schemas.graph import EdgeType, NodeType
from episodic_agent.schemas.labels import (
    ConflictResolutionType,
    LabelConflict,
)
from episodic_agent.utils.confidence import ConfidenceHelper, ConfidenceSignal


# =============================================================================
# PersistentEpisodeStore Tests
# =============================================================================

class TestPersistentEpisodeStore:
    """Tests for JSONL-based episode persistence."""

    def _make_episode(self, episode_id: str, location: str = "kitchen") -> Episode:
        """Create a test episode with required fields."""
        return Episode(
            episode_id=episode_id,
            location_label=location,
            start_time=datetime.now(),
            end_time=datetime.now(),
            step_count=10,
            source_acf_id=f"acf_{episode_id}",
        )

    def test_store_and_retrieve(self) -> None:
        """Store an episode and retrieve it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PersistentEpisodeStore(Path(tmpdir) / "episodes.jsonl")
            
            episode = self._make_episode("ep_001", "kitchen")
            store.store(episode)
            
            retrieved = store.get("ep_001")
            assert retrieved is not None
            assert retrieved.episode_id == "ep_001"
            assert retrieved.location_label == "kitchen"

    def test_append_only_persistence(self) -> None:
        """Verify episodes are appended to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "episodes.jsonl"
            store = PersistentEpisodeStore(jsonl_path)
            
            for i in range(3):
                episode = self._make_episode(f"ep_{i:03d}", f"room_{i}")
                store.store(episode)
            
            # Read the JSONL file directly
            with open(jsonl_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            assert json.loads(lines[0])["episode_id"] == "ep_000"
            assert json.loads(lines[1])["episode_id"] == "ep_001"
            assert json.loads(lines[2])["episode_id"] == "ep_002"

    def test_load_on_init(self) -> None:
        """Verify episodes are loaded from existing file on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "episodes.jsonl"
            
            # Store some episodes
            store1 = PersistentEpisodeStore(jsonl_path)
            for i in range(3):
                episode = self._make_episode(f"ep_{i:03d}", f"room_{i}")
                store1.store(episode)
            
            # Create new store instance
            store2 = PersistentEpisodeStore(jsonl_path)
            
            assert store2.count() == 3
            assert store2.get("ep_001") is not None

    def test_get_by_location(self) -> None:
        """Test filtering episodes by location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PersistentEpisodeStore(Path(tmpdir) / "episodes.jsonl")
            
            locations = ["kitchen", "kitchen", "bedroom", "kitchen"]
            for i, loc in enumerate(locations):
                episode = self._make_episode(f"ep_{i:03d}", loc)
                store.store(episode)
            
            kitchen_episodes = store.get_by_location("kitchen")
            assert len(kitchen_episodes) == 3

    def test_get_recent(self) -> None:
        """Test getting recent episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PersistentEpisodeStore(Path(tmpdir) / "episodes.jsonl")
            
            for i in range(5):
                episode = self._make_episode(f"ep_{i:03d}", f"room_{i}")
                store.store(episode)
            
            recent = store.get_recent(3)
            assert len(recent) == 3
            assert recent[0].episode_id == "ep_004"  # Most recent first


# =============================================================================
# LabeledGraphStore Tests
# =============================================================================

class TestLabeledGraphStore:
    """Tests for graph memory with label indexing."""

    def test_add_and_get_node(self) -> None:
        """Add a node and retrieve it."""
        store = LabeledGraphStore()
        
        node = GraphNode(
            node_id="node_001",
            node_type=NodeType.ENTITY,
            label="coffee_mug",
            confidence=0.9,
        )
        
        store.add_node(node)
        
        retrieved = store.get_node("node_001")
        assert retrieved is not None
        assert retrieved.label == "coffee_mug"

    def test_label_index(self) -> None:
        """Test label-based lookup."""
        store = LabeledGraphStore()
        
        # Add nodes with same label
        for i in range(3):
            node = GraphNode(
                node_id=f"node_{i:03d}",
                node_type=NodeType.ENTITY,
                label="chair",
                confidence=0.5 + i * 0.1,
            )
            store.add_node(node)
        
        # Different label
        store.add_node(GraphNode(
            node_id="node_other",
            node_type=NodeType.ENTITY,
            label="table",
            confidence=0.8,
        ))
        
        chair_nodes = store.get_nodes_by_label("chair")
        assert len(chair_nodes) == 3

    def test_edges(self) -> None:
        """Test edge creation and neighbor queries."""
        store = LabeledGraphStore()
        
        # Add nodes
        store.add_node(GraphNode(
            node_id="person_1",
            node_type=NodeType.ENTITY,
            label="person",
        ))
        store.add_node(GraphNode(
            node_id="kitchen_1",
            node_type=NodeType.LOCATION,
            label="kitchen",
        ))
        
        # Add edge
        edge = GraphEdge(
            edge_id="edge_001",
            source_node_id="person_1",
            target_node_id="kitchen_1",
            edge_type=EdgeType.TYPICAL_IN,
            confidence=0.9,
        )
        store.add_edge(edge)
        
        # Query neighbors
        neighbors = store.get_neighbors("person_1")
        assert len(neighbors) == 1
        assert neighbors[0].node_id == "kitchen_1"

    def test_get_nodes_by_type(self) -> None:
        """Test filtering nodes by type."""
        store = LabeledGraphStore()
        
        store.add_node(GraphNode(node_id="e1", node_type=NodeType.ENTITY, label="chair"))
        store.add_node(GraphNode(node_id="e2", node_type=NodeType.ENTITY, label="table"))
        store.add_node(GraphNode(node_id="l1", node_type=NodeType.LOCATION, label="room"))
        
        entities = store.get_nodes_by_type(NodeType.ENTITY)
        assert len(entities) == 2
        
        locations = store.get_nodes_by_type(NodeType.LOCATION)
        assert len(locations) == 1

    def test_alias_edge(self) -> None:
        """Test creating alias relationships."""
        store = LabeledGraphStore()
        
        store.add_node(GraphNode(node_id="n1", node_type=NodeType.ENTITY, label="mug"))
        store.add_node(GraphNode(node_id="n2", node_type=NodeType.ENTITY, label="cup"))
        
        edge = store.create_alias_edge("n1", "n2")
        assert edge.edge_type == EdgeType.ALIAS_OF


# =============================================================================
# LabelManager Tests
# =============================================================================

class TestLabelManager:
    """Tests for label assignment and conflict resolution."""

    def test_assign_new_label(self) -> None:
        """Assign a label to a new node without conflicts."""
        store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        manager = LabelManager(store, dialog)
        
        node = GraphNode(
            node_id="node_001",
            node_type=NodeType.ENTITY,
            label="coffee_mug",
            confidence=0.9,
        )
        store.add_node(node)
        
        success, conflict = manager.assign_label("node_001", "coffee_mug")
        
        assert success is True
        assert conflict is None

    def test_detect_label_conflict(self) -> None:
        """Detect conflict when label already exists for different node."""
        store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        manager = LabelManager(store, dialog)
        
        # Add first node with label
        node1 = GraphNode(
            node_id="node_001",
            node_type=NodeType.ENTITY,
            label="mug",
            confidence=0.9,
        )
        store.add_node(node1)
        manager.assign_label("node_001", "mug")
        
        # Add second node and try to assign same label
        node2 = GraphNode(
            node_id="node_002",
            node_type=NodeType.ENTITY,
            label="cup",  # Different initial label
            confidence=0.8,
        )
        store.add_node(node2)
        
        # Try to assign "mug" to node2 - should create conflict
        success, conflict = manager.assign_label("node_002", "mug")
        
        # Should detect conflict
        assert success is False
        assert conflict is not None
        assert conflict.label == "mug"

    def test_conflict_resolution_merge(self) -> None:
        """Test that merge resolution creates merge edge."""
        store = LabeledGraphStore()
        dialog = AutoAcceptDialogManager()
        manager = LabelManager(store, dialog)
        
        # Setup two nodes
        node1 = GraphNode(node_id="n1", node_type=NodeType.ENTITY, label="chair_1")
        node2 = GraphNode(node_id="n2", node_type=NodeType.ENTITY, label="chair_2")
        store.add_node(node1)
        store.add_node(node2)
        
        manager.assign_label("n1", "chair")
        
        # Now try to assign same label to n2 - creates conflict
        success, conflict = manager.assign_label("n2", "chair")
        assert success is False
        assert conflict is not None
        
        # Resolve the conflict (auto-accept chooses first option = merge)
        resolution = manager.resolve_conflict(conflict.conflict_id)
        
        # Check for merge edges
        edges = store.get_outgoing_edges("n2")
        merge_edges = [e for e in edges if e.edge_type == EdgeType.MERGED_INTO]
        assert len(merge_edges) == 1


# =============================================================================
# HysteresisBoundaryDetector Tests
# =============================================================================

class TestHysteresisBoundaryDetector:
    """Tests for boundary detection with hysteresis."""

    def _make_acf(
        self,
        step: int,
        location: str = "room_a",
        confidence: float = 0.8,
    ) -> ActiveContextFrame:
        """Create a test ACF."""
        return ActiveContextFrame(
            acf_id=f"acf_{step}",
            location_label=location,
            location_confidence=confidence,
            step_count=step,
        )

    def test_timeout_boundary(self) -> None:
        """Test timeout triggers boundary after max frames."""
        detector = HysteresisBoundaryDetector(
            timeout_frames=10,
            min_frames_hysteresis=3,
        )
        
        # Run 9 frames - no boundary
        for i in range(9):
            triggered, _ = detector.check(self._make_acf(i))
            assert triggered is False
        
        # 10th frame triggers timeout
        triggered, reason = detector.check(self._make_acf(9))
        assert triggered is True
        assert BoundaryReason.TIMEOUT.value in reason

    def test_interval_boundary(self) -> None:
        """Test periodic interval triggers boundary."""
        detector = HysteresisBoundaryDetector(
            timeout_frames=1000,
            interval_frames=50,
        )
        
        # Run 49 frames - no boundary
        for i in range(1, 50):
            triggered, _ = detector.check(self._make_acf(i))
            assert triggered is False
        
        # 50th frame triggers interval
        triggered, reason = detector.check(self._make_acf(50))
        assert triggered is True
        assert BoundaryReason.INTERVAL.value in reason

    def test_location_change_with_hysteresis(self) -> None:
        """Test location change respects hysteresis minimum frames."""
        detector = HysteresisBoundaryDetector(
            timeout_frames=1000,
            min_frames_hysteresis=5,
            location_confidence_threshold=0.7,
        )
        
        # Run 6 frames in room_a to establish location (past hysteresis minimum)
        for i in range(6):
            triggered, _ = detector.check(self._make_acf(i, "room_a", 0.9))
            assert triggered is False, f"Should not trigger while establishing location at step {i}"
        
        # Now change to room_b with high confidence
        # First frame of change (i=6) - increments change counter
        triggered, _ = detector.check(self._make_acf(6, "room_b", 0.9))
        assert triggered is False, "First frame of change should not trigger"
        
        # Second frame of change (i=7) - should trigger
        triggered, reason = detector.check(self._make_acf(7, "room_b", 0.9))
        
        assert triggered is True, "Should trigger after sustained location change"
        assert BoundaryReason.LOCATION_CHANGE.value in reason

    def test_low_confidence_no_boundary(self) -> None:
        """Test that low confidence location changes don't trigger."""
        detector = HysteresisBoundaryDetector(
            timeout_frames=1000,
            min_frames_hysteresis=3,
            location_confidence_threshold=0.8,
        )
        
        # Establish room_a
        for i in range(5):
            detector.check(self._make_acf(i, "room_a", 0.9))
        
        # Change to room_b with low confidence
        for i in range(5, 10):
            triggered, _ = detector.check(self._make_acf(i, "room_b", 0.3))
            # Should not trigger on location change due to low confidence
            assert triggered is False

    def test_reset_after_boundary(self) -> None:
        """Test that state resets after boundary."""
        detector = HysteresisBoundaryDetector(
            timeout_frames=5,
        )
        
        # Trigger timeout
        for i in range(5):
            detector.check(self._make_acf(i))
        
        # State should be reset
        assert detector.frames_since_boundary == 0


# =============================================================================
# ConfidenceHelper Tests
# =============================================================================

class TestConfidenceHelper:
    """Tests for confidence calculation utilities."""

    def test_combine_weighted(self) -> None:
        """Test weighted combination of confidence signals."""
        helper = ConfidenceHelper()
        
        signals = [
            ConfidenceSignal(name="visual", value=0.8, weight=2.0),
            ConfidenceSignal(name="audio", value=0.6, weight=1.0),
        ]
        
        result = helper.combine_weighted(signals)
        
        # (0.8 * 2 + 0.6 * 1) / 3 = 2.2 / 3 ≈ 0.733
        assert 0.73 <= result <= 0.74

    def test_combine_max(self) -> None:
        """Test max combination."""
        helper = ConfidenceHelper()
        
        signals = [
            ConfidenceSignal(name="a", value=0.3, weight=1.0),
            ConfidenceSignal(name="b", value=0.9, weight=1.0),
            ConfidenceSignal(name="c", value=0.5, weight=1.0),
        ]
        
        result = helper.combine_max(signals)
        assert result == 0.9

    def test_threshold_checks(self) -> None:
        """Test low/high threshold checks."""
        helper = ConfidenceHelper(t_low=0.3, t_high=0.7)
        
        assert helper.is_low(0.2) is True
        assert helper.is_low(0.5) is False
        
        assert helper.is_high(0.8) is True
        assert helper.is_high(0.5) is False

    def test_categorize(self) -> None:
        """Test confidence categorization."""
        helper = ConfidenceHelper(t_low=0.3, t_high=0.7)
        
        assert helper.categorize(0.1) == "low"
        assert helper.categorize(0.5) == "medium"
        assert helper.categorize(0.9) == "high"

    def test_decay(self) -> None:
        """Test confidence decay."""
        helper = ConfidenceHelper()
        
        # Single step decay with factor 0.9
        result = helper.decay(1.0, factor=0.9)
        assert result == 0.9
        
        # Multiple applications
        conf = 1.0
        for _ in range(10):
            conf = helper.decay(conf, factor=0.9)
        # 1.0 * 0.9^10 ≈ 0.349
        assert 0.34 <= conf <= 0.36

    def test_boost(self) -> None:
        """Test confidence boost with clipping."""
        helper = ConfidenceHelper()
        
        result = helper.boost(0.5, amount=0.3)
        assert result == 0.8
        
        # Test clipping
        result = helper.boost(0.9, amount=0.5)
        assert result == 1.0


# =============================================================================
# Dialog Manager Tests
# =============================================================================

class TestAutoAcceptDialogManager:
    """Tests for auto-accept dialog manager."""

    def test_confirm_returns_default(self) -> None:
        """Confirm returns the default value."""
        dialog = AutoAcceptDialogManager()
        
        assert dialog.confirm("Test?", default=True) is True
        assert dialog.confirm("Test?", default=False) is False

    def test_ask_label_returns_first_suggestion(self) -> None:
        """Ask label returns first suggestion if available."""
        dialog = AutoAcceptDialogManager()
        
        result = dialog.ask_label("Label?", suggestions=["apple", "banana"])
        assert result == "apple"

    def test_ask_label_generates_if_no_suggestions(self) -> None:
        """Ask label generates a label if no suggestions."""
        dialog = AutoAcceptDialogManager(default_label="test")
        
        result = dialog.ask_label("Label?")
        assert result.startswith("test_")

    def test_resolve_conflict_returns_first(self) -> None:
        """Resolve conflict returns first option."""
        dialog = AutoAcceptDialogManager()
        conflict = LabelConflict(
            conflict_id="c1",
            label="test",
            existing_node_id="n1",
            new_node_id="n2",
            reason="Same label",
        )
        
        result = dialog.resolve_conflict(conflict, ["Merge", "Keep both"])
        assert result == 0

    def test_interactions_recorded(self) -> None:
        """All interactions are recorded."""
        dialog = AutoAcceptDialogManager()
        
        dialog.confirm("Q1?")
        dialog.ask_label("Q2?", ["a", "b"])
        dialog.notify("Info")
        
        interactions = dialog.interactions
        assert len(interactions) == 3
        assert interactions[0]["type"] == "confirm"
        assert interactions[1]["type"] == "ask_label"
        assert interactions[2]["type"] == "notify"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
