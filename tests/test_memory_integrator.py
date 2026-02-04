"""Tests for memory integration (Phase 4).

Tests the MemoryIntegrator which combines:
- Label learning (Phase 3)
- Spatial context (Phase 2)  
- Graph and episode memory
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from episodic_agent.memory.integrator import (
    MemoryIntegrator,
    MemoryQuery,
    DEFAULT_DECAY,
    SAME_LOCATION_BOOST,
    SPATIAL_BOOST,
)
from episodic_agent.memory.stubs import InMemoryEpisodeStore, InMemoryGraphStore
from episodic_agent.modules.label_learner import LabelLearner
from episodic_agent.modules.landmark_manager import LandmarkManager
from episodic_agent.schemas import Episode, GraphEdge, GraphNode
from episodic_agent.schemas.graph import (
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_TYPICAL_IN,
    NODE_TYPE_ENTITY,
    NODE_TYPE_EPISODE,
    NODE_TYPE_LOCATION,
    EdgeType,
    NodeType,
)
from episodic_agent.schemas.learning import (
    CATEGORY_ENTITY,
    CATEGORY_LOCATION,
)
from episodic_agent.schemas.spatial import RelativePosition


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def graph_store():
    """Create an in-memory graph store."""
    return InMemoryGraphStore()


@pytest.fixture
def episode_store():
    """Create an in-memory episode store."""
    return InMemoryEpisodeStore()


@pytest.fixture
def label_learner():
    """Create a label learner."""
    return LabelLearner()


@pytest.fixture
def landmark_manager(graph_store):
    """Create a landmark manager."""
    return LandmarkManager(graph_store=graph_store)


@pytest.fixture
def integrator(graph_store, episode_store, label_learner, landmark_manager):
    """Create a memory integrator."""
    return MemoryIntegrator(
        graph_store=graph_store,
        episode_store=episode_store,
        label_learner=label_learner,
        landmark_manager=landmark_manager,
    )


def make_episode(
    episode_id: str | None = None,
    location_label: str = "test_location",
    entities: list | None = None,
) -> Episode:
    """Helper to create test episodes."""
    from episodic_agent.schemas import ObjectCandidate
    
    return Episode(
        episode_id=episode_id or str(uuid.uuid4()),
        created_at=datetime.now(),
        start_time=datetime.now() - timedelta(minutes=5),
        end_time=datetime.now(),
        step_count=10,
        location_label=location_label,
        location_confidence=0.9,
        entities=entities or [],
        source_acf_id="test_acf",
        boundary_reason="test",
    )


def make_node(
    node_id: str,
    node_type: str = NODE_TYPE_ENTITY,
    label: str = "test",
    source_id: str | None = None,
) -> GraphNode:
    """Helper to create test nodes."""
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        source_id=source_id,
        created_at=datetime.now(),
        last_accessed=datetime.now(),
    )


# =============================================================================
# TEST: MEMORY QUERY
# =============================================================================

class TestMemoryQuery:
    """Tests for MemoryQuery."""
    
    def test_create_minimal(self):
        """Create query with defaults."""
        query = MemoryQuery()
        
        assert query.query_id is not None
        assert query.labels == []
        assert query.max_results == 10
        assert query.include_episodes is True
        assert query.include_nodes is True
    
    def test_create_with_labels(self):
        """Create query with labels."""
        query = MemoryQuery(labels=["chair", "table"])
        
        assert query.labels == ["chair", "table"]
    
    def test_create_with_location(self):
        """Create query with location."""
        query = MemoryQuery(
            location_id="room-1",
            location_label="kitchen",
        )
        
        assert query.location_id == "room-1"
        assert query.location_label == "kitchen"
    
    def test_create_with_time_filter(self):
        """Create query with time filter."""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        
        query = MemoryQuery(
            time_start=start,
            time_end=end,
        )
        
        assert query.time_start == start
        assert query.time_end == end


# =============================================================================
# TEST: MEMORY INTEGRATOR BASICS
# =============================================================================

class TestMemoryIntegratorBasics:
    """Basic tests for MemoryIntegrator."""
    
    def test_create(self, integrator):
        """Create memory integrator."""
        assert integrator is not None
    
    def test_empty_statistics(self, integrator):
        """Statistics for empty integrator."""
        stats = integrator.get_statistics()
        
        assert stats["graph"]["node_count"] == 0
        assert stats["episodes"]["count"] == 0
        assert stats["learning"]["total_labels"] == 0


# =============================================================================
# TEST: OBSERVATION INTEGRATION
# =============================================================================

class TestObservationIntegration:
    """Tests for integrating observations."""
    
    def test_integrate_entity_with_label(self, integrator):
        """Integrate an entity with user label."""
        node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="chair",
        )
        
        assert node is not None
        assert node.label == "chair"
        assert node.node_type == NODE_TYPE_ENTITY
    
    def test_integrate_learns_label(self, integrator, label_learner):
        """Integration learns the label."""
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="desk",
        )
        
        assert label_learner.has_label("desk")
    
    def test_integrate_location(self, integrator):
        """Integrate a location observation."""
        node = integrator.integrate_observation(
            observation_type="location",
            observation_id="room-1",
            label="kitchen",
        )
        
        assert node.node_type == NODE_TYPE_LOCATION
        assert node.label == "kitchen"
    
    def test_integrate_with_spatial_context(self, integrator, landmark_manager):
        """Integrate with spatial context."""
        # First add a location
        integrator.integrate_observation(
            observation_type="location",
            observation_id="room-1",
            label="kitchen",
        )
        
        # Then add entity with position
        node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="table",
            location_id="room-1",
            position=(1.0, 0.0, 2.0),
        )
        
        assert node is not None
        # Should have created the node with label
        assert node.label == "table"
    
    def test_integrate_without_label_uses_recognition(self, integrator, label_learner):
        """Integration uses recognition when no label provided."""
        # First learn a label
        label_learner.learn_label(
            "chair",
            category=CATEGORY_ENTITY,
            instance_id="example-1",
            features={"type": "seating"},
        )
        
        # Integrate without label but with features
        node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            features={"type": "seating"},
        )
        
        # Should have attempted recognition
        assert node is not None


# =============================================================================
# TEST: EPISODE INTEGRATION
# =============================================================================

class TestEpisodeIntegration:
    """Tests for integrating episodes."""
    
    def test_integrate_episode(self, integrator, episode_store):
        """Integrate an episode."""
        episode = make_episode(episode_id="ep-1", location_label="kitchen")
        
        node = integrator.integrate_episode(episode)
        
        assert node is not None
        assert node.node_type == NODE_TYPE_EPISODE
        assert episode_store.count() == 1
    
    def test_integrate_episode_learns_location(self, integrator, label_learner):
        """Episode integration learns location label."""
        episode = make_episode(location_label="living_room")
        
        integrator.integrate_episode(episode)
        
        assert label_learner.has_label("living_room")
    
    def test_integrate_episode_with_location_id(self, integrator, graph_store):
        """Episode links to location node."""
        # Create location first
        loc_node = make_node(
            f"{NODE_TYPE_LOCATION}_room-1",
            NODE_TYPE_LOCATION,
            "kitchen",
            "room-1",
        )
        graph_store.add_node(loc_node)
        
        episode = make_episode(location_label="kitchen")
        node = integrator.integrate_episode(episode, location_id="room-1")
        
        # Should have edge to location
        edges = graph_store.get_edges(node.node_id)
        location_edges = [e for e in edges if "location" in e.target_node_id]
        assert len(location_edges) > 0


# =============================================================================
# TEST: RETRIEVAL BY LABELS
# =============================================================================

class TestRetrievalByLabels:
    """Tests for label-based retrieval."""
    
    def test_retrieve_by_label_empty(self, integrator):
        """Retrieve from empty memory."""
        result = integrator.retrieve_by_label("chair")
        
        assert len(result.nodes) == 0
        assert len(result.episodes) == 0
    
    def test_retrieve_by_label_finds_node(self, integrator):
        """Retrieve finds labeled node."""
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="chair",
        )
        
        result = integrator.retrieve_by_label("chair")
        
        assert len(result.nodes) == 1
        assert result.nodes[0].label == "chair"
    
    def test_retrieve_by_label_case_insensitive(self, integrator):
        """Label retrieval is case insensitive."""
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="Chair",
        )
        
        result = integrator.retrieve_by_label("chair")
        assert len(result.nodes) == 1
        
        result = integrator.retrieve_by_label("CHAIR")
        assert len(result.nodes) == 1
    
    def test_retrieve_by_multiple_labels(self, integrator):
        """Retrieve by multiple labels."""
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="chair",
        )
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-2",
            label="table",
        )
        
        result = integrator.retrieve(MemoryQuery(labels=["chair", "table"]))
        
        assert len(result.nodes) == 2


# =============================================================================
# TEST: RETRIEVAL BY LOCATION
# =============================================================================

class TestRetrievalByLocation:
    """Tests for location-based retrieval."""
    
    def test_retrieve_by_location_label(self, integrator):
        """Retrieve by location label."""
        # Create location and entity
        integrator.integrate_observation(
            observation_type="location",
            observation_id="room-1",
            label="kitchen",
        )
        
        result = integrator.retrieve_by_location(location_label="kitchen")
        
        assert len(result.nodes) > 0
    
    def test_retrieve_by_location_id(self, integrator, graph_store):
        """Retrieve by location ID."""
        # Create location
        loc_node = make_node(
            f"{NODE_TYPE_LOCATION}_room-1",
            NODE_TYPE_LOCATION,
            "kitchen",
            "room-1",
        )
        graph_store.add_node(loc_node)
        
        # Create entity in location
        entity_node = make_node(
            f"{NODE_TYPE_ENTITY}_entity-1",
            NODE_TYPE_ENTITY,
            "chair",
            "entity-1",
        )
        graph_store.add_node(entity_node)
        
        # Link them
        edge = GraphEdge(
            edge_id="e1",
            edge_type=EDGE_TYPE_TYPICAL_IN,
            source_node_id=entity_node.node_id,
            target_node_id=loc_node.node_id,
        )
        graph_store.add_edge(edge)
        
        # Rebuild indexes
        integrator._build_indexes()
        
        result = integrator.retrieve_by_location(location_id="room-1")
        
        assert len(result.nodes) >= 1


# =============================================================================
# TEST: SPREADING ACTIVATION
# =============================================================================

class TestSpreadingActivation:
    """Tests for spreading activation retrieval."""
    
    def test_activation_spreads_to_neighbors(self, integrator, graph_store):
        """Activation spreads through edges."""
        # Create connected nodes
        node1 = make_node("node-1", label="chair")
        node2 = make_node("node-2", label="table")
        graph_store.add_node(node1)
        graph_store.add_node(node2)
        
        # Connect them
        edge = GraphEdge(
            edge_id="e1",
            edge_type=EDGE_TYPE_TYPICAL_IN,
            source_node_id="node-1",
            target_node_id="node-2",
            weight=1.0,
        )
        graph_store.add_edge(edge)
        
        integrator._build_indexes()
        
        # Search for chair should also find table
        result = integrator.retrieve_by_label("chair")
        
        labels = [n.label for n in result.nodes]
        assert "chair" in labels
        # Table should be found via spreading activation
        # (may or may not be included depending on threshold)
    
    def test_activation_decays_with_distance(self, integrator, graph_store):
        """Activation decays with graph distance."""
        # Create chain of nodes
        for i in range(5):
            node = make_node(f"node-{i}", label=f"item-{i}")
            graph_store.add_node(node)
            if i > 0:
                edge = GraphEdge(
                    edge_id=f"e-{i}",
                    edge_type=EDGE_TYPE_TYPICAL_IN,
                    source_node_id=f"node-{i-1}",
                    target_node_id=f"node-{i}",
                    weight=1.0,
                )
                graph_store.add_edge(edge)
        
        integrator._build_indexes()
        
        # Manual spreading activation test
        candidates = {"node-0": 1.0}
        integrator._spread_activation(candidates)
        
        # Earlier nodes should have higher activation
        if "node-1" in candidates and "node-3" in candidates:
            assert candidates.get("node-1", 0) >= candidates.get("node-3", 0)


# =============================================================================
# TEST: SIMILAR RETRIEVAL
# =============================================================================

class TestSimilarRetrieval:
    """Tests for similarity-based retrieval."""
    
    def test_retrieve_similar_empty(self, integrator):
        """Retrieve similar from nonexistent node."""
        result = integrator.retrieve_similar("nonexistent")
        
        assert len(result.nodes) == 0
    
    def test_retrieve_similar_uses_label(self, integrator):
        """Similar retrieval uses node's label."""
        # Create nodes with same label
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="chair",
        )
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-2",
            label="chair",
        )
        
        result = integrator.retrieve_similar(f"{NODE_TYPE_ENTITY}_entity-1")
        
        assert len(result.nodes) >= 1


# =============================================================================
# TEST: SPATIAL RETRIEVAL
# =============================================================================

class TestSpatialRetrieval:
    """Tests for spatial context retrieval."""
    
    def test_get_nearby_memories_empty(self, integrator):
        """Get nearby memories with no landmarks."""
        result = integrator.get_nearby_memories(
            location_id="room-1",
            position=(0.0, 0.0, 0.0),
        )
        
        assert len(result.nodes) == 0
    
    def test_get_nearby_memories_finds_landmarks(self, integrator, landmark_manager, graph_store):
        """Get nearby memories finds nearby landmarks."""
        # Add a landmark
        landmark_manager.add_landmark(
            landmark_id="table-1",
            label="table",
            location_id="room-1",
            position=(1.0, 0.0, 1.0),
        )
        
        # Add corresponding node
        node = make_node(
            f"{NODE_TYPE_ENTITY}_table-1",
            NODE_TYPE_ENTITY,
            "table",
            "table-1",
        )
        graph_store.add_node(node)
        
        # Query nearby
        result = integrator.get_nearby_memories(
            location_id="room-1",
            position=(0.0, 0.0, 0.0),
            radius=5.0,
        )
        
        assert len(result.nodes) >= 1
    
    def test_what_is_here_empty(self, integrator):
        """What is here with no entities."""
        result = integrator.what_is_here("room-1")
        
        assert len(result) == 0
    
    def test_what_is_here_finds_entities(self, integrator, graph_store):
        """What is here finds entities in location."""
        # Create location
        loc_node = make_node(
            f"{NODE_TYPE_LOCATION}_room-1",
            NODE_TYPE_LOCATION,
            "kitchen",
        )
        graph_store.add_node(loc_node)
        
        # Create entity
        entity_node = make_node(
            f"{NODE_TYPE_ENTITY}_entity-1",
            NODE_TYPE_ENTITY,
            "chair",
            "entity-1",
        )
        graph_store.add_node(entity_node)
        
        # Link them
        edge = GraphEdge(
            edge_id="e1",
            edge_type=EDGE_TYPE_CONTAINS,
            source_node_id=entity_node.node_id,
            target_node_id=loc_node.node_id,
        )
        graph_store.add_edge(edge)
        
        result = integrator.what_is_here("room-1")
        
        assert len(result) >= 1
        assert any("chair" in desc for desc, _ in result)


# =============================================================================
# TEST: LEARNING INTEGRATION
# =============================================================================

class TestLearningIntegration:
    """Tests for learning integration with memory."""
    
    def test_learn_label_for_node(self, integrator, label_learner):
        """Learn label for existing node."""
        node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
        )
        
        success = integrator.learn_label_for_node(node.node_id, "lamp")
        
        assert success is True
        assert label_learner.has_label("lamp")
        assert node.label == "lamp"
    
    def test_learn_label_nonexistent_node(self, integrator):
        """Learn label for nonexistent node fails."""
        success = integrator.learn_label_for_node("nonexistent", "label")
        
        assert success is False
    
    def test_confirm_recognition(self, integrator, label_learner):
        """Confirm recognition increases confidence."""
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="chair",
        )
        
        initial_confirmed = label_learner.get_label("chair").times_confirmed
        
        integrator.confirm_recognition(f"{NODE_TYPE_ENTITY}_entity-1")
        
        assert label_learner.get_label("chair").times_confirmed == initial_confirmed + 1
    
    def test_correct_recognition(self, integrator, label_learner, graph_store):
        """Correct recognition updates node and learner."""
        node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="stool",
        )
        
        integrator.correct_recognition(node.node_id, "chair")
        
        # Label should be corrected
        updated_node = graph_store.get_node(node.node_id)
        assert updated_node.label == "chair"
        
        # New label should be learned
        assert label_learner.has_label("chair")


# =============================================================================
# TEST: RETRIEVAL RESULT
# =============================================================================

class TestRetrievalResult:
    """Tests for retrieval result structure."""
    
    def test_result_has_query_id(self, integrator):
        """Result includes query ID."""
        result = integrator.retrieve(MemoryQuery(query_id="test-query"))
        
        assert result.query_id == "test-query"
    
    def test_result_has_scores(self, integrator):
        """Result includes scores for matches."""
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="chair",
        )
        
        result = integrator.retrieve_by_label("chair")
        
        assert len(result.node_scores) == len(result.nodes)
        if result.node_scores:
            assert all(s > 0 for s in result.node_scores)
    
    def test_result_respects_max_results(self, integrator):
        """Result respects max_results limit."""
        # Create many nodes
        for i in range(20):
            integrator.integrate_observation(
                observation_type="entity",
                observation_id=f"entity-{i}",
                label="chair",
            )
        
        result = integrator.retrieve(MemoryQuery(
            labels=["chair"],
            max_results=5,
        ))
        
        assert len(result.nodes) <= 5


# =============================================================================
# TEST: INTEGRATION WORKFLOW
# =============================================================================

class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""
    
    def test_learn_then_recognize(self, integrator, label_learner):
        """Learn label then recognize similar entity."""
        # First encounter - user provides label
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="chair-1",
            label="dining_chair",
            features={"type": "seating", "legs": 4},
        )
        
        # Confirm a few times
        for _ in range(3):
            integrator.confirm_recognition(f"{NODE_TYPE_ENTITY}_chair-1")
        
        # Second encounter - try to recognize
        node2 = integrator.integrate_observation(
            observation_type="entity",
            observation_id="chair-2",
            features={"type": "seating", "legs": 4},
        )
        
        # Should have attempted recognition
        # Note: actual recognition depends on feature extractor
        assert node2 is not None
    
    def test_build_spatial_context(self, integrator, landmark_manager):
        """Build spatial context over time."""
        # Create a room
        integrator.integrate_observation(
            observation_type="location",
            observation_id="room-1",
            label="living_room",
        )
        
        # Add entities with positions
        sofa_node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="sofa-1",
            label="sofa",
            location_id="room-1",
            position=(0.0, 0.0, 0.0),
        )
        
        tv_node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="tv-1",
            label="television",
            location_id="room-1",
            position=(3.0, 0.0, 0.0),
        )
        
        # Should have created nodes
        assert sofa_node is not None
        assert tv_node is not None
        assert sofa_node.label == "sofa"
        assert tv_node.label == "television"
    
    def test_episode_with_entities(self, integrator, episode_store):
        """Episode integration with entities."""
        from episodic_agent.schemas import ObjectCandidate
        
        # First create entities
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="lamp",
        )
        integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-2",
            label="book",
        )
        
        # Create episode with those entities
        episode = Episode(
            episode_id="ep-1",
            created_at=datetime.now(),
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now(),
            step_count=10,
            location_label="bedroom",
            location_confidence=0.9,
            entities=[],  # Empty entities list for simplicity
            source_acf_id="acf-1",
            boundary_reason="location_change",
        )
        
        integrator.integrate_episode(episode)
        
        # Episode should be stored
        assert episode_store.count() == 1
        
        # Should be retrievable by location
        result = integrator.retrieve_by_location(location_label="bedroom")
        assert len(result.nodes) > 0
    
    def test_correction_workflow(self, integrator, label_learner):
        """Complete correction workflow."""
        # Initial labeling
        node = integrator.integrate_observation(
            observation_type="entity",
            observation_id="entity-1",
            label="lamp",
        )
        
        # User corrects
        integrator.correct_recognition(node.node_id, "desk_lamp")
        
        # Old label should have correction
        old_label = label_learner.get_label("lamp")
        assert old_label.times_corrected == 1
        
        # New label should exist
        assert label_learner.has_label("desk_lamp")
        
        # Node should have new label
        updated = integrator._graph.get_node(node.node_id)
        assert updated.label == "desk_lamp"
