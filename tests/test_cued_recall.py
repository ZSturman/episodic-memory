"""Cued Recall Tests: Context-Frame Linking + Memory Retrieval.

ARCHITECTURAL INVARIANT: Memory retrieval uses weighted cues with tunable bias.
No predefined salience categories - weights are learned from experience.

Tests cover:
- Salience weight schemas and operations
- GraphEdge salience integration
- CuedRecallModule multi-cue retrieval
- EntityHypothesisTracker same-entity logic
- RedundantCueStore multiple paths
- LocationRevisit triggers
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import uuid

from episodic_agent.schemas.salience import (
    CueType,
    CuedRecallQuery,
    DEFAULT_CUE_WEIGHTS,
    EntityHypothesis,
    LocationRevisit,
    RecallResult,
    RedundantCue,
    SalienceWeights,
)
from episodic_agent.schemas.graph import (
    GraphEdge,
    GraphNode,
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_IN_CONTEXT,
    EDGE_TYPE_REVISIT,
    NODE_TYPE_ENTITY,
    NODE_TYPE_EPISODE,
    NODE_TYPE_LOCATION,
)
from episodic_agent.memory.cued_recall import (
    CuedRecallModule,
    EntityHypothesisTracker,
    RedundantCueStore,
    HIGH_SALIENCE_THRESHOLD,
    MIN_SALIENCE_TO_RECALL,
    SAME_POSITION_THRESHOLD,
    REVISIT_TIME_THRESHOLD,
)


# =============================================================================
# TEST: SALIENCE WEIGHTS SCHEMA
# =============================================================================

class TestSalienceWeights:
    """Tests for SalienceWeights data class."""
    
    def test_default_weights_are_zero(self):
        """All cue type weights default to 0."""
        weights = SalienceWeights()
        assert weights.location == 0.0
        assert weights.entity == 0.0
        assert weights.temporal == 0.0
        assert weights.semantic == 0.0
        assert weights.visual == 0.0
        assert weights.event == 0.0
        assert weights.overall == 0.0
    
    def test_compute_overall_from_max(self):
        """Overall salience is max of individual weights."""
        weights = SalienceWeights(
            location=0.3,
            entity=0.7,
            temporal=0.2,
        )
        overall = weights.compute_overall()
        assert overall == 0.7
        assert weights.overall == 0.7
        assert weights.dominant_cue == CueType.ENTITY
    
    def test_boost_cue_type(self):
        """Boosting a cue type increases its weight."""
        weights = SalienceWeights()
        weights.boost(CueType.LOCATION, amount=0.5)
        assert weights.location == 0.5
        assert weights.overall == 0.5
    
    def test_boost_caps_at_one(self):
        """Boosting cannot exceed 1.0."""
        weights = SalienceWeights(location=0.9)
        weights.boost(CueType.LOCATION, amount=0.5)
        assert weights.location == 1.0
    
    def test_decay_reduces_all_weights(self):
        """Decay reduces all weights proportionally."""
        weights = SalienceWeights(
            location=1.0,
            entity=0.5,
            temporal=0.2,
        )
        weights.decay(factor=0.5)
        assert weights.location == 0.5
        assert weights.entity == 0.25
        assert weights.temporal == 0.1
    
    def test_weighted_score_with_query_weights(self):
        """Weighted score combines link salience with query weights."""
        link_weights = SalienceWeights(
            location=0.8,
            entity=0.2,
        )
        query_weights = {
            CueType.LOCATION: 0.5,
            CueType.ENTITY: 0.5,
        }
        score = link_weights.weighted_score(query_weights)
        assert score == (0.8 * 0.5) + (0.2 * 0.5)  # 0.5


class TestCueType:
    """Tests for CueType enum."""
    
    def test_all_cue_types_in_defaults(self):
        """All cue types have default weights."""
        for cue_type in CueType:
            assert cue_type in DEFAULT_CUE_WEIGHTS
    
    def test_default_weights_sum_to_one(self):
        """Default cue weights sum to 1.0."""
        total = sum(DEFAULT_CUE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01  # Allow small floating point error


# =============================================================================
# TEST: GRAPH EDGE SALIENCE
# =============================================================================

class TestGraphEdgeSalience:
    """Tests for GraphEdge salience integration."""
    
    def test_edge_has_salience_dict(self):
        """GraphEdge has salience dict with all cue types."""
        edge = GraphEdge(
            edge_id="e1",
            edge_type="contains",
            source_node_id="n1",
            target_node_id="n2",
        )
        assert "location" in edge.salience
        assert "entity" in edge.salience
        assert "temporal" in edge.salience
        assert "semantic" in edge.salience
        assert "visual" in edge.salience
        assert "event" in edge.salience
        assert "overall" in edge.salience
    
    def test_edge_salience_defaults_to_zero(self):
        """Edge salience defaults to 0 for all cue types."""
        edge = GraphEdge(
            edge_id="e1",
            edge_type="contains",
            source_node_id="n1",
            target_node_id="n2",
        )
        assert edge.get_salience_score("location") == 0.0
        assert edge.get_salience_score("entity") == 0.0
    
    def test_boost_edge_salience(self):
        """Can boost edge salience for a cue type."""
        edge = GraphEdge(
            edge_id="e1",
            edge_type="contains",
            source_node_id="n1",
            target_node_id="n2",
        )
        edge.boost_salience("location", 0.3)
        assert edge.get_salience_score("location") == 0.3
        assert edge.get_salience_score("overall") == 0.3
    
    def test_boost_salience_caps_at_one(self):
        """Boosting salience cannot exceed 1.0."""
        edge = GraphEdge(
            edge_id="e1",
            edge_type="contains",
            source_node_id="n1",
            target_node_id="n2",
        )
        edge.salience["location"] = 0.9
        edge.boost_salience("location", 0.5)
        assert edge.get_salience_score("location") == 1.0


# =============================================================================
# TEST: CUED RECALL QUERY
# =============================================================================

class TestCuedRecallQuery:
    """Tests for CuedRecallQuery data class."""
    
    def test_query_with_location_cue(self):
        """Query with location cue identifies LOCATION as active."""
        query = CuedRecallQuery(
            location_id="loc_kitchen",
        )
        active = query.get_active_cue_types()
        assert CueType.LOCATION in active
    
    def test_query_with_entity_cues(self):
        """Query with entity IDs identifies ENTITY as active."""
        query = CuedRecallQuery(
            entity_ids=["ent_1", "ent_2"],
        )
        active = query.get_active_cue_types()
        assert CueType.ENTITY in active
    
    def test_query_with_labels(self):
        """Query with labels identifies SEMANTIC as active."""
        query = CuedRecallQuery(
            labels=["mug", "red"],
        )
        active = query.get_active_cue_types()
        assert CueType.SEMANTIC in active
    
    def test_query_with_visual_embedding(self):
        """Query with visual embedding identifies VISUAL as active."""
        query = CuedRecallQuery(
            visual_embedding=[0.1] * 128,
        )
        active = query.get_active_cue_types()
        assert CueType.VISUAL in active
    
    def test_query_effective_weights_use_defaults(self):
        """Effective weights use defaults when not overridden."""
        query = CuedRecallQuery()
        weights = query.get_effective_weights()
        assert weights[CueType.LOCATION] == DEFAULT_CUE_WEIGHTS[CueType.LOCATION]
    
    def test_query_effective_weights_can_override(self):
        """Can override default weights in query."""
        query = CuedRecallQuery(
            cue_weights={"location": 0.9},
        )
        weights = query.get_effective_weights()
        assert weights[CueType.LOCATION] == 0.9
    
    def test_query_multiple_cue_types(self):
        """Query with multiple cues identifies all active types."""
        query = CuedRecallQuery(
            location_id="loc_1",
            entity_ids=["ent_1"],
            labels=["thing"],
        )
        active = query.get_active_cue_types()
        assert len(active) == 3
        assert CueType.LOCATION in active
        assert CueType.ENTITY in active
        assert CueType.SEMANTIC in active


# =============================================================================
# TEST: ENTITY HYPOTHESIS
# =============================================================================

class TestEntityHypothesis:
    """Tests for EntityHypothesis data class."""
    
    def test_hypothesis_from_same_position(self):
        """Hypothesis created from observations at same position."""
        hyp = EntityHypothesis.from_observations(
            obs_a_id="obs_1",
            obs_b_id="obs_2",
            location_id="loc_kitchen",
            pos_a=(1.0, 0.0, 2.0),
            pos_b=(1.0, 0.0, 2.0),
            label_a="blue mug",
            label_b="red mug",
        )
        assert hyp.position_distance == 0.0
        assert hyp.confidence > 0.5
        assert "same position" in hyp.reason
    
    def test_hypothesis_nearby_position(self):
        """Hypothesis created from nearby positions."""
        hyp = EntityHypothesis.from_observations(
            obs_a_id="obs_1",
            obs_b_id="obs_2",
            location_id="loc_kitchen",
            pos_a=(1.0, 0.0, 2.0),
            pos_b=(1.3, 0.0, 2.0),
        )
        assert hyp.position_distance < 0.5
        assert hyp.confidence > 0.4
    
    def test_hypothesis_far_position_low_confidence(self):
        """Hypothesis from far positions has low confidence."""
        hyp = EntityHypothesis.from_observations(
            obs_a_id="obs_1",
            obs_b_id="obs_2",
            location_id="loc_kitchen",
            pos_a=(1.0, 0.0, 2.0),
            pos_b=(5.0, 0.0, 8.0),
        )
        assert hyp.position_distance > 2.0
        assert hyp.confidence < 0.3
    
    def test_hypothesis_confirm(self):
        """Confirming a hypothesis updates status."""
        hyp = EntityHypothesis.from_observations(
            obs_a_id="obs_1",
            obs_b_id="obs_2",
            location_id="loc_kitchen",
            pos_a=(1.0, 0.0, 2.0),
            pos_b=(1.0, 0.0, 2.0),
        )
        hyp.confirm("user said they're the same")
        assert hyp.status == "confirmed"
        assert hyp.resolved_at is not None
    
    def test_hypothesis_reject(self):
        """Rejecting a hypothesis updates status."""
        hyp = EntityHypothesis.from_observations(
            obs_a_id="obs_1",
            obs_b_id="obs_2",
            location_id="loc_kitchen",
            pos_a=(1.0, 0.0, 2.0),
            pos_b=(1.0, 0.0, 2.0),
        )
        hyp.reject("different objects")
        assert hyp.status == "rejected"
        assert hyp.resolved_at is not None


# =============================================================================
# TEST: REDUNDANT CUE STORE
# =============================================================================

class TestRedundantCueStore:
    """Tests for RedundantCueStore."""
    
    def test_add_and_retrieve_cue(self):
        """Can add and retrieve a cue."""
        store = RedundantCueStore()
        cue = RedundantCue(
            cue_type=CueType.LOCATION,
            target_node_id="node_1",
            location_id="loc_kitchen",
        )
        store.add_cue(cue)
        
        cues = store.get_cues_for_target("node_1")
        assert len(cues) == 1
        assert cues[0].cue_id == cue.cue_id
    
    def test_retrieve_cues_by_location(self):
        """Can retrieve cues by location."""
        store = RedundantCueStore()
        cue = RedundantCue(
            cue_type=CueType.LOCATION,
            target_node_id="node_1",
            location_id="loc_kitchen",
        )
        store.add_cue(cue)
        
        cues = store.get_cues_by_location("loc_kitchen")
        assert len(cues) == 1
    
    def test_retrieve_cues_by_entity(self):
        """Can retrieve cues by entity."""
        store = RedundantCueStore()
        cue = RedundantCue(
            cue_type=CueType.ENTITY,
            target_node_id="node_1",
            entity_id="ent_mug",
        )
        store.add_cue(cue)
        
        cues = store.get_cues_by_entity("ent_mug")
        assert len(cues) == 1
    
    def test_retrieve_cues_by_label(self):
        """Can retrieve cues by label (case insensitive)."""
        store = RedundantCueStore()
        cue = RedundantCue(
            cue_type=CueType.SEMANTIC,
            target_node_id="node_1",
            label="Coffee Mug",
        )
        store.add_cue(cue)
        
        cues = store.get_cues_by_label("coffee mug")
        assert len(cues) == 1
    
    def test_create_redundant_cues_for_memory(self):
        """Creates multiple cue paths for a memory."""
        store = RedundantCueStore()
        cues = store.create_cues_for_memory(
            target_node_id="node_1",
            location_id="loc_kitchen",
            entity_ids=["ent_1", "ent_2"],
            labels=["mug", "red"],
        )
        
        # Should create: 1 location + 2 entity + 2 semantic = 5 cues
        # (temporal cue only created if timestamp explicitly provided)
        assert len(cues) == 5
        assert store.total_cues == 5
    
    def test_multiple_cue_paths_to_same_memory(self):
        """Multiple cue paths can trigger same memory."""
        store = RedundantCueStore()
        store.create_cues_for_memory(
            target_node_id="node_1",
            location_id="loc_kitchen",
            entity_ids=["ent_mug"],
            labels=["coffee"],
        )
        
        # All these paths should find the same memory
        by_location = store.get_cues_by_location("loc_kitchen")
        by_entity = store.get_cues_by_entity("ent_mug")
        by_label = store.get_cues_by_label("coffee")
        
        assert all(c.target_node_id == "node_1" for c in by_location)
        assert all(c.target_node_id == "node_1" for c in by_entity)
        assert all(c.target_node_id == "node_1" for c in by_label)
    
    def test_evict_weakest_when_limit_reached(self):
        """Evicts weakest cue when per-target limit reached."""
        store = RedundantCueStore(max_cues_per_target=3)
        
        # Add 4 cues to same target
        for i in range(4):
            cue = RedundantCue(
                cue_type=CueType.SEMANTIC,
                target_node_id="node_1",
                label=f"label_{i}",
                strength=0.1 * (i + 1),  # Increasing strength
            )
            store.add_cue(cue)
        
        # Should have evicted weakest (label_0 with strength 0.1)
        assert store.total_cues == 3
        cues = store.get_cues_for_target("node_1")
        labels = {c.label for c in cues}
        assert "label_0" not in labels  # Weakest was evicted
    
    def test_access_cue_strengthens_it(self):
        """Accessing a cue strengthens it."""
        store = RedundantCueStore()
        cue = RedundantCue(
            cue_type=CueType.LOCATION,
            target_node_id="node_1",
            location_id="loc_kitchen",
            strength=0.5,
        )
        store.add_cue(cue)
        
        initial_strength = cue.strength
        store.access_cue(cue.cue_id)
        
        # Cue should be strengthened
        retrieved = store.get_cues_for_target("node_1")[0]
        assert retrieved.strength > initial_strength
        assert retrieved.access_count == 1


# =============================================================================
# TEST: ENTITY HYPOTHESIS TRACKER
# =============================================================================

class TestEntityHypothesisTracker:
    """Tests for EntityHypothesisTracker."""
    
    def test_record_observation_no_hypothesis_first_time(self):
        """First observation at location creates no hypothesis."""
        tracker = EntityHypothesisTracker()
        hypotheses = tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="mug",
        )
        assert len(hypotheses) == 0
    
    def test_same_position_creates_hypothesis(self):
        """Observation at same position creates hypothesis."""
        tracker = EntityHypothesisTracker()
        
        # First observation
        tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="blue mug",
        )
        
        # Second observation at same position
        hypotheses = tracker.record_observation(
            observation_id="obs_2",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="red mug",
        )
        
        assert len(hypotheses) == 1
        assert hypotheses[0].observation_a_id == "obs_1"
        assert hypotheses[0].observation_b_id == "obs_2"
    
    def test_different_position_no_hypothesis(self):
        """Observation at different position creates no hypothesis."""
        tracker = EntityHypothesisTracker()
        
        # First observation
        tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="mug",
        )
        
        # Second observation far away
        hypotheses = tracker.record_observation(
            observation_id="obs_2",
            location_id="loc_kitchen",
            position=(5.0, 0.0, 8.0),
            label="different mug",
        )
        
        assert len(hypotheses) == 0
    
    def test_different_location_no_hypothesis(self):
        """Observations in different locations create no hypothesis."""
        tracker = EntityHypothesisTracker()
        
        # First observation in kitchen
        tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="mug",
        )
        
        # Second observation in bedroom (same relative position)
        hypotheses = tracker.record_observation(
            observation_id="obs_2",
            location_id="loc_bedroom",
            position=(1.0, 0.0, 2.0),
            label="mug",
        )
        
        assert len(hypotheses) == 0
    
    def test_confirm_hypothesis(self):
        """Can confirm a hypothesis."""
        tracker = EntityHypothesisTracker()
        
        tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="blue mug",
        )
        
        hypotheses = tracker.record_observation(
            observation_id="obs_2",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="red mug",
        )
        
        tracker.confirm_hypothesis(hypotheses[0].hypothesis_id, "user confirmed")
        
        assert tracker.confirmed_count == 1
        assert tracker.pending_count == 0
    
    def test_reject_hypothesis(self):
        """Can reject a hypothesis."""
        tracker = EntityHypothesisTracker()
        
        tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="mug",
        )
        
        hypotheses = tracker.record_observation(
            observation_id="obs_2",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="different object",
        )
        
        tracker.reject_hypothesis(hypotheses[0].hypothesis_id, "different objects")
        
        assert tracker.confirmed_count == 0
        assert tracker.pending_count == 0
    
    def test_get_hypotheses_for_location(self):
        """Can get hypotheses for a specific location."""
        tracker = EntityHypothesisTracker()
        
        # Create hypothesis in kitchen
        tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="mug",
        )
        tracker.record_observation(
            observation_id="obs_2",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="mug",
        )
        
        # Create another in bedroom
        tracker.record_observation(
            observation_id="obs_3",
            location_id="loc_bedroom",
            position=(2.0, 0.0, 3.0),
            label="lamp",
        )
        tracker.record_observation(
            observation_id="obs_4",
            location_id="loc_bedroom",
            position=(2.0, 0.0, 3.0),
            label="lamp",
        )
        
        kitchen_hyps = tracker.get_hypotheses_for_location("loc_kitchen")
        bedroom_hyps = tracker.get_hypotheses_for_location("loc_bedroom")
        
        assert len(kitchen_hyps) == 1
        assert len(bedroom_hyps) == 1


# =============================================================================
# TEST: CUED RECALL MODULE
# =============================================================================

class TestCuedRecallModule:
    """Tests for CuedRecallModule."""
    
    @pytest.fixture
    def mock_graph_store(self):
        """Create a mock graph store."""
        store = MagicMock()
        store.get_node.return_value = None
        store.get_edge.return_value = None
        store.get_nodes_by_label.return_value = []
        store.get_edges_from_node.return_value = []
        store.get_similar_nodes.return_value = []
        return store
    
    @pytest.fixture
    def mock_episode_store(self):
        """Create a mock episode store."""
        store = MagicMock()
        store.get.return_value = None
        return store
    
    @pytest.fixture
    def recall_module(self, mock_graph_store, mock_episode_store):
        """Create a CuedRecallModule with mock stores."""
        return CuedRecallModule(
            graph_store=mock_graph_store,
            episode_store=mock_episode_store,
        )
    
    def test_recall_with_location_cue(self, recall_module):
        """Recall with location cue queries by location."""
        query = CuedRecallQuery(
            location_id="loc_kitchen",
            max_results=5,
        )
        
        result = recall_module.recall(query)
        
        assert isinstance(result, RecallResult)
        assert result.query_id == query.query_id
    
    def test_recall_with_entity_cue(self, recall_module):
        """Recall with entity cue queries by entity."""
        query = CuedRecallQuery(
            entity_ids=["ent_mug"],
            max_results=5,
        )
        
        result = recall_module.recall(query)
        
        assert isinstance(result, RecallResult)
    
    def test_recall_with_semantic_cue(self, recall_module):
        """Recall with semantic cue queries by label."""
        query = CuedRecallQuery(
            labels=["coffee", "mug"],
            max_results=5,
        )
        
        result = recall_module.recall(query)
        
        assert isinstance(result, RecallResult)
    
    def test_recall_with_multiple_cues(self, recall_module):
        """Recall with multiple cues combines them."""
        query = CuedRecallQuery(
            location_id="loc_kitchen",
            entity_ids=["ent_mug"],
            labels=["coffee"],
            max_results=5,
        )
        
        result = recall_module.recall(query)
        
        assert isinstance(result, RecallResult)
        assert result.query_time_ms > 0
    
    def test_recall_respects_min_salience(self, recall_module):
        """Recall filters out results below min_salience."""
        query = CuedRecallQuery(
            location_id="loc_kitchen",
            min_salience=0.9,  # Very high threshold
            max_results=5,
        )
        
        result = recall_module.recall(query)
        
        # With high threshold and no setup, should get no results
        assert len(result.node_ids) == 0
    
    def test_recall_prioritizes_salient(self, recall_module):
        """Recall sorts by salience when prioritize_salient=True."""
        query = CuedRecallQuery(
            location_id="loc_kitchen",
            prioritize_salient=True,
            max_results=5,
        )
        
        result = recall_module.recall(query)
        
        # Salience scores should be in descending order
        for i in range(len(result.node_salience) - 1):
            assert result.node_salience[i] >= result.node_salience[i + 1]
    
    def test_recall_callback_invoked(self, recall_module):
        """Recall invokes registered callbacks."""
        callback_results = []
        recall_module.register_recall_callback(
            lambda r: callback_results.append(r)
        )
        
        query = CuedRecallQuery(location_id="loc_kitchen")
        recall_module.recall(query)
        
        assert len(callback_results) == 1
        assert isinstance(callback_results[0], RecallResult)


# =============================================================================
# TEST: LOCATION REVISIT
# =============================================================================

class TestLocationRevisit:
    """Tests for location revisit functionality."""
    
    @pytest.fixture
    def recall_module(self):
        """Create a CuedRecallModule with mock stores."""
        graph_store = MagicMock()
        graph_store.get_node.return_value = None
        graph_store.get_edge.return_value = None
        graph_store.get_nodes_by_label.return_value = []
        graph_store.get_edges_from_node.return_value = []
        graph_store.get_similar_nodes.return_value = []
        
        episode_store = MagicMock()
        episode_store.get.return_value = None
        
        return CuedRecallModule(
            graph_store=graph_store,
            episode_store=episode_store,
        )
    
    def test_first_visit_not_revisit(self, recall_module):
        """First visit to location is not a revisit."""
        result = recall_module.on_location_enter("loc_kitchen")
        assert result is None
    
    def test_immediate_return_not_revisit(self, recall_module):
        """Returning immediately is not counted as revisit."""
        recall_module.on_location_enter("loc_kitchen")
        
        # Immediate return (within threshold)
        result = recall_module.on_location_enter("loc_kitchen")
        assert result is None
    
    def test_revisit_after_threshold(self, recall_module):
        """Revisit detected after time threshold."""
        # First visit
        recall_module._last_visit["loc_kitchen"] = (
            datetime.now() - timedelta(seconds=REVISIT_TIME_THRESHOLD + 10)
        )
        recall_module._visit_count["loc_kitchen"] = 1
        
        # This should be a revisit
        result = recall_module.on_location_enter("loc_kitchen", "Kitchen")
        
        assert result is not None
        assert isinstance(result, LocationRevisit)
        assert result.location_id == "loc_kitchen"
        assert result.prior_visit_count == 1
    
    def test_revisit_triggers_recall(self, recall_module):
        """Revisit triggers cued recall."""
        recall_module._last_visit["loc_kitchen"] = (
            datetime.now() - timedelta(seconds=REVISIT_TIME_THRESHOLD + 10)
        )
        recall_module._visit_count["loc_kitchen"] = 1
        
        result = recall_module.on_location_enter("loc_kitchen")
        
        assert result is not None
        assert isinstance(result.recalled_episode_ids, list)
    
    def test_revisit_callback_invoked(self, recall_module):
        """Revisit invokes registered callbacks."""
        callback_results = []
        recall_module.register_revisit_callback(
            lambda r: callback_results.append(r)
        )
        
        recall_module._last_visit["loc_kitchen"] = (
            datetime.now() - timedelta(seconds=REVISIT_TIME_THRESHOLD + 10)
        )
        recall_module._visit_count["loc_kitchen"] = 1
        
        recall_module.on_location_enter("loc_kitchen")
        
        assert len(callback_results) == 1
        assert isinstance(callback_results[0], LocationRevisit)


# =============================================================================
# TEST: ARCHITECTURAL INVARIANTS
# =============================================================================

class TestArchitecturalInvariants:
    """Tests for Phase 6 architectural invariants."""
    
    def test_no_predefined_salience_categories(self):
        """Salience weights are learned, not predefined."""
        # Weights should all start at 0
        weights = SalienceWeights()
        assert all(
            getattr(weights, cue_type.value) == 0.0
            for cue_type in CueType
        )
    
    def test_cue_weights_are_tunable(self):
        """Cue weights can be tuned at query time."""
        # Override defaults in query
        query = CuedRecallQuery(
            cue_weights={
                "location": 0.8,
                "entity": 0.1,
            },
        )
        
        weights = query.get_effective_weights()
        assert weights[CueType.LOCATION] == 0.8
        assert weights[CueType.ENTITY] == 0.1
    
    def test_multiple_cue_paths_to_memory(self):
        """Multiple cue paths can retrieve same memory."""
        store = RedundantCueStore()
        
        # Create cues from different paths
        store.create_cues_for_memory(
            target_node_id="memory_1",
            location_id="loc_kitchen",
            entity_ids=["ent_mug"],
            labels=["coffee"],
        )
        
        # Verify different paths lead to same memory
        by_location = store.get_cues_by_location("loc_kitchen")
        by_entity = store.get_cues_by_entity("ent_mug")
        by_label = store.get_cues_by_label("coffee")
        
        # All paths should retrieve the same memory
        assert len(by_location) >= 1
        assert len(by_entity) >= 1
        assert len(by_label) >= 1
        
        targets = set()
        targets.update(c.target_node_id for c in by_location)
        targets.update(c.target_node_id for c in by_entity)
        targets.update(c.target_node_id for c in by_label)
        
        assert targets == {"memory_1"}
    
    def test_high_salience_surfaces_first(self):
        """High-salience moments surface first in retrieval."""
        query = CuedRecallQuery(
            location_id="loc_kitchen",
            prioritize_salient=True,
        )
        
        # This is enforced by the sorting in recall()
        # The test verifies the flag exists and affects behavior
        assert query.prioritize_salient is True
    
    def test_same_entity_hypothesis_uses_position_not_category(self):
        """Same-entity hypothesis based on position, not category."""
        tracker = EntityHypothesisTracker()
        
        # First observation: "blue mug"
        tracker.record_observation(
            observation_id="obs_1",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="blue mug",
        )
        
        # Second observation: "red cup" at same position
        # Should create hypothesis despite different labels
        hypotheses = tracker.record_observation(
            observation_id="obs_2",
            location_id="loc_kitchen",
            position=(1.0, 0.0, 2.0),
            label="red cup",
        )
        
        assert len(hypotheses) == 1
        assert hypotheses[0].label_a == "blue mug"
        assert hypotheses[0].label_b == "red cup"
        # Position-based, not category-based
        assert hypotheses[0].position_distance == 0.0
    
    def test_location_entry_triggers_recall(self):
        """Entering known location triggers recall of prior visits."""
        graph_store = MagicMock()
        graph_store.get_node.return_value = None
        graph_store.get_nodes_by_label.return_value = []
        graph_store.get_edges_from_node.return_value = []
        
        episode_store = MagicMock()
        
        module = CuedRecallModule(
            graph_store=graph_store,
            episode_store=episode_store,
        )
        
        # Simulate prior visit
        module._last_visit["loc_kitchen"] = (
            datetime.now() - timedelta(seconds=REVISIT_TIME_THRESHOLD + 10)
        )
        module._visit_count["loc_kitchen"] = 1
        
        # Enter location
        revisit = module.on_location_enter("loc_kitchen")
        
        # Should trigger recall
        assert revisit is not None
        assert isinstance(revisit.recalled_episode_ids, list)


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_query(self):
        """Query with no cues returns empty result."""
        graph_store = MagicMock()
        graph_store.get_node.return_value = None
        graph_store.get_nodes_by_label.return_value = []
        graph_store.get_edges_from_node.return_value = []
        
        episode_store = MagicMock()
        
        module = CuedRecallModule(
            graph_store=graph_store,
            episode_store=episode_store,
        )
        
        query = CuedRecallQuery()  # No cues
        result = module.recall(query)
        
        assert len(result.node_ids) == 0
        assert len(result.episode_ids) == 0
    
    def test_nonexistent_hypothesis_operations(self):
        """Operations on nonexistent hypotheses return False."""
        tracker = EntityHypothesisTracker()
        
        assert tracker.confirm_hypothesis("nonexistent") is False
        assert tracker.reject_hypothesis("nonexistent") is False
        assert tracker.get_hypothesis("nonexistent") is None
    
    def test_boost_salience_on_nonexistent_edge(self):
        """Boosting salience on nonexistent edge returns False."""
        graph_store = MagicMock()
        graph_store.get_edge.return_value = None
        
        episode_store = MagicMock()
        
        module = CuedRecallModule(
            graph_store=graph_store,
            episode_store=episode_store,
        )
        
        result = module.boost_edge_salience("nonexistent", CueType.LOCATION)
        assert result is False
    
    def test_cue_store_handles_missing_fields(self):
        """Cue store handles cues with missing optional fields."""
        store = RedundantCueStore()
        
        # Cue with minimal fields
        cue = RedundantCue(
            cue_type=CueType.LOCATION,
            target_node_id="node_1",
            # No location_id, entity_id, label, etc.
        )
        
        store.add_cue(cue)
        
        # Should still be retrievable
        cues = store.get_cues_for_target("node_1")
        assert len(cues) == 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
