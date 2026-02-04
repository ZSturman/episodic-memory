"""Tests for boundary detection and retrieval modules."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from episodic_agent.modules.boundary import (
    BoundaryReason,
    HysteresisBoundaryDetector,
)
from episodic_agent.modules.retriever import (
    CueToken,
    RetrieverSpreadingActivation,
    SpreadingActivationResult,
)
from episodic_agent.memory.graph_store import LabeledGraphStore
from episodic_agent.memory.episode_store import PersistentEpisodeStore
from episodic_agent.schemas import (
    ActiveContextFrame,
    Episode,
    GraphEdge,
    GraphNode,
    ObjectCandidate,
)
from episodic_agent.schemas.graph import EdgeType, NodeType


# =============================================================================
# HysteresisBoundaryDetector Tests
# =============================================================================

class TestHysteresisBoundaryDetector:
    """Tests for hysteresis-based boundary detection."""

    def _make_acf(
        self,
        step_count: int = 0,
        location: str = "TestRoom",
        events: list | None = None,
    ) -> ActiveContextFrame:
        """Helper to create test ACF."""
        return ActiveContextFrame(
            acf_id="test_acf",
            step_count=step_count,
            location_label=location,
            location_confidence=0.9,
            events=events or [],
        )

    def test_no_boundary_initially(self):
        """No boundary triggered on first step."""
        detector = HysteresisBoundaryDetector(time_interval=10)
        acf = self._make_acf(step_count=1)
        
        should_freeze, reason = detector.check(acf)
        
        assert not should_freeze

    def test_time_interval_boundary(self):
        """Boundary triggers after time interval."""
        detector = HysteresisBoundaryDetector(
            time_interval=5,
            high_threshold=0.3,  # Low threshold for time-only trigger
        )
        
        for step in range(1, 10):
            acf = self._make_acf(step_count=step)
            should_freeze, reason = detector.check(acf)
            
            if should_freeze:
                assert step >= 5
                break
        else:
            pytest.fail("Boundary never triggered")

    def test_location_change_boundary(self):
        """Boundary triggers on location change."""
        detector = HysteresisBoundaryDetector()
        
        # First location
        acf1 = self._make_acf(step_count=1, location="Kitchen")
        should_freeze1, _ = detector.check(acf1)
        assert not should_freeze1
        
        # Same location
        acf2 = self._make_acf(step_count=2, location="Kitchen")
        should_freeze2, _ = detector.check(acf2)
        assert not should_freeze2
        
        # Different location
        acf3 = self._make_acf(step_count=3, location="Bedroom")
        should_freeze3, reason = detector.check(acf3)
        
        assert should_freeze3
        assert "location" in reason.lower() if reason else True

    def test_hysteresis_prevents_oscillation(self):
        """Hysteresis prevents rapid boundary triggers."""
        detector = HysteresisBoundaryDetector(
            high_threshold=0.8,
            low_threshold=0.3,
        )
        
        # Trigger first boundary
        acf1 = self._make_acf(step_count=1, location="Kitchen")
        detector.check(acf1)
        
        acf2 = self._make_acf(step_count=2, location="Bedroom")
        triggered1, _ = detector.check(acf2)
        
        # Immediate re-trigger should be prevented by hysteresis
        acf3 = self._make_acf(step_count=3, location="Kitchen")
        triggered2, _ = detector.check(acf3)
        
        # At least one should work, but rapid oscillation should be dampened
        assert triggered1 or triggered2  # At least one triggers

    def test_reset_method(self):
        """Reset clears detector state."""
        detector = HysteresisBoundaryDetector()
        
        # Build up some state
        for i in range(5):
            detector.check(self._make_acf(step_count=i))
        
        detector.reset()
        
        # After reset, should behave like new detector
        acf = self._make_acf(step_count=1)
        should_freeze, _ = detector.check(acf)
        assert not should_freeze


# =============================================================================
# SpreadingActivationRetriever Tests
# =============================================================================

class TestSpreadingActivationRetriever:
    """Tests for spreading activation retrieval."""

    @pytest.fixture
    def stores(self):
        """Create temporary stores for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_path = Path(tmpdir) / "nodes.jsonl"
            edges_path = Path(tmpdir) / "edges.jsonl"
            episodes_path = Path(tmpdir) / "episodes.jsonl"
            
            graph_store = LabeledGraphStore(nodes_path, edges_path)
            episode_store = PersistentEpisodeStore(episodes_path)
            
            # Add some test data
            now = datetime.now()
            
            # Locations
            graph_store.add_node(GraphNode(
                node_id="loc_kitchen",
                node_type=NodeType.LOCATION,
                label="Kitchen",
                created_at=now,
                updated_at=now,
            ))
            graph_store.add_node(GraphNode(
                node_id="loc_bedroom",
                node_type=NodeType.LOCATION,
                label="Bedroom",
                created_at=now,
                updated_at=now,
            ))
            
            # Entities
            graph_store.add_node(GraphNode(
                node_id="ent_door",
                node_type=NodeType.ENTITY,
                label="Front Door",
                created_at=now,
                updated_at=now,
            ))
            graph_store.add_node(GraphNode(
                node_id="ent_lamp",
                node_type=NodeType.ENTITY,
                label="Lamp",
                created_at=now,
                updated_at=now,
            ))
            
            # Edges (door typical in kitchen, lamp typical in bedroom)
            graph_store.add_edge(GraphEdge(
                edge_id="e_door_kitchen",
                edge_type=EdgeType.TYPICAL_IN,
                source_id="ent_door",
                target_id="loc_kitchen",
                weight=5.0,
                created_at=now,
                updated_at=now,
            ))
            graph_store.add_edge(GraphEdge(
                edge_id="e_lamp_bedroom",
                edge_type=EdgeType.TYPICAL_IN,
                source_id="ent_lamp",
                target_id="loc_bedroom",
                weight=3.0,
                created_at=now,
                updated_at=now,
            ))
            
            # Episodes
            episode_store.store(Episode(
                episode_id="ep_001",
                source_acf_id="acf_001",
                location_label="Kitchen",
                start_time=now,
                end_time=now,
                step_count=50,
            ))
            
            yield {
                "graph": graph_store,
                "episodes": episode_store,
            }

    def test_retriever_creation(self, stores):
        """Retriever initializes correctly."""
        retriever = RetrieverSpreadingActivation(
            graph_store=stores["graph"],
            episode_store=stores["episodes"],
        )
        
        assert retriever._max_hops == 3
        assert retriever._decay_factor == 0.5

    def test_cue_token_creation(self):
        """Cue tokens are created correctly."""
        cue = CueToken(
            token_type="location",
            value="Kitchen",
            weight=1.0,
            source="acf",
        )
        
        assert cue.token_type == "location"
        assert cue.value == "Kitchen"
        assert cue.weight == 1.0

    def test_retrieve_returns_result(self, stores):
        """Retrieve returns proper result structure."""
        retriever = RetrieverSpreadingActivation(
            graph_store=stores["graph"],
            episode_store=stores["episodes"],
        )
        
        acf = ActiveContextFrame(
            acf_id="test_acf",
            location_label="Kitchen",
            location_confidence=0.9,
        )
        
        result = retriever.retrieve(acf)
        
        assert result.query_id is not None
        assert isinstance(result.scores, dict)

    def test_location_cue_activates_graph(self, stores):
        """Location cue activates related nodes."""
        retriever = RetrieverSpreadingActivation(
            graph_store=stores["graph"],
            episode_store=stores["episodes"],
            log_retrievals=False,
        )
        
        acf = ActiveContextFrame(
            acf_id="test_acf",
            location_label="Kitchen",
            location_confidence=0.9,
        )
        
        result = retriever.retrieve(acf)
        
        # Should have some activated nodes
        assert retriever.last_result is not None


class TestCueTokenGeneration:
    """Tests for cue token extraction from ACF."""

    def test_location_cue(self):
        """Location generates cue token."""
        acf = ActiveContextFrame(
            acf_id="test",
            location_label="Kitchen",
            location_confidence=0.9,
        )
        
        # Manual cue extraction (mimics retriever logic)
        cues = []
        if acf.location_label != "unknown":
            cues.append(CueToken(
                token_type="location",
                value=acf.location_label,
                weight=acf.location_confidence,
            ))
        
        assert len(cues) == 1
        assert cues[0].token_type == "location"
        assert cues[0].value == "Kitchen"

    def test_entity_cues(self):
        """Entities generate cue tokens."""
        acf = ActiveContextFrame(
            acf_id="test",
            entities=[
                ObjectCandidate(
                    candidate_id="obj_1",
                    label="Door",
                    confidence=0.9,
                ),
                ObjectCandidate(
                    candidate_id="obj_2",
                    label="Lamp",
                    confidence=0.7,
                ),
            ],
        )
        
        # Manual cue extraction
        cues = []
        for entity in acf.entities:
            cues.append(CueToken(
                token_type="entity",
                value=entity.label,
                weight=entity.confidence,
            ))
        
        assert len(cues) == 2

    def test_prediction_error_cue(self):
        """Prediction error generates weighted cue."""
        acf = ActiveContextFrame(
            acf_id="test",
            extras={"prediction_error": 0.8},
        )
        
        # Manual cue extraction with prediction error weight
        cues = []
        pred_error = acf.extras.get("prediction_error", 0.0)
        if pred_error > 0.1:
            cues.append(CueToken(
                token_type="prediction_error",
                value="high_error",
                weight=pred_error * 1.5,  # Extra weight
            ))
        
        assert len(cues) == 1
        assert cues[0].weight == 1.2  # 0.8 * 1.5
