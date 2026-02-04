"""Tests for core module interfaces and orchestrator."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from episodic_agent.core.interfaces import (
    ACFBuilder,
    BoundaryDetector,
    DialogManager,
    EntityResolver,
    EpisodeStore,
    EventResolver,
    GraphStore,
    LocationResolver,
    PerceptionModule,
    Retriever,
    SensorProvider,
)
from episodic_agent.core.orchestrator import AgentOrchestrator
from episodic_agent.schemas import (
    ActiveContextFrame,
    Episode,
    ObjectCandidate,
    Percept,
    RetrievalResult,
    SensorFrame,
)


# =============================================================================
# Mock Implementations for Testing
# =============================================================================

class MockSensorProvider(SensorProvider):
    """Mock sensor provider for testing."""
    
    def __init__(self, frame_count: int = 10):
        self._frame_count = frame_count
        self._current = 0
    
    def get_frame(self) -> SensorFrame:
        self._current += 1
        return SensorFrame(
            frame_id=self._current,
            sensor_type="mock",
            raw_data={"test": True},
        )
    
    def has_frames(self) -> bool:
        return self._current < self._frame_count


class MockPerception(PerceptionModule):
    """Mock perception module."""
    
    def process(self, frame: SensorFrame) -> Percept:
        return Percept(
            percept_id=f"percept_{frame.frame_id}",
            source_frame_id=frame.frame_id,
            scene_embedding=[0.1, 0.2, 0.3],
            candidates=[
                ObjectCandidate(
                    candidate_id=f"obj_{frame.frame_id}",
                    label="TestObject",
                    confidence=0.9,
                ),
            ],
        )


class MockACFBuilder(ACFBuilder):
    """Mock ACF builder."""
    
    def create_acf(self) -> ActiveContextFrame:
        return ActiveContextFrame(acf_id=f"acf_{datetime.now().timestamp()}")
    
    def update_acf(self, acf: ActiveContextFrame, percept: Percept) -> ActiveContextFrame:
        acf.current_percept = percept
        return acf


class MockLocationResolver(LocationResolver):
    """Mock location resolver."""
    
    def __init__(self, location: str = "TestRoom", confidence: float = 0.9):
        self._location = location
        self._confidence = confidence
    
    def resolve(self, percept: Percept, acf: ActiveContextFrame) -> tuple[str, float]:
        return self._location, self._confidence


class MockEntityResolver(EntityResolver):
    """Mock entity resolver."""
    
    def resolve(self, percept: Percept, acf: ActiveContextFrame) -> list[ObjectCandidate]:
        return percept.candidates


class MockEventResolver(EventResolver):
    """Mock event resolver."""
    
    def resolve(self, percept: Percept, acf: ActiveContextFrame) -> list[dict]:
        return []


class MockBoundaryDetector(BoundaryDetector):
    """Mock boundary detector."""
    
    def __init__(self, trigger_at_step: int = 5):
        self._trigger_at = trigger_at_step
    
    def check(self, acf: ActiveContextFrame) -> tuple[bool, str | None]:
        if acf.step_count >= self._trigger_at:
            return True, "test_boundary"
        return False, None


class MockRetriever(Retriever):
    """Mock retriever."""
    
    def retrieve(self, acf: ActiveContextFrame) -> RetrievalResult:
        return RetrievalResult(query_id=f"query_{acf.acf_id}")


class MockDialogManager(DialogManager):
    """Mock dialog manager."""
    
    def request_label(self, prompt: str, candidates: list[str] | None = None) -> str | None:
        return candidates[0] if candidates else "mock_label"
    
    def confirm(self, prompt: str, default: bool = True) -> bool:
        return default
    
    def resolve_conflict(self, prompt: str, options: list[str]) -> int:
        return 0  # Always pick first option
    
    def notify(self, message: str) -> None:
        pass


class MockEpisodeStore(EpisodeStore):
    """Mock episode store."""
    
    def __init__(self):
        self._episodes: dict[str, Episode] = {}
    
    def store(self, episode: Episode) -> None:
        self._episodes[episode.episode_id] = episode
    
    def get(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)
    
    def get_all(self) -> list[Episode]:
        return list(self._episodes.values())
    
    def count(self) -> int:
        return len(self._episodes)


class MockGraphStore(GraphStore):
    """Mock graph store."""
    
    def __init__(self):
        self._nodes = {}
        self._edges = {}
    
    def add_node(self, node) -> None:
        self._nodes[node.node_id] = node
    
    def get_node(self, node_id: str):
        return self._nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type):
        return [n for n in self._nodes.values() if n.node_type == node_type]
    
    def add_edge(self, edge) -> None:
        self._edges[edge.edge_id] = edge
    
    def get_edges(self, node_id: str):
        """Get all edges connected to a node (source or target)."""
        return [e for e in self._edges.values() 
                if e.source_node_id == node_id or e.target_node_id == node_id]
    
    def get_all_nodes(self):
        """Get all nodes in the graph."""
        return list(self._nodes.values())


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestAgentOrchestrator:
    """Tests for the AgentOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mock modules."""
        return AgentOrchestrator(
            sensor=MockSensorProvider(frame_count=20),
            perception=MockPerception(),
            acf_builder=MockACFBuilder(),
            location_resolver=MockLocationResolver(),
            entity_resolver=MockEntityResolver(),
            event_resolver=MockEventResolver(),
            retriever=MockRetriever(),
            boundary_detector=MockBoundaryDetector(trigger_at_step=5),
            dialog_manager=MockDialogManager(),
            episode_store=MockEpisodeStore(),
            graph_store=MockGraphStore(),
            run_id="test_run",
        )

    def test_initial_state(self, orchestrator):
        """Orchestrator starts with no ACF and step 0."""
        assert orchestrator.acf is None
        assert orchestrator.step_number == 0
        assert orchestrator.episode_count == 0

    def test_single_step(self, orchestrator):
        """Execute single step successfully."""
        result = orchestrator.step()
        
        assert result.step_number == 1
        assert result.location_label == "TestRoom"
        assert result.location_confidence == 0.9
        assert result.entity_count == 1

    def test_acf_created_on_first_step(self, orchestrator):
        """ACF is created on first step."""
        assert orchestrator.acf is None
        
        orchestrator.step()
        
        assert orchestrator.acf is not None
        assert orchestrator.acf.step_count == 1

    def test_step_count_increments(self, orchestrator):
        """Step count increments each step."""
        for i in range(5):
            result = orchestrator.step()
            assert result.step_number == i + 1

    def test_boundary_triggers_episode(self, orchestrator):
        """Boundary detection triggers episode freeze."""
        # Run until boundary
        for i in range(5):
            result = orchestrator.step()
            if i < 4:
                assert not result.boundary_triggered
        
        # Step 5 should trigger boundary
        assert result.boundary_triggered
        assert orchestrator.episode_count == 1

    def test_multiple_episodes(self, orchestrator):
        """Multiple episodes can be created."""
        # Run 15 steps (should trigger at 5 and 10)
        for i in range(15):
            orchestrator.step()
        
        # Should have multiple episodes
        assert orchestrator.episode_count >= 2

    def test_location_updates_acf(self, orchestrator):
        """Location is set in ACF."""
        orchestrator.step()
        
        assert orchestrator.acf.location_label == "TestRoom"
        assert orchestrator.acf.location_confidence == 0.9

    def test_entities_in_acf(self, orchestrator):
        """Entities are stored in ACF."""
        orchestrator.step()
        
        assert len(orchestrator.acf.entities) == 1
        assert orchestrator.acf.entities[0].label == "TestObject"


class TestOrchestratorStepOrder:
    """Tests verifying the immutable step order."""

    def test_step_order(self):
        """Verify modules are called in correct order."""
        call_order = []
        
        class TrackedSensor(MockSensorProvider):
            def get_frame(self):
                call_order.append("sensor")
                return super().get_frame()
        
        class TrackedPerception(MockPerception):
            def process(self, frame):
                call_order.append("perception")
                return super().process(frame)
        
        class TrackedACFBuilder(MockACFBuilder):
            def update_acf(self, acf, percept):
                call_order.append("acf_builder")
                return super().update_acf(acf, percept)
        
        class TrackedLocation(MockLocationResolver):
            def resolve(self, percept, acf):
                call_order.append("location")
                return super().resolve(percept, acf)
        
        class TrackedEntity(MockEntityResolver):
            def resolve(self, percept, acf):
                call_order.append("entity")
                return super().resolve(percept, acf)
        
        class TrackedEvent(MockEventResolver):
            def resolve(self, percept, acf):
                call_order.append("event")
                return super().resolve(percept, acf)
        
        class TrackedRetriever(MockRetriever):
            def retrieve(self, acf):
                call_order.append("retriever")
                return super().retrieve(acf)
        
        class TrackedBoundary(MockBoundaryDetector):
            def check(self, acf):
                call_order.append("boundary")
                return False, None
        
        orchestrator = AgentOrchestrator(
            sensor=TrackedSensor(),
            perception=TrackedPerception(),
            acf_builder=TrackedACFBuilder(),
            location_resolver=TrackedLocation(),
            entity_resolver=TrackedEntity(),
            event_resolver=TrackedEvent(),
            retriever=TrackedRetriever(),
            boundary_detector=TrackedBoundary(),
            dialog_manager=MockDialogManager(),
            episode_store=MockEpisodeStore(),
            graph_store=MockGraphStore(),
            run_id="test",
        )
        
        orchestrator.step()
        
        # Verify order
        expected = ["sensor", "perception", "acf_builder", "location", "entity", "event", "retriever", "boundary"]
        assert call_order == expected
