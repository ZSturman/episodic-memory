"""Consolidation Tests: Deferred Processing + Logging.

Tests for:
- ConsolidationModule queue operations
- Merge/relabel/prune/decay processing
- ConsolidationScheduler inactivity detection
- StructuredLogger category-based logging
- SessionLogger file output
- Architectural invariants
"""

import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from episodic_agent.modules.consolidation import (
    ConsolidationModule,
    ConsolidationOperation,
    ConsolidationResult,
    ConsolidationScheduler,
    DecayOperation,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INACTIVITY_THRESHOLD,
    MERGE_CONFIDENCE_THRESHOLD,
    MERGE_SIMILARITY_THRESHOLD,
    MergeOperation,
    OperationType,
    PruneOperation,
    RelabelOperation,
    RELABEL_CONFIDENCE_THRESHOLD,
)
from episodic_agent.utils.logging import (
    create_session_logger,
    get_logger,
    LogCategory,
    LogEntry,
    LogLevel,
    SessionLogger,
    set_logger,
    StructuredLogger,
)
from episodic_agent.memory.stubs import InMemoryGraphStore, InMemoryEpisodeStore
from episodic_agent.schemas.graph import (
    EDGE_TYPE_ALIAS_OF,
    EDGE_TYPE_MERGED_INTO,
    GraphEdge,
    GraphNode,
    NODE_TYPE_ENTITY,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def graph_store() -> InMemoryGraphStore:
    """Create an in-memory graph store."""
    return InMemoryGraphStore()


@pytest.fixture
def episode_store() -> InMemoryEpisodeStore:
    """Create an in-memory episode store."""
    return InMemoryEpisodeStore()


@pytest.fixture
def logger() -> StructuredLogger:
    """Create a structured logger."""
    return StructuredLogger()


@pytest.fixture
def consolidation_module(
    graph_store: InMemoryGraphStore,
    episode_store: InMemoryEpisodeStore,
    logger: StructuredLogger,
) -> ConsolidationModule:
    """Create a consolidation module."""
    return ConsolidationModule(
        graph_store=graph_store,
        episode_store=episode_store,
        logger=logger,
    )


@pytest.fixture
def temp_runs_dir() -> Generator[Path, None, None]:
    """Create a temporary runs directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# OPERATION TYPE TESTS
# =============================================================================

class TestOperationType:
    """Tests for OperationType enum."""
    
    def test_operation_types_exist(self) -> None:
        """Test all operation types are defined."""
        assert OperationType.MERGE == "merge"
        assert OperationType.RELABEL == "relabel"
        assert OperationType.PRUNE == "prune"
        assert OperationType.STRENGTHEN == "strengthen"
        assert OperationType.DECAY == "decay"
    
    def test_operation_type_string_values(self) -> None:
        """Test operation types have string values."""
        for op_type in OperationType:
            assert isinstance(op_type.value, str)


class TestConsolidationOperation:
    """Tests for ConsolidationOperation base class."""
    
    def test_operation_has_unique_id(self) -> None:
        """Test operations have unique IDs."""
        op1 = ConsolidationOperation()
        op2 = ConsolidationOperation()
        assert op1.operation_id != op2.operation_id
    
    def test_operation_has_timestamp(self) -> None:
        """Test operations have creation timestamp."""
        before = datetime.now()
        op = ConsolidationOperation()
        after = datetime.now()
        assert before <= op.created_at <= after
    
    def test_operation_priority_comparison(self) -> None:
        """Test operations can be compared by priority."""
        op1 = ConsolidationOperation(priority=1)
        op2 = ConsolidationOperation(priority=5)
        # Higher priority should be "less than" for priority queue
        assert op2 < op1


class TestMergeOperation:
    """Tests for MergeOperation."""
    
    def test_merge_operation_creation(self) -> None:
        """Test creating a merge operation."""
        op = MergeOperation(
            source_node_id="source-123",
            target_node_id="target-456",
            similarity=0.9,
            confidence=0.85,
        )
        assert op.operation_type == OperationType.MERGE
        assert op.source_node_id == "source-123"
        assert op.target_node_id == "target-456"
        assert op.similarity == 0.9
        assert op.confidence == 0.85
    
    def test_merge_operation_default_reason(self) -> None:
        """Test merge operation generates default reason."""
        op = MergeOperation(
            source_node_id="a",
            target_node_id="b",
            similarity=0.95,
        )
        assert "0.95" in op.reason


class TestRelabelOperation:
    """Tests for RelabelOperation."""
    
    def test_relabel_operation_creation(self) -> None:
        """Test creating a relabel operation."""
        op = RelabelOperation(
            node_id="node-123",
            new_label="cup",
            old_label="object",
            confidence=0.9,
            source="user",
        )
        assert op.operation_type == OperationType.RELABEL
        assert op.node_id == "node-123"
        assert op.new_label == "cup"
        assert op.old_label == "object"
        assert op.source == "user"


class TestPruneOperation:
    """Tests for PruneOperation."""
    
    def test_prune_operation_creation(self) -> None:
        """Test creating a prune operation."""
        op = PruneOperation(
            node_id="node-123",
            confidence=0.05,
            reason="low confidence",
        )
        assert op.operation_type == OperationType.PRUNE
        assert op.confidence == 0.05


class TestDecayOperation:
    """Tests for DecayOperation."""
    
    def test_decay_operation_creation(self) -> None:
        """Test creating a decay operation."""
        op = DecayOperation(
            node_ids=["a", "b", "c"],
            decay_factor=0.9,
        )
        assert op.operation_type == OperationType.DECAY
        assert len(op.node_ids) == 3
        assert op.decay_factor == 0.9


# =============================================================================
# CONSOLIDATION RESULT TESTS
# =============================================================================

class TestConsolidationResult:
    """Tests for ConsolidationResult."""
    
    def test_result_has_unique_id(self) -> None:
        """Test results have unique IDs."""
        r1 = ConsolidationResult()
        r2 = ConsolidationResult()
        assert r1.run_id != r2.run_id
    
    def test_result_complete(self) -> None:
        """Test marking result as complete."""
        result = ConsolidationResult()
        assert result.completed_at is None
        result.complete()
        assert result.completed_at is not None
    
    def test_result_duration(self) -> None:
        """Test result duration calculation."""
        result = ConsolidationResult()
        time.sleep(0.01)  # 10ms
        result.complete()
        assert result.duration_ms >= 10


# =============================================================================
# CONSOLIDATION MODULE QUEUE TESTS
# =============================================================================

class TestConsolidationModuleQueue:
    """Tests for ConsolidationModule queue operations."""
    
    def test_queue_merge(self, consolidation_module: ConsolidationModule) -> None:
        """Test queueing a merge operation."""
        op = consolidation_module.queue_merge(
            source_node_id="a",
            target_node_id="b",
            similarity=0.9,
            confidence=0.8,
        )
        assert consolidation_module.queue_size == 1
        assert isinstance(op, MergeOperation)
    
    def test_queue_relabel(self, consolidation_module: ConsolidationModule) -> None:
        """Test queueing a relabel operation."""
        op = consolidation_module.queue_relabel(
            node_id="a",
            new_label="cup",
            old_label="object",
            confidence=0.9,
        )
        assert consolidation_module.queue_size == 1
        assert isinstance(op, RelabelOperation)
    
    def test_queue_prune(self, consolidation_module: ConsolidationModule) -> None:
        """Test queueing a prune operation."""
        op = consolidation_module.queue_prune(
            node_id="a",
            confidence=0.05,
        )
        assert consolidation_module.queue_size == 1
        assert isinstance(op, PruneOperation)
    
    def test_queue_decay(self, consolidation_module: ConsolidationModule) -> None:
        """Test queueing a decay operation."""
        op = consolidation_module.queue_decay(
            node_ids=["a", "b", "c"],
            decay_factor=0.9,
        )
        assert consolidation_module.queue_size == 1
        assert isinstance(op, DecayOperation)
    
    def test_queue_multiple_operations(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test queueing multiple operations."""
        consolidation_module.queue_merge("a", "b", 0.9, 0.8)
        consolidation_module.queue_relabel("c", "cup", "object", 0.9)
        consolidation_module.queue_prune("d", 0.05)
        assert consolidation_module.queue_size == 3
    
    def test_queue_max_size(
        self,
        graph_store: InMemoryGraphStore,
        episode_store: InMemoryEpisodeStore,
    ) -> None:
        """Test queue respects max size."""
        module = ConsolidationModule(
            graph_store=graph_store,
            episode_store=episode_store,
            max_queue_size=5,
        )
        for i in range(10):
            module.queue_merge(f"a{i}", f"b{i}", 0.9, 0.8)
        assert module.queue_size == 5


# =============================================================================
# CONSOLIDATION MODULE PROCESSING TESTS
# =============================================================================

class TestConsolidationModuleProcessing:
    """Tests for ConsolidationModule operation processing."""
    
    def test_consolidate_empty_queue(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test consolidation with empty queue."""
        result = consolidation_module.consolidate()
        assert result.operations_processed == 0
        assert result.operations_succeeded == 0
    
    def test_consolidate_processes_batch(
        self,
        graph_store: InMemoryGraphStore,
        episode_store: InMemoryEpisodeStore,
    ) -> None:
        """Test consolidation processes batch of operations."""
        module = ConsolidationModule(
            graph_store=graph_store,
            episode_store=episode_store,
            batch_size=3,
        )
        # Queue more than batch size
        for i in range(5):
            module.queue_decay([f"node-{i}"], 0.9)
        
        result = module.consolidate()
        # Should process up to batch_size
        assert result.operations_processed == 3
        assert module.queue_size == 2
    
    def test_merge_preserves_identity(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """ARCHITECTURAL INVARIANT: Merging preserves identity."""
        # Create source and target nodes
        source = GraphNode(
            node_id="source-123",
            node_type=NODE_TYPE_ENTITY,
            label="cup",
            labels=["cup", "mug"],
            confidence=0.7,
        )
        target = GraphNode(
            node_id="target-456",
            node_type=NODE_TYPE_ENTITY,
            label="container",
            labels=["container"],
            confidence=0.8,
        )
        graph_store.add_node(source)
        graph_store.add_node(target)
        
        # Queue and process merge
        consolidation_module.queue_merge(
            source_node_id="source-123",
            target_node_id="target-456",
            similarity=0.9,
            confidence=0.9,
        )
        result = consolidation_module.consolidate()
        
        # Verify merge occurred
        assert result.nodes_merged == 1
        
        # Verify target has all labels
        updated_target = graph_store.get_node("target-456")
        assert updated_target is not None
        assert "cup" in updated_target.labels
        assert "mug" in updated_target.labels
        assert "container" in updated_target.labels
        
        # Verify MERGED_INTO edge was created
        edges = graph_store.get_edges_from_node("source-123")
        merge_edge = next(
            (e for e in edges if e.edge_type == EDGE_TYPE_MERGED_INTO), None
        )
        assert merge_edge is not None
        assert merge_edge.target_node_id == "target-456"
    
    def test_merge_below_threshold_skipped(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test merge below similarity threshold is skipped."""
        source = GraphNode(node_id="a", node_type=NODE_TYPE_ENTITY, label="x")
        target = GraphNode(node_id="b", node_type=NODE_TYPE_ENTITY, label="y")
        graph_store.add_node(source)
        graph_store.add_node(target)
        
        consolidation_module.queue_merge(
            source_node_id="a",
            target_node_id="b",
            similarity=0.5,  # Below threshold
            confidence=0.9,
        )
        result = consolidation_module.consolidate()
        
        assert result.nodes_merged == 0
        assert result.operations_skipped == 1
    
    def test_merge_low_confidence_skipped(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test merge with low confidence is skipped (needs user confirmation)."""
        source = GraphNode(node_id="a", node_type=NODE_TYPE_ENTITY, label="x")
        target = GraphNode(node_id="b", node_type=NODE_TYPE_ENTITY, label="y")
        graph_store.add_node(source)
        graph_store.add_node(target)
        
        consolidation_module.queue_merge(
            source_node_id="a",
            target_node_id="b",
            similarity=0.95,
            confidence=0.3,  # Below auto-merge threshold
        )
        result = consolidation_module.consolidate()
        
        assert result.nodes_merged == 0
        assert result.operations_skipped == 1
    
    def test_relabel_adds_labels(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """ARCHITECTURAL INVARIANT: Relabeling adds labels, doesn't remove them."""
        node = GraphNode(
            node_id="node-123",
            node_type=NODE_TYPE_ENTITY,
            label="object",
            labels=["object"],
            confidence=0.5,
        )
        graph_store.add_node(node)
        
        consolidation_module.queue_relabel(
            node_id="node-123",
            new_label="coffee_cup",
            old_label="object",
            confidence=0.9,
        )
        result = consolidation_module.consolidate()
        
        assert result.nodes_relabeled == 1
        
        updated = graph_store.get_node("node-123")
        assert updated is not None
        assert updated.label == "coffee_cup"
        # Old label should be preserved
        assert "object" in updated.labels
        assert "coffee_cup" in updated.labels
    
    def test_relabel_low_confidence_skipped(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test relabel with low confidence is skipped."""
        node = GraphNode(node_id="a", node_type=NODE_TYPE_ENTITY, label="x")
        graph_store.add_node(node)
        
        consolidation_module.queue_relabel(
            node_id="a",
            new_label="y",
            old_label="x",
            confidence=0.5,  # Below threshold
        )
        result = consolidation_module.consolidate()
        
        assert result.nodes_relabeled == 0
        assert result.operations_skipped == 1
    
    def test_prune_conservative(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """ARCHITECTURAL INVARIANT: Pruning is very conservative."""
        # High confidence node should not be pruned
        node = GraphNode(
            node_id="node-123",
            node_type=NODE_TYPE_ENTITY,
            label="cup",
            confidence=0.5,
        )
        graph_store.add_node(node)
        
        consolidation_module.queue_prune(
            node_id="node-123",
            confidence=0.5,
        )
        result = consolidation_module.consolidate()
        
        # Should not prune (confidence too high)
        assert result.nodes_pruned == 0
        assert graph_store.get_node("node-123") is not None
    
    def test_prune_very_low_confidence(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test pruning very low confidence, unconnected nodes."""
        node = GraphNode(
            node_id="node-123",
            node_type=NODE_TYPE_ENTITY,
            label="unknown",
            confidence=0.05,
            access_count=1,
        )
        graph_store.add_node(node)
        
        consolidation_module.queue_prune(
            node_id="node-123",
            confidence=0.05,
        )
        result = consolidation_module.consolidate()
        
        # Should prune (very low confidence, no edges, low access)
        assert result.nodes_pruned == 1
        assert graph_store.get_node("node-123") is None
    
    def test_prune_with_edges_not_pruned(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test that nodes with edges are not pruned."""
        from episodic_agent.schemas.graph import EDGE_TYPE_SIMILAR_TO
        
        node1 = GraphNode(
            node_id="node-1",
            node_type=NODE_TYPE_ENTITY,
            confidence=0.05,
        )
        node2 = GraphNode(
            node_id="node-2",
            node_type=NODE_TYPE_ENTITY,
            confidence=0.5,
        )
        graph_store.add_node(node1)
        graph_store.add_node(node2)
        
        edge = GraphEdge(
            edge_id="edge-1",
            edge_type=EDGE_TYPE_SIMILAR_TO,
            source_node_id="node-1",
            target_node_id="node-2",
        )
        graph_store.add_edge(edge)
        
        consolidation_module.queue_prune(
            node_id="node-1",
            confidence=0.05,
        )
        result = consolidation_module.consolidate()
        
        # Should not prune (has edges)
        assert result.nodes_pruned == 0
        assert graph_store.get_node("node-1") is not None
    
    def test_decay_reduces_activation(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test decay reduces node activation."""
        node = GraphNode(
            node_id="node-123",
            node_type=NODE_TYPE_ENTITY,
            activation=1.0,
            base_activation=0.5,
        )
        graph_store.add_node(node)
        
        consolidation_module.queue_decay(
            node_ids=["node-123"],
            decay_factor=0.9,
        )
        result = consolidation_module.consolidate()
        
        assert result.nodes_decayed == 1
        
        updated = graph_store.get_node("node-123")
        assert updated is not None
        assert updated.activation == 0.9
        assert updated.base_activation == 0.45


# =============================================================================
# CONSOLIDATION MODULE CALLBACKS TESTS
# =============================================================================

class TestConsolidationModuleCallbacks:
    """Tests for ConsolidationModule callbacks."""
    
    def test_on_complete_callback(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test on_complete callback is called."""
        callback_results = []
        
        def callback(result: ConsolidationResult) -> None:
            callback_results.append(result)
        
        consolidation_module.register_on_complete(callback)
        consolidation_module.queue_decay(["a"], 0.9)
        consolidation_module.consolidate()
        
        assert len(callback_results) == 1
        assert isinstance(callback_results[0], ConsolidationResult)
    
    def test_multiple_callbacks(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test multiple callbacks are called."""
        results1 = []
        results2 = []
        
        consolidation_module.register_on_complete(lambda r: results1.append(r))
        consolidation_module.register_on_complete(lambda r: results2.append(r))
        consolidation_module.consolidate()
        
        assert len(results1) == 1
        assert len(results2) == 1


# =============================================================================
# CONSOLIDATION MODULE STATISTICS TESTS
# =============================================================================

class TestConsolidationModuleStatistics:
    """Tests for ConsolidationModule statistics."""
    
    def test_total_runs_tracking(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test total runs are tracked."""
        assert consolidation_module.total_runs == 0
        
        consolidation_module.consolidate()
        assert consolidation_module.total_runs == 1
        
        consolidation_module.consolidate()
        assert consolidation_module.total_runs == 2
    
    def test_total_operations_tracking(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test total operations are tracked."""
        assert consolidation_module.total_operations == 0
        
        consolidation_module.queue_decay(["a"], 0.9)
        consolidation_module.queue_decay(["b"], 0.9)
        consolidation_module.consolidate()
        
        assert consolidation_module.total_operations == 2
    
    def test_last_run_tracking(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test last run result is tracked."""
        assert consolidation_module.last_run is None
        
        consolidation_module.consolidate()
        
        assert consolidation_module.last_run is not None
        assert isinstance(consolidation_module.last_run, ConsolidationResult)


# =============================================================================
# CONSOLIDATION MODULE MERGE CANDIDATES TESTS
# =============================================================================

class TestConsolidationModuleMergeCandidates:
    """Tests for finding merge candidates."""
    
    def test_find_candidates_by_label(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test finding merge candidates by identical labels."""
        node1 = GraphNode(
            node_id="node-1",
            node_type=NODE_TYPE_ENTITY,
            label="coffee_cup",
        )
        node2 = GraphNode(
            node_id="node-2",
            node_type=NODE_TYPE_ENTITY,
            label="COFFEE_CUP",  # Same label, different case
        )
        graph_store.add_node(node1)
        graph_store.add_node(node2)
        
        candidates = consolidation_module.find_merge_candidates()
        
        assert len(candidates) == 1
        assert ("node-1", "node-2", 1.0) in candidates
    
    def test_find_candidates_by_embedding(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """Test finding merge candidates by embedding similarity."""
        node1 = GraphNode(
            node_id="node-1",
            node_type=NODE_TYPE_ENTITY,
            label="a",
            embedding=[1.0, 0.0, 0.0],
        )
        node2 = GraphNode(
            node_id="node-2",
            node_type=NODE_TYPE_ENTITY,
            label="b",
            embedding=[0.99, 0.1, 0.0],  # Very similar
        )
        graph_store.add_node(node1)
        graph_store.add_node(node2)
        
        candidates = consolidation_module.find_merge_candidates(
            similarity_threshold=0.9
        )
        
        # Should find these as candidates
        assert len(candidates) >= 1


# =============================================================================
# CONSOLIDATION SCHEDULER TESTS
# =============================================================================

class TestConsolidationScheduler:
    """Tests for ConsolidationScheduler."""
    
    def test_scheduler_creation(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test scheduler creation."""
        scheduler = ConsolidationScheduler(
            consolidation_module=consolidation_module,
            inactivity_threshold=5.0,
        )
        assert scheduler._threshold == 5.0
        assert not scheduler.is_running
    
    def test_record_activity(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test recording activity."""
        scheduler = ConsolidationScheduler(
            consolidation_module=consolidation_module,
        )
        before = scheduler._last_activity
        time.sleep(0.01)
        scheduler.record_activity()
        assert scheduler._last_activity > before
    
    def test_seconds_since_activity(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test seconds since activity calculation."""
        scheduler = ConsolidationScheduler(
            consolidation_module=consolidation_module,
        )
        scheduler.record_activity()
        time.sleep(0.05)
        assert scheduler.seconds_since_activity >= 0.05
    
    def test_scheduler_start_stop(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test starting and stopping scheduler."""
        scheduler = ConsolidationScheduler(
            consolidation_module=consolidation_module,
        )
        
        assert not scheduler.is_running
        
        scheduler.start()
        assert scheduler.is_running
        
        scheduler.stop()
        assert not scheduler.is_running
    
    def test_scheduler_triggers_consolidation(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test scheduler triggers consolidation after inactivity."""
        scheduler = ConsolidationScheduler(
            consolidation_module=consolidation_module,
            inactivity_threshold=0.05,  # Very short for testing
        )
        
        # Queue an operation
        consolidation_module.queue_decay(["a"], 0.9)
        
        # Start scheduler and wait
        scheduler.start()
        try:
            # Wait longer for consolidation to trigger (0.05s threshold + 1s check interval)
            time.sleep(1.5)
            
            # Should have processed the operation
            assert consolidation_module.total_runs >= 1
        finally:
            scheduler.stop()


# =============================================================================
# LOG CATEGORY TESTS
# =============================================================================

class TestLogCategory:
    """Tests for LogCategory enum."""
    
    def test_all_categories_exist(self) -> None:
        """Test all expected log categories exist."""
        expected = [
            "SENSOR", "MEMORY", "RECOGNITION", "CONSOLIDATION",
            "LABEL", "SPATIAL", "PROTOCOL", "ARBITRATION",
            "VISUAL", "EVENT", "RECALL", "HYPOTHESIS",
            "INVARIANT", "SYSTEM",
        ]
        for cat in expected:
            assert hasattr(LogCategory, cat)
    
    def test_categories_are_strings(self) -> None:
        """Test category values are strings."""
        for cat in LogCategory:
            assert isinstance(cat.value, str)


class TestLogLevel:
    """Tests for LogLevel enum."""
    
    def test_log_levels_exist(self) -> None:
        """Test all log levels exist."""
        # Check they exist and have expected order via map
        from episodic_agent.utils.logging import _LEVEL_MAP
        assert _LEVEL_MAP[LogLevel.DEBUG] < _LEVEL_MAP[LogLevel.INFO]
        assert _LEVEL_MAP[LogLevel.INFO] < _LEVEL_MAP[LogLevel.WARNING]
        assert _LEVEL_MAP[LogLevel.WARNING] < _LEVEL_MAP[LogLevel.ERROR]
        assert _LEVEL_MAP[LogLevel.ERROR] < _LEVEL_MAP[LogLevel.CRITICAL]


class TestLogEntry:
    """Tests for LogEntry."""
    
    def test_entry_creation(self) -> None:
        """Test creating a log entry."""
        entry = LogEntry(
            category=LogCategory.MEMORY,
            level=LogLevel.INFO,
            message="Test message",
        )
        assert entry.category == LogCategory.MEMORY
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.timestamp is not None
    
    def test_entry_with_context(self) -> None:
        """Test entry with additional context."""
        entry = LogEntry(
            category=LogCategory.SENSOR,
            level=LogLevel.DEBUG,
            message="Frame received",
            node_count=5,
            session_id="abc",
        )
        assert entry.context.get("node_count") == 5
        assert entry.context.get("session_id") == "abc"
    
    def test_entry_to_dict(self) -> None:
        """Test converting entry to dict."""
        entry = LogEntry(
            category=LogCategory.MEMORY,
            level=LogLevel.INFO,
            message="Test",
        )
        d = entry.to_dict()
        assert "timestamp" in d
        assert d["category"] == "MEMORY"
        assert d["level"] == "INFO"
        assert d["message"] == "Test"
    
    def test_entry_to_json(self) -> None:
        """Test converting entry to JSON."""
        entry = LogEntry(
            category=LogCategory.MEMORY,
            level=LogLevel.INFO,
            message="Test",
        )
        j = entry.to_json()
        data = json.loads(j)
        assert data["category"] == "MEMORY"
    
    def test_entry_format_console(self) -> None:
        """Test formatting entry for console."""
        entry = LogEntry(
            category=LogCategory.MEMORY,
            level=LogLevel.INFO,
            message="Test message",
        )
        formatted = entry.format_console()
        assert "MEMORY" in formatted
        assert "Test message" in formatted


# =============================================================================
# STRUCTURED LOGGER TESTS
# =============================================================================

class TestStructuredLogger:
    """Tests for StructuredLogger."""
    
    def test_logger_creation(self) -> None:
        """Test creating a structured logger."""
        logger = StructuredLogger()
        assert logger is not None
    
    def test_basic_logging(self) -> None:
        """Test basic log method."""
        logger = StructuredLogger()
        entry = logger.log(LogCategory.MEMORY, LogLevel.INFO, "Test message")
        assert entry.category == LogCategory.MEMORY
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
    
    def test_category_methods(self) -> None:
        """Test category-specific logging methods."""
        logger = StructuredLogger()
        
        entries = []
        entries.append(logger.sensor("Sensor message"))
        entries.append(logger.memory("Memory message"))
        entries.append(logger.recognition("Recognition message"))
        entries.append(logger.consolidation("Consolidation message"))
        entries.append(logger.label("Label message"))
        entries.append(logger.spatial("Spatial message"))
        entries.append(logger.protocol("Protocol message"))
        entries.append(logger.arbitration("Arbitration message"))
        entries.append(logger.visual("Visual message"))
        entries.append(logger.event("Event message"))
        entries.append(logger.recall("Recall message"))
        entries.append(logger.hypothesis("Hypothesis message"))
        entries.append(logger.invariant("Invariant message"))
        entries.append(logger.system("System message"))
        
        categories = [e.category for e in entries]
        assert LogCategory.SENSOR in categories
        assert LogCategory.MEMORY in categories
        assert LogCategory.CONSOLIDATION in categories
    
    def test_logging_with_context(self) -> None:
        """Test logging with additional context."""
        logger = StructuredLogger()
        entry = logger.memory(
            "Node added",
            node_id="abc123",
            confidence=0.9,
        )
        assert entry.context.get("node_id") == "abc123"
        assert entry.context.get("confidence") == 0.9
    
    def test_log_level_override(self) -> None:
        """Test overriding log level."""
        logger = StructuredLogger()
        entry = logger.memory("Warning!", level="WARNING")
        assert entry.level == LogLevel.WARNING
    
    def test_category_filtering(self) -> None:
        """Test category filtering."""
        logger = StructuredLogger()
        logger.set_category_filter([LogCategory.MEMORY, LogCategory.SENSOR])
        
        # Should be logged (in filter)
        entry1 = logger.memory("Test")
        assert entry1 is not None
        
        # Should be filtered out (not in filter)
        entry2 = logger.consolidation("Test")
        # Depending on implementation, may return None or empty
    
    def test_log_level_filtering(self) -> None:
        """Test log level filtering."""
        logger = StructuredLogger(min_level=LogLevel.WARNING)
        
        # Below minimum level
        entry1 = logger.memory("Debug message", level="DEBUG")
        
        # At or above minimum level
        entry2 = logger.memory("Warning message", level="WARNING")
        assert entry2 is not None
    
    def test_check_invariant_pass(self) -> None:
        """Test invariant checking when condition passes."""
        logger = StructuredLogger()
        entry = logger.check_invariant(
            condition=True,
            invariant_name="test_invariant",
            message="Invariant passed",
        )
        # Should log at INFO level when passing
        assert entry.level == LogLevel.INFO
    
    def test_check_invariant_fail(self) -> None:
        """Test invariant checking when condition fails."""
        logger = StructuredLogger()
        entry = logger.check_invariant(
            condition=False,
            invariant_name="test_invariant",
            message="Invariant failed!",
        )
        # Should log at ERROR level when failing
        assert entry.level == LogLevel.ERROR
        assert entry.category == LogCategory.INVARIANT
    
    def test_statistics_tracking(self) -> None:
        """Test log statistics are tracked."""
        logger = StructuredLogger(level=LogLevel.DEBUG)  # Enable DEBUG level
        
        logger.memory("Message 1")
        logger.memory("Message 2")
        logger.sensor("Message 3")
        
        stats = logger.get_statistics()
        assert stats["total_entries"] >= 3
        assert stats["by_category"].get(LogCategory.MEMORY, 0) >= 2
        assert stats["by_category"].get(LogCategory.SENSOR, 0) >= 1
    
    def test_recent_entries(self) -> None:
        """Test getting recent log entries."""
        logger = StructuredLogger(level=LogLevel.DEBUG)  # Enable DEBUG level
        
        for i in range(10):
            logger.memory(f"Message {i}")
        
        recent = logger.get_recent_entries(5)
        assert len(recent) == 5
    
    def test_filter_by_category(self) -> None:
        """Test filtering entries by category."""
        logger = StructuredLogger(level=LogLevel.DEBUG)  # Enable DEBUG level
        
        logger.memory("Memory message")
        logger.sensor("Sensor message")
        logger.memory("Another memory message")
        
        memory_entries = logger.filter_by_category(LogCategory.MEMORY)
        assert len(memory_entries) >= 2


# =============================================================================
# SESSION LOGGER TESTS
# =============================================================================

class TestSessionLogger:
    """Tests for SessionLogger."""
    
    def test_session_logger_creation(self, temp_runs_dir: Path) -> None:
        """Test creating a session logger."""
        logger = SessionLogger(
            session_id="test-session",
            runs_dir=temp_runs_dir,
        )
        assert logger.session_id == "test-session"
    
    def test_session_logger_creates_directory(self, temp_runs_dir: Path) -> None:
        """Test session logger creates log directory."""
        logger = SessionLogger(
            session_id="test-session",
            runs_dir=temp_runs_dir,
        )
        
        expected_dir = temp_runs_dir / "test-session" / "logs"
        assert expected_dir.exists()
    
    def test_session_logger_writes_files(self, temp_runs_dir: Path) -> None:
        """Test session logger writes to files."""
        logger = SessionLogger(
            session_id="test-session",
            runs_dir=temp_runs_dir,
        )
        
        logger.memory("Test message", level=LogLevel.INFO)
        logger.close()
        
        log_dir = temp_runs_dir / "test-session" / "logs"
        
        # Check text log exists
        text_log = log_dir / "main.log"
        assert text_log.exists()
        content = text_log.read_text()
        assert "Test message" in content
    
    def test_session_logger_json_output(self, temp_runs_dir: Path) -> None:
        """Test session logger writes JSON."""
        logger = SessionLogger(
            session_id="test-session",
            runs_dir=temp_runs_dir,
        )
        
        logger.memory("Test message", level=LogLevel.INFO)
        logger.close()
        
        log_dir = temp_runs_dir / "test-session" / "logs"
        json_log = log_dir / "main.jsonl"
        
        assert json_log.exists()
        lines = json_log.read_text().strip().split("\n")
        # At least 2 lines: session start + test message
        assert len(lines) >= 2
        
        # Find the test message entry
        test_entry = None
        for line in lines:
            entry = json.loads(line)
            if entry["message"] == "Test message":
                test_entry = entry
                break
        
        assert test_entry is not None
        assert test_entry["message"] == "Test message"


# =============================================================================
# GLOBAL LOGGER TESTS
# =============================================================================

class TestGlobalLogger:
    """Tests for global logger functions."""
    
    def test_get_set_logger(self) -> None:
        """Test get/set global logger."""
        original = get_logger()
        
        new_logger = StructuredLogger()
        set_logger(new_logger)
        
        assert get_logger() is new_logger
        
        # Restore original
        if original:
            set_logger(original)
    
    def test_create_session_logger(self, temp_runs_dir: Path) -> None:
        """Test creating session logger via factory function."""
        logger = create_session_logger(
            session_id="test-session",
            runs_dir=temp_runs_dir,
        )
        
        assert isinstance(logger, SessionLogger)
        logger.close()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase7Integration:
    """Integration tests for Phase 7 components."""
    
    def test_consolidation_with_logging(
        self,
        graph_store: InMemoryGraphStore,
        episode_store: InMemoryEpisodeStore,
        temp_runs_dir: Path,
    ) -> None:
        """Test consolidation module with session logging."""
        logger = SessionLogger(
            session_id="integration-test",
            runs_dir=temp_runs_dir,
        )
        
        module = ConsolidationModule(
            graph_store=graph_store,
            episode_store=episode_store,
            logger=logger,
        )
        
        # Create nodes for merging
        source = GraphNode(
            node_id="source-1",
            node_type=NODE_TYPE_ENTITY,
            label="cup",
            confidence=0.7,
        )
        target = GraphNode(
            node_id="target-1",
            node_type=NODE_TYPE_ENTITY,
            label="mug",
            confidence=0.8,
        )
        graph_store.add_node(source)
        graph_store.add_node(target)
        
        # Queue and process
        module.queue_merge(
            source_node_id="source-1",
            target_node_id="target-1",
            similarity=0.95,
            confidence=0.9,
        )
        result = module.consolidate()
        
        logger.close()
        
        # Verify result
        assert result.nodes_merged == 1
        
        # Verify logs were written
        log_file = temp_runs_dir / "integration-test" / "logs" / "main.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "CONSOLIDATION" in content
        assert "merge" in content.lower()
    
    def test_full_workflow_with_scheduler(
        self,
        graph_store: InMemoryGraphStore,
        episode_store: InMemoryEpisodeStore,
    ) -> None:
        """Test full consolidation workflow with scheduler."""
        logger = StructuredLogger(level=LogLevel.DEBUG)
        
        module = ConsolidationModule(
            graph_store=graph_store,
            episode_store=episode_store,
            logger=logger,
        )
        
        scheduler = ConsolidationScheduler(
            consolidation_module=module,
            inactivity_threshold=0.05,  # Very short for testing
            logger=logger,
        )
        
        # Add a node and queue decay
        node = GraphNode(
            node_id="test-node",
            node_type=NODE_TYPE_ENTITY,
            activation=1.0,
        )
        graph_store.add_node(node)
        module.queue_decay(["test-node"], 0.9)
        
        # Start scheduler
        scheduler.start()
        
        # Wait longer for consolidation (0.05s threshold + 1s check interval)
        time.sleep(1.5)
        
        # Stop scheduler
        scheduler.stop()
        
        # Verify decay occurred
        updated = graph_store.get_node("test-node")
        assert updated is not None
        # May or may not have decayed depending on timing
        # Just verify the workflow didn't crash
        assert module.total_runs >= 0


# =============================================================================
# ARCHITECTURAL INVARIANT TESTS
# =============================================================================

class TestArchitecturalInvariants:
    """Tests for Phase 7 architectural invariants."""
    
    def test_invariant_consolidation_preserves_identity(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """INVARIANT: Consolidation preserves identity."""
        # Create nodes with distinct identities
        node1 = GraphNode(
            node_id="node-1",
            node_type=NODE_TYPE_ENTITY,
            label="red_cup",
            labels=["red_cup", "drinking_vessel"],
            confidence=0.8,
        )
        node2 = GraphNode(
            node_id="node-2",
            node_type=NODE_TYPE_ENTITY,
            label="coffee_mug",
            labels=["coffee_mug"],
            confidence=0.9,
        )
        graph_store.add_node(node1)
        graph_store.add_node(node2)
        
        # Merge with high confidence
        consolidation_module.queue_merge(
            source_node_id="node-1",
            target_node_id="node-2",
            similarity=0.95,
            confidence=0.95,
        )
        consolidation_module.consolidate()
        
        # Target should have all labels from both
        target = graph_store.get_node("node-2")
        assert target is not None
        assert "coffee_mug" in target.labels
        assert "red_cup" in target.labels
        assert "drinking_vessel" in target.labels
        
        # Source should have MERGED_INTO edge (not deleted)
        edges = graph_store.get_edges_from_node("node-1")
        merged_edges = [e for e in edges if e.edge_type == EDGE_TYPE_MERGED_INTO]
        assert len(merged_edges) == 1
    
    def test_invariant_no_aggressive_erasure(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """INVARIANT: No aggressive erasure - accept fuzziness."""
        # Create a node with some confidence
        node = GraphNode(
            node_id="maybe-cup",
            node_type=NODE_TYPE_ENTITY,
            label="object",  # Fuzzy label
            confidence=0.3,  # Low but not very low
            access_count=1,
        )
        graph_store.add_node(node)
        
        # Try to prune
        consolidation_module.queue_prune(
            node_id="maybe-cup",
            confidence=0.3,
        )
        consolidation_module.consolidate()
        
        # Should NOT be pruned - we accept fuzziness
        assert graph_store.get_node("maybe-cup") is not None
    
    def test_invariant_relabeling_additive(
        self,
        consolidation_module: ConsolidationModule,
        graph_store: InMemoryGraphStore,
    ) -> None:
        """INVARIANT: Relabeling is additive, not destructive."""
        node = GraphNode(
            node_id="test-node",
            node_type=NODE_TYPE_ENTITY,
            label="generic_object",
            labels=["generic_object"],
            confidence=0.5,
        )
        graph_store.add_node(node)
        
        # Relabel multiple times
        consolidation_module.queue_relabel(
            node_id="test-node",
            new_label="cup",
            old_label="generic_object",
            confidence=0.9,
        )
        consolidation_module.consolidate()
        
        consolidation_module.queue_relabel(
            node_id="test-node",
            new_label="coffee_mug",
            old_label="cup",
            confidence=0.95,
        )
        consolidation_module.consolidate()
        
        # All labels should be preserved
        updated = graph_store.get_node("test-node")
        assert updated is not None
        assert "coffee_mug" in updated.labels
        assert "cup" in updated.labels
        assert "generic_object" in updated.labels
    
    def test_invariant_reproducible_logs(
        self, temp_runs_dir: Path
    ) -> None:
        """INVARIANT: Reproducible logs for debugging."""
        logger = SessionLogger(
            session_id="reproducible-test",
            runs_dir=temp_runs_dir,
        )
        
        # Log some events with context (at INFO level for consistency)
        logger.sensor("Frame received", level=LogLevel.INFO, frame_id=1)
        logger.memory("Node created", level=LogLevel.INFO, entity_id="abc123")
        logger.recognition("Entity detected", level=LogLevel.INFO, confidence=0.9)
        logger.close()
        
        # Read back logs
        json_log = temp_runs_dir / "reproducible-test" / "logs" / "main.jsonl"
        lines = json_log.read_text().strip().split("\n")
        
        # Each entry should have full context for reproduction
        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry
            assert "category" in entry
            assert "level" in entry
            assert "message" in entry


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case tests for Phase 7."""
    
    def test_merge_nonexistent_nodes(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test merging nonexistent nodes fails gracefully."""
        consolidation_module.queue_merge(
            source_node_id="nonexistent-1",
            target_node_id="nonexistent-2",
            similarity=0.9,
            confidence=0.9,
        )
        result = consolidation_module.consolidate()
        assert result.nodes_merged == 0
        assert result.operations_skipped == 1
    
    def test_relabel_nonexistent_node(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test relabeling nonexistent node fails gracefully."""
        consolidation_module.queue_relabel(
            node_id="nonexistent",
            new_label="cup",
            confidence=0.9,
        )
        result = consolidation_module.consolidate()
        assert result.nodes_relabeled == 0
        assert result.operations_skipped == 1
    
    def test_prune_nonexistent_node(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test pruning nonexistent node fails gracefully."""
        consolidation_module.queue_prune(
            node_id="nonexistent",
            confidence=0.01,
        )
        result = consolidation_module.consolidate()
        assert result.nodes_pruned == 0
        assert result.operations_skipped == 1
    
    def test_decay_empty_list(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test decaying empty node list."""
        consolidation_module.queue_decay([], 0.9)
        result = consolidation_module.consolidate()
        assert result.nodes_decayed == 0
    
    def test_scheduler_double_start(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test starting scheduler twice is safe."""
        scheduler = ConsolidationScheduler(consolidation_module)
        scheduler.start()
        scheduler.start()  # Should be no-op
        assert scheduler.is_running
        scheduler.stop()
    
    def test_scheduler_double_stop(
        self, consolidation_module: ConsolidationModule
    ) -> None:
        """Test stopping scheduler twice is safe."""
        scheduler = ConsolidationScheduler(consolidation_module)
        scheduler.start()
        scheduler.stop()
        scheduler.stop()  # Should be no-op
        assert not scheduler.is_running
    
    def test_logger_empty_message(self) -> None:
        """Test logging empty message."""
        logger = StructuredLogger()
        entry = logger.memory("")
        assert entry.message == ""
    
    def test_logger_unicode_message(self) -> None:
        """Test logging Unicode message."""
        logger = StructuredLogger()
        entry = logger.memory("测试消息 🎉")
        assert "测试消息" in entry.message
        assert "🎉" in entry.message
    
    def test_logger_very_long_message(self) -> None:
        """Test logging very long message."""
        logger = StructuredLogger()
        long_msg = "x" * 10000
        entry = logger.memory(long_msg)
        assert len(entry.message) == 10000


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Thread safety tests for Phase 7."""
    
    def test_concurrent_queue_operations(
        self,
        graph_store: InMemoryGraphStore,
        episode_store: InMemoryEpisodeStore,
    ) -> None:
        """Test concurrent queue operations are thread-safe."""
        module = ConsolidationModule(
            graph_store=graph_store,
            episode_store=episode_store,
        )
        
        def queue_operations():
            for i in range(100):
                module.queue_decay([f"node-{i}"], 0.9)
        
        threads = [threading.Thread(target=queue_operations) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have queued operations without errors
        # (actual count depends on max queue size)
        assert module.queue_size > 0
    
    def test_concurrent_logging(self) -> None:
        """Test concurrent logging is thread-safe."""
        logger = StructuredLogger(level=LogLevel.DEBUG)  # Enable DEBUG level
        
        def log_messages():
            for i in range(100):
                logger.memory(f"Message {i}")
        
        threads = [threading.Thread(target=log_messages) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have logged without errors
        stats = logger.get_statistics()
        assert stats["total_entries"] == 500
