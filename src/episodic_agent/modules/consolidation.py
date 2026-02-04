"""Memory consolidation module for background graph maintenance.

ARCHITECTURAL INVARIANT: Consolidation preserves identity and accepts fuzziness.
No aggressive erasure - memory integrity is paramount.

This module provides:
- ConsolidationModule: Queues and processes consolidation operations
- MergeOperation: Intelligent node merging that preserves identity
- RelabelOperation: Deferred relabeling of nodes
- ConsolidationScheduler: Triggers consolidation after inactivity
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from episodic_agent.core.interfaces import Consolidator, EpisodeStore, GraphStore
from episodic_agent.schemas.graph import (
    EDGE_TYPE_ALIAS_OF,
    EDGE_TYPE_MERGED_INTO,
    EDGE_TYPE_SIMILAR_TO,
    GraphEdge,
    GraphNode,
    NODE_TYPE_ENTITY,
)

if TYPE_CHECKING:
    from episodic_agent.utils.logging import SessionLogger, StructuredLogger


# =============================================================================
# CONFIGURATION
# =============================================================================

# Consolidation timing
DEFAULT_INACTIVITY_THRESHOLD = 30.0  # Seconds of inactivity before consolidation
DEFAULT_MAX_QUEUE_SIZE = 1000        # Maximum queued operations
DEFAULT_BATCH_SIZE = 50              # Operations to process per consolidation run

# Merge thresholds
MERGE_SIMILARITY_THRESHOLD = 0.85    # Minimum similarity to consider merge
MERGE_CONFIDENCE_THRESHOLD = 0.7     # Minimum confidence to auto-merge
FUZZY_MATCH_THRESHOLD = 0.6          # Threshold for "fuzzy" matching

# Relabel thresholds
RELABEL_CONFIDENCE_THRESHOLD = 0.8   # Minimum confidence to auto-relabel


# =============================================================================
# OPERATION TYPES
# =============================================================================

class OperationType(str, Enum):
    """Types of consolidation operations."""
    MERGE = "merge"           # Merge two nodes
    RELABEL = "relabel"       # Relabel a node
    PRUNE = "prune"           # Remove low-confidence node
    STRENGTHEN = "strengthen"  # Strengthen high-confidence links
    DECAY = "decay"           # Decay unused nodes/edges


@dataclass
class ConsolidationOperation:
    """Base class for consolidation operations."""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType = OperationType.MERGE
    priority: int = 0  # Higher = more urgent
    created_at: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other: "ConsolidationOperation") -> bool:
        """Compare by priority (for priority queue)."""
        return self.priority > other.priority


@dataclass
class MergeOperation(ConsolidationOperation):
    """Operation to merge two nodes.
    
    ARCHITECTURAL INVARIANT: Merging preserves identity.
    The merged node retains all labels and connections from both sources.
    """
    operation_type: OperationType = OperationType.MERGE
    source_node_id: str = ""
    target_node_id: str = ""
    similarity: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    
    def __post_init__(self):
        if not self.reason:
            self.reason = f"Similarity {self.similarity:.2f}"


@dataclass
class RelabelOperation(ConsolidationOperation):
    """Operation to relabel a node.
    
    ARCHITECTURAL INVARIANT: Relabeling adds labels, doesn't remove them.
    Old labels become aliases.
    """
    operation_type: OperationType = OperationType.RELABEL
    node_id: str = ""
    new_label: str = ""
    old_label: str = ""
    confidence: float = 0.0
    source: str = ""  # Where the new label came from (user, inference, etc.)


@dataclass
class PruneOperation(ConsolidationOperation):
    """Operation to prune a low-confidence node."""
    operation_type: OperationType = OperationType.PRUNE
    node_id: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass
class DecayOperation(ConsolidationOperation):
    """Operation to decay activation of unused nodes."""
    operation_type: OperationType = OperationType.DECAY
    node_ids: list[str] = field(default_factory=list)
    decay_factor: float = 0.95


# =============================================================================
# CONSOLIDATION RESULT
# =============================================================================

@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    
    operations_processed: int = 0
    operations_succeeded: int = 0
    operations_failed: int = 0
    operations_skipped: int = 0
    
    nodes_merged: int = 0
    nodes_relabeled: int = 0
    nodes_pruned: int = 0
    nodes_decayed: int = 0
    
    errors: list[str] = field(default_factory=list)
    
    def complete(self) -> None:
        """Mark consolidation as complete."""
        self.completed_at = datetime.now()
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds() * 1000


# =============================================================================
# CONSOLIDATION MODULE
# =============================================================================

class ConsolidationModule(Consolidator):
    """Memory consolidation module with queued operations.
    
    ARCHITECTURAL INVARIANT: Consolidation preserves identity and accepts fuzziness.
    - Merging combines nodes but retains all labels as aliases
    - Relabeling adds new labels but keeps old ones as alternatives
    - Pruning only removes very low-confidence, unconnected nodes
    - Decay is gradual and reversible through re-access
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        episode_store: EpisodeStore,
        logger: "StructuredLogger | SessionLogger | None" = None,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize the consolidation module.
        
        Args:
            graph_store: Graph memory storage
            episode_store: Episode memory storage
            logger: Optional structured logger
            max_queue_size: Maximum operations in queue
            batch_size: Operations per consolidation run
        """
        self._graph = graph_store
        self._episodes = episode_store
        self._logger = logger
        self._max_queue_size = max_queue_size
        self._batch_size = batch_size
        
        # Operation queue (priority queue behavior via sorted insert)
        self._queue: deque[ConsolidationOperation] = deque(maxlen=max_queue_size)
        self._queue_lock = threading.Lock()
        
        # Statistics
        self._total_runs = 0
        self._total_operations = 0
        self._last_run: ConsolidationResult | None = None
        
        # Callbacks
        self._on_complete_callbacks: list[Callable[[ConsolidationResult], None]] = []
    
    # -------------------------------------------------------------------------
    # QUEUE MANAGEMENT
    # -------------------------------------------------------------------------
    
    def queue_merge(
        self,
        source_node_id: str,
        target_node_id: str,
        similarity: float,
        confidence: float = 0.0,
        reason: str = "",
        priority: int = 0,
    ) -> MergeOperation:
        """Queue a merge operation.
        
        Args:
            source_node_id: Node to merge from
            target_node_id: Node to merge into
            similarity: Similarity score between nodes
            confidence: Confidence in the merge
            reason: Reason for the merge
            priority: Operation priority
            
        Returns:
            The queued operation
        """
        op = MergeOperation(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            similarity=similarity,
            confidence=confidence,
            reason=reason,
            priority=priority,
        )
        self._enqueue(op)
        
        if self._logger:
            self._logger.consolidation(
                f"Queued merge: {source_node_id[:8]} -> {target_node_id[:8]} "
                f"(sim={similarity:.2f}, conf={confidence:.2f})",
            )
        
        return op
    
    def queue_relabel(
        self,
        node_id: str,
        new_label: str,
        old_label: str = "",
        confidence: float = 0.0,
        source: str = "inference",
        priority: int = 0,
    ) -> RelabelOperation:
        """Queue a relabel operation.
        
        Args:
            node_id: Node to relabel
            new_label: New label to apply
            old_label: Current label (for logging)
            confidence: Confidence in the new label
            source: Source of the new label
            priority: Operation priority
            
        Returns:
            The queued operation
        """
        op = RelabelOperation(
            node_id=node_id,
            new_label=new_label,
            old_label=old_label,
            confidence=confidence,
            source=source,
            priority=priority,
        )
        self._enqueue(op)
        
        if self._logger:
            self._logger.consolidation(
                f"Queued relabel: {node_id[:8]} '{old_label}' -> '{new_label}' "
                f"(conf={confidence:.2f}, src={source})",
            )
        
        return op
    
    def queue_prune(
        self,
        node_id: str,
        confidence: float,
        reason: str = "",
        priority: int = -1,  # Low priority by default
    ) -> PruneOperation:
        """Queue a prune operation.
        
        Args:
            node_id: Node to prune
            confidence: Current confidence (should be low)
            reason: Reason for pruning
            priority: Operation priority
            
        Returns:
            The queued operation
        """
        op = PruneOperation(
            node_id=node_id,
            confidence=confidence,
            reason=reason,
            priority=priority,
        )
        self._enqueue(op)
        
        if self._logger:
            self._logger.consolidation(
                f"Queued prune: {node_id[:8]} (conf={confidence:.2f}, reason={reason})",
            )
        
        return op
    
    def queue_decay(
        self,
        node_ids: list[str],
        decay_factor: float = 0.95,
        priority: int = -2,  # Very low priority
    ) -> DecayOperation:
        """Queue a decay operation.
        
        Args:
            node_ids: Nodes to decay
            decay_factor: Decay multiplier (0-1)
            priority: Operation priority
            
        Returns:
            The queued operation
        """
        op = DecayOperation(
            node_ids=node_ids,
            decay_factor=decay_factor,
            priority=priority,
        )
        self._enqueue(op)
        
        if self._logger:
            self._logger.consolidation(
                f"Queued decay: {len(node_ids)} nodes (factor={decay_factor:.2f})",
            )
        
        return op
    
    def _enqueue(self, op: ConsolidationOperation) -> None:
        """Add operation to queue."""
        with self._queue_lock:
            self._queue.append(op)
    
    def _dequeue_batch(self) -> list[ConsolidationOperation]:
        """Get a batch of operations from queue."""
        with self._queue_lock:
            batch = []
            for _ in range(min(self._batch_size, len(self._queue))):
                if self._queue:
                    batch.append(self._queue.popleft())
            # Sort by priority
            batch.sort(key=lambda x: x.priority, reverse=True)
            return batch
    
    @property
    def queue_size(self) -> int:
        """Current queue size."""
        with self._queue_lock:
            return len(self._queue)
    
    # -------------------------------------------------------------------------
    # CONSOLIDATION INTERFACE
    # -------------------------------------------------------------------------
    
    def consolidate(
        self,
        episode_store: EpisodeStore | None = None,
        graph_store: GraphStore | None = None,
    ) -> ConsolidationResult:
        """Run consolidation on memory stores.
        
        Processes queued operations in priority order.
        
        Args:
            episode_store: Episode storage (uses internal if not provided)
            graph_store: Graph storage (uses internal if not provided)
            
        Returns:
            ConsolidationResult with operation counts
        """
        graph = graph_store or self._graph
        episodes = episode_store or self._episodes
        
        result = ConsolidationResult()
        
        if self._logger:
            self._logger.consolidation(
                f"Starting consolidation - {self.queue_size} items queued",
            )
        
        # Get batch of operations
        batch = self._dequeue_batch()
        
        for op in batch:
            result.operations_processed += 1
            
            try:
                if isinstance(op, MergeOperation):
                    success = self._process_merge(op, graph)
                    if success:
                        result.nodes_merged += 1
                        result.operations_succeeded += 1
                    else:
                        result.operations_skipped += 1
                        
                elif isinstance(op, RelabelOperation):
                    success = self._process_relabel(op, graph)
                    if success:
                        result.nodes_relabeled += 1
                        result.operations_succeeded += 1
                    else:
                        result.operations_skipped += 1
                        
                elif isinstance(op, PruneOperation):
                    success = self._process_prune(op, graph)
                    if success:
                        result.nodes_pruned += 1
                        result.operations_succeeded += 1
                    else:
                        result.operations_skipped += 1
                        
                elif isinstance(op, DecayOperation):
                    count = self._process_decay(op, graph)
                    result.nodes_decayed += count
                    result.operations_succeeded += 1
                    
            except Exception as e:
                result.operations_failed += 1
                result.errors.append(f"{op.operation_type.value}: {str(e)}")
                if self._logger:
                    self._logger.consolidation(
                        f"Operation failed: {op.operation_type.value} - {str(e)}",
                        level="ERROR",
                    )
        
        result.complete()
        self._total_runs += 1
        self._total_operations += result.operations_processed
        self._last_run = result
        
        if self._logger:
            self._logger.consolidation(
                f"Consolidation complete - processed={result.operations_processed}, "
                f"succeeded={result.operations_succeeded}, failed={result.operations_failed}, "
                f"duration={result.duration_ms:.1f}ms",
            )
        
        # Notify callbacks
        for callback in self._on_complete_callbacks:
            callback(result)
        
        return result
    
    # -------------------------------------------------------------------------
    # OPERATION PROCESSING
    # -------------------------------------------------------------------------
    
    def _process_merge(self, op: MergeOperation, graph: GraphStore) -> bool:
        """Process a merge operation.
        
        ARCHITECTURAL INVARIANT: Merging preserves identity.
        - Target node retains its ID
        - Source labels become aliases of target
        - All edges from source are redirected to target
        - Source node gets MERGED_INTO edge to target
        """
        source = graph.get_node(op.source_node_id)
        target = graph.get_node(op.target_node_id)
        
        if not source or not target:
            return False
        
        # Check similarity threshold
        if op.similarity < MERGE_SIMILARITY_THRESHOLD:
            return False
        
        # Check confidence threshold for auto-merge
        if op.confidence < MERGE_CONFIDENCE_THRESHOLD:
            # Below auto-merge threshold, skip (needs user confirmation)
            return False
        
        # Add source labels to target as aliases
        for label in source.labels:
            if label not in target.labels:
                target.labels.append(label)
        
        if source.label and source.label not in target.labels:
            target.labels.append(source.label)
        
        # Update target confidence (take max)
        target.confidence = max(target.confidence, source.confidence)
        
        # Merge embeddings (average)
        if source.embedding and target.embedding:
            merged_embedding = [
                (s + t) / 2 for s, t in zip(source.embedding, target.embedding)
            ]
            target.embedding = merged_embedding
        elif source.embedding:
            target.embedding = source.embedding
        
        # Update target in graph
        graph.update_node(target)
        
        # Redirect edges from source to target
        source_edges = graph.get_edges_from_node(op.source_node_id)
        for edge in source_edges:
            if edge.target_node_id != op.target_node_id:
                # Create new edge from target
                new_edge = GraphEdge(
                    edge_id=str(uuid.uuid4()),
                    edge_type=edge.edge_type,
                    source_node_id=op.target_node_id,
                    target_node_id=edge.target_node_id,
                    weight=edge.weight,
                    confidence=edge.confidence,
                    salience=edge.salience,
                )
                graph.add_edge(new_edge)
        
        # Create MERGED_INTO edge
        merge_edge = GraphEdge(
            edge_id=str(uuid.uuid4()),
            edge_type=EDGE_TYPE_MERGED_INTO,
            source_node_id=op.source_node_id,
            target_node_id=op.target_node_id,
            confidence=op.confidence,
            extras={"reason": op.reason, "similarity": op.similarity},
        )
        graph.add_edge(merge_edge)
        
        if self._logger:
            self._logger.consolidation(
                f"Merged: {op.source_node_id[:8]} -> {op.target_node_id[:8]} "
                f"(labels: {target.labels})",
            )
        
        return True
    
    def _process_relabel(self, op: RelabelOperation, graph: GraphStore) -> bool:
        """Process a relabel operation.
        
        ARCHITECTURAL INVARIANT: Relabeling adds labels, doesn't remove them.
        Old label becomes an alias.
        """
        node = graph.get_node(op.node_id)
        if not node:
            return False
        
        # Check confidence threshold
        if op.confidence < RELABEL_CONFIDENCE_THRESHOLD:
            return False
        
        old_label = node.label
        
        # Keep old label as alias
        if old_label and old_label != "unknown" and old_label not in node.labels:
            node.labels.append(old_label)
        
        # Set new primary label
        node.label = op.new_label
        
        # Add new label to labels list if not present
        if op.new_label not in node.labels:
            node.labels.insert(0, op.new_label)
        
        # Update confidence
        node.confidence = max(node.confidence, op.confidence)
        
        graph.update_node(node)
        
        # Create ALIAS_OF edge if old label was meaningful
        if old_label and old_label != "unknown":
            # Find or create concept node for old label
            alias_edge = GraphEdge(
                edge_id=str(uuid.uuid4()),
                edge_type=EDGE_TYPE_ALIAS_OF,
                source_node_id=op.node_id,
                target_node_id=op.node_id,  # Self-edge for alias tracking
                extras={"old_label": old_label, "new_label": op.new_label, "source": op.source},
            )
            graph.add_edge(alias_edge)
        
        if self._logger:
            self._logger.consolidation(
                f"Relabeled: {op.node_id[:8]} '{old_label}' -> '{op.new_label}' "
                f"(src={op.source})",
            )
        
        return True
    
    def _process_prune(self, op: PruneOperation, graph: GraphStore) -> bool:
        """Process a prune operation.
        
        ARCHITECTURAL INVARIANT: Only prune very low-confidence, unconnected nodes.
        We accept fuzziness - don't aggressively erase.
        """
        node = graph.get_node(op.node_id)
        if not node:
            return False
        
        # Very conservative pruning
        if node.confidence > 0.1:
            # Too confident to prune
            return False
        
        if node.access_count > 2:
            # Has been accessed, don't prune
            return False
        
        # Check if node has any edges
        edges = graph.get_edges_from_node(op.node_id)
        if len(edges) > 0:
            # Has connections, don't prune
            return False
        
        # Safe to prune
        graph.remove_node(op.node_id)
        
        if self._logger:
            self._logger.consolidation(
                f"Pruned: {op.node_id[:8]} (reason={op.reason})",
            )
        
        return True
    
    def _process_decay(self, op: DecayOperation, graph: GraphStore) -> int:
        """Process a decay operation.
        
        Gradually decays activation of unused nodes.
        """
        count = 0
        for node_id in op.node_ids:
            node = graph.get_node(node_id)
            if node:
                node.activation *= op.decay_factor
                node.base_activation *= op.decay_factor
                graph.update_node(node)
                count += 1
        
        if self._logger and count > 0:
            self._logger.consolidation(
                f"Decayed: {count} nodes (factor={op.decay_factor:.2f})",
            )
        
        return count
    
    # -------------------------------------------------------------------------
    # AUTOMATIC CONSOLIDATION DETECTION
    # -------------------------------------------------------------------------
    
    def find_merge_candidates(
        self,
        similarity_threshold: float = MERGE_SIMILARITY_THRESHOLD,
    ) -> list[tuple[str, str, float]]:
        """Find nodes that could be merged based on similarity.
        
        Args:
            similarity_threshold: Minimum similarity to consider
            
        Returns:
            List of (node_a_id, node_b_id, similarity) tuples
        """
        candidates = []
        
        # Get all entity nodes
        # This is a simplified version - real implementation would use
        # more efficient similarity search
        nodes = list(self._graph.get_all_nodes())
        entity_nodes = [n for n in nodes if n.node_type == NODE_TYPE_ENTITY]
        
        for i, node_a in enumerate(entity_nodes):
            for node_b in entity_nodes[i+1:]:
                # Compare embeddings if available
                if node_a.embedding and node_b.embedding:
                    similarity = self._cosine_similarity(
                        node_a.embedding, node_b.embedding
                    )
                    if similarity >= similarity_threshold:
                        candidates.append((node_a.node_id, node_b.node_id, similarity))
                
                # Compare labels
                elif node_a.label and node_b.label:
                    if node_a.label.lower() == node_b.label.lower():
                        candidates.append((node_a.node_id, node_b.node_id, 1.0))
        
        return candidates
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    
    def register_on_complete(
        self,
        callback: Callable[[ConsolidationResult], None],
    ) -> None:
        """Register callback for consolidation completion."""
        self._on_complete_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    @property
    def total_runs(self) -> int:
        """Total consolidation runs."""
        return self._total_runs
    
    @property
    def total_operations(self) -> int:
        """Total operations processed."""
        return self._total_operations
    
    @property
    def last_run(self) -> ConsolidationResult | None:
        """Result of last consolidation run."""
        return self._last_run


# =============================================================================
# CONSOLIDATION SCHEDULER
# =============================================================================

class ConsolidationScheduler:
    """Schedules consolidation runs based on inactivity.
    
    Triggers consolidation after a period of inactivity (no new frames).
    """
    
    def __init__(
        self,
        consolidation_module: ConsolidationModule,
        inactivity_threshold: float = DEFAULT_INACTIVITY_THRESHOLD,
        logger: "StructuredLogger | SessionLogger | None" = None,
    ):
        """Initialize the scheduler.
        
        Args:
            consolidation_module: The consolidation module to trigger
            inactivity_threshold: Seconds of inactivity before triggering
            logger: Optional structured logger
        """
        self._module = consolidation_module
        self._threshold = inactivity_threshold
        self._logger = logger
        
        self._last_activity = time.time()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
    
    def record_activity(self) -> None:
        """Record that activity occurred (e.g., new frame received)."""
        self._last_activity = time.time()
    
    def start(self) -> None:
        """Start the scheduler thread."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        if self._logger:
            self._logger.consolidation(
                f"Scheduler started (threshold={self._threshold}s)",
            )
    
    def stop(self) -> None:
        """Stop the scheduler thread."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._logger:
            self._logger.consolidation("Scheduler stopped")
    
    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running and not self._stop_event.is_set():
            # Check for inactivity
            elapsed = time.time() - self._last_activity
            
            if elapsed >= self._threshold:
                # Trigger consolidation if there are queued operations
                if self._module.queue_size > 0:
                    if self._logger:
                        self._logger.consolidation(
                            f"Inactivity detected ({elapsed:.1f}s), triggering consolidation",
                        )
                    self._module.consolidate()
                    self._last_activity = time.time()  # Reset after consolidation
            
            # Sleep before next check
            self._stop_event.wait(timeout=1.0)
    
    @property
    def is_running(self) -> bool:
        """Whether scheduler is running."""
        return self._running
    
    @property
    def seconds_since_activity(self) -> float:
        """Seconds since last activity."""
        return time.time() - self._last_activity


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "DEFAULT_INACTIVITY_THRESHOLD",
    "DEFAULT_MAX_QUEUE_SIZE",
    "DEFAULT_BATCH_SIZE",
    "MERGE_SIMILARITY_THRESHOLD",
    "MERGE_CONFIDENCE_THRESHOLD",
    "FUZZY_MATCH_THRESHOLD",
    "RELABEL_CONFIDENCE_THRESHOLD",
    # Enums
    "OperationType",
    # Operations
    "ConsolidationOperation",
    "MergeOperation",
    "RelabelOperation",
    "PruneOperation",
    "DecayOperation",
    # Result
    "ConsolidationResult",
    # Classes
    "ConsolidationModule",
    "ConsolidationScheduler",
]
