"""Cued recall module for context-frame linking.

ARCHITECTURAL INVARIANT: Memory retrieval uses weighted cues with tunable bias.
No predefined salience categories - weights are learned from experience.

This module provides:
- CuedRecallModule: Multi-cue retrieval with salience weighting
- RedundantCueStore: Storage for multiple cue paths to same memory
- EntityHypothesisTracker: Same-entity hypothesis generation and tracking
- LocationRevisitHandler: Triggers recall when entering known locations
"""

from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from episodic_agent.core.interfaces import EpisodeStore, GraphStore
from episodic_agent.schemas.graph import (
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_IN_CONTEXT,
    EDGE_TYPE_OCCURRED_IN,
    EDGE_TYPE_REVISIT,
    EDGE_TYPE_SIMILAR_TO,
    EDGE_TYPE_SPATIAL,
    EDGE_TYPE_TEMPORAL,
    GraphEdge,
    GraphNode,
    NODE_TYPE_ENTITY,
    NODE_TYPE_EPISODE,
    NODE_TYPE_LOCATION,
)
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


# =============================================================================
# CONFIGURATION
# =============================================================================

# Salience thresholds
MIN_SALIENCE_TO_RECALL = 0.1          # Minimum salience to include in recall
HIGH_SALIENCE_THRESHOLD = 0.7          # Threshold for "salient moment"

# Entity hypothesis thresholds
SAME_POSITION_THRESHOLD = 0.5          # Max distance to consider "same position"
VISUAL_SIMILARITY_THRESHOLD = 0.3      # Min visual similarity for hypothesis
HYPOTHESIS_CONFIDENCE_THRESHOLD = 0.5  # Min confidence to create hypothesis

# Revisit thresholds
REVISIT_TIME_THRESHOLD = 60.0          # Seconds between visits to trigger revisit
MAX_RECALLED_EPISODES = 10             # Max episodes to recall on revisit

# Cue storage
MAX_CUES_PER_TARGET = 50               # Max redundant cues per memory


# =============================================================================
# REDUNDANT CUE STORE
# =============================================================================

class RedundantCueStore:
    """Stores multiple cue paths to the same memory.
    
    ARCHITECTURAL INVARIANT: Multiple retrieval paths can trigger the same
    memory, providing resilience and natural recall patterns.
    """
    
    def __init__(self, max_cues_per_target: int = MAX_CUES_PER_TARGET):
        """Initialize the cue store.
        
        Args:
            max_cues_per_target: Maximum cues per memory target
        """
        self._max_cues = max_cues_per_target
        
        # Cue storage: cue_id -> RedundantCue
        self._cues: dict[str, RedundantCue] = {}
        
        # Indexes for fast lookup
        self._by_target: dict[str, set[str]] = defaultdict(set)      # target_node_id -> cue_ids
        self._by_type: dict[CueType, set[str]] = defaultdict(set)    # cue_type -> cue_ids
        self._by_location: dict[str, set[str]] = defaultdict(set)    # location_id -> cue_ids
        self._by_entity: dict[str, set[str]] = defaultdict(set)      # entity_id -> cue_ids
        self._by_label: dict[str, set[str]] = defaultdict(set)       # label -> cue_ids
    
    def add_cue(self, cue: RedundantCue) -> bool:
        """Add a redundant cue.
        
        Args:
            cue: The cue to add
            
        Returns:
            True if added, False if limit reached
        """
        # Check limit
        target_cues = self._by_target[cue.target_node_id]
        if len(target_cues) >= self._max_cues:
            # Evict weakest cue
            self._evict_weakest(cue.target_node_id)
        
        # Store cue
        self._cues[cue.cue_id] = cue
        
        # Update indexes
        self._by_target[cue.target_node_id].add(cue.cue_id)
        self._by_type[cue.cue_type].add(cue.cue_id)
        
        if cue.location_id:
            self._by_location[cue.location_id].add(cue.cue_id)
        if cue.entity_id:
            self._by_entity[cue.entity_id].add(cue.cue_id)
        if cue.label:
            self._by_label[cue.label.lower()].add(cue.cue_id)
        
        return True
    
    def get_cues_for_target(self, target_node_id: str) -> list[RedundantCue]:
        """Get all cues pointing to a target memory."""
        cue_ids = self._by_target.get(target_node_id, set())
        return [self._cues[cid] for cid in cue_ids if cid in self._cues]
    
    def get_cues_by_location(self, location_id: str) -> list[RedundantCue]:
        """Get all cues associated with a location."""
        cue_ids = self._by_location.get(location_id, set())
        return [self._cues[cid] for cid in cue_ids if cid in self._cues]
    
    def get_cues_by_entity(self, entity_id: str) -> list[RedundantCue]:
        """Get all cues associated with an entity."""
        cue_ids = self._by_entity.get(entity_id, set())
        return [self._cues[cid] for cid in cue_ids if cid in self._cues]
    
    def get_cues_by_label(self, label: str) -> list[RedundantCue]:
        """Get all cues associated with a label."""
        cue_ids = self._by_label.get(label.lower(), set())
        return [self._cues[cid] for cid in cue_ids if cid in self._cues]
    
    def get_cues_by_type(self, cue_type: CueType) -> list[RedundantCue]:
        """Get all cues of a specific type."""
        cue_ids = self._by_type.get(cue_type, set())
        return [self._cues[cid] for cid in cue_ids if cid in self._cues]
    
    def access_cue(self, cue_id: str) -> None:
        """Record that a cue was used (strengthens it)."""
        if cue_id in self._cues:
            self._cues[cue_id].access()
    
    def _evict_weakest(self, target_node_id: str) -> None:
        """Evict the weakest cue for a target."""
        cue_ids = self._by_target.get(target_node_id, set())
        if not cue_ids:
            return
        
        # Find weakest by strength
        weakest_id = min(
            cue_ids,
            key=lambda cid: self._cues.get(cid, RedundantCue(
                cue_type=CueType.LOCATION,
                target_node_id=target_node_id,
            )).strength,
        )
        
        self._remove_cue(weakest_id)
    
    def _remove_cue(self, cue_id: str) -> None:
        """Remove a cue and update indexes."""
        cue = self._cues.pop(cue_id, None)
        if not cue:
            return
        
        self._by_target[cue.target_node_id].discard(cue_id)
        self._by_type[cue.cue_type].discard(cue_id)
        
        if cue.location_id:
            self._by_location[cue.location_id].discard(cue_id)
        if cue.entity_id:
            self._by_entity[cue.entity_id].discard(cue_id)
        if cue.label:
            self._by_label[cue.label.lower()].discard(cue_id)
    
    def create_cues_for_memory(
        self,
        target_node_id: str,
        location_id: str | None = None,
        entity_ids: list[str] | None = None,
        labels: list[str] | None = None,
        timestamp: datetime | None = None,
        episode_id: str | None = None,
    ) -> list[RedundantCue]:
        """Create redundant cues for a memory.
        
        Creates cues from multiple paths to enable natural recall.
        
        Args:
            target_node_id: The memory node to create cues for
            location_id: Associated location
            entity_ids: Associated entities
            labels: Associated labels
            timestamp: Associated time
            episode_id: Associated episode
            
        Returns:
            List of created cues
        """
        cues = []
        
        # Location cue
        if location_id:
            cue = RedundantCue(
                cue_type=CueType.LOCATION,
                target_node_id=target_node_id,
                target_episode_id=episode_id,
                location_id=location_id,
            )
            self.add_cue(cue)
            cues.append(cue)
        
        # Entity cues
        for entity_id in (entity_ids or []):
            cue = RedundantCue(
                cue_type=CueType.ENTITY,
                target_node_id=target_node_id,
                target_episode_id=episode_id,
                entity_id=entity_id,
            )
            self.add_cue(cue)
            cues.append(cue)
        
        # Semantic cues
        for label in (labels or []):
            cue = RedundantCue(
                cue_type=CueType.SEMANTIC,
                target_node_id=target_node_id,
                target_episode_id=episode_id,
                label=label,
            )
            self.add_cue(cue)
            cues.append(cue)
        
        # Temporal cue
        if timestamp:
            cue = RedundantCue(
                cue_type=CueType.TEMPORAL,
                target_node_id=target_node_id,
                target_episode_id=episode_id,
                timestamp=timestamp,
            )
            self.add_cue(cue)
            cues.append(cue)
        
        return cues
    
    @property
    def total_cues(self) -> int:
        """Total number of cues stored."""
        return len(self._cues)
    
    @property
    def targets_count(self) -> int:
        """Number of unique targets with cues."""
        return len(self._by_target)


# =============================================================================
# ENTITY HYPOTHESIS TRACKER
# =============================================================================

class EntityHypothesisTracker:
    """Tracks same-entity hypotheses across observations.
    
    ARCHITECTURAL INVARIANT: Same-entity hypotheses are based on spatial
    and visual consistency, not predefined object categories.
    
    Example: Blue mug at position (1, 0, 2) becomes red mug at same
    position → hypothesize they're the same entity (the mug changed color).
    """
    
    def __init__(
        self,
        position_threshold: float = SAME_POSITION_THRESHOLD,
        visual_threshold: float = VISUAL_SIMILARITY_THRESHOLD,
        confidence_threshold: float = HYPOTHESIS_CONFIDENCE_THRESHOLD,
    ):
        """Initialize the tracker.
        
        Args:
            position_threshold: Max distance to consider "same position"
            visual_threshold: Min visual similarity for hypothesis
            confidence_threshold: Min confidence to create hypothesis
        """
        self._position_threshold = position_threshold
        self._visual_threshold = visual_threshold
        self._confidence_threshold = confidence_threshold
        
        # Hypothesis storage: hypothesis_id -> EntityHypothesis
        self._hypotheses: dict[str, EntityHypothesis] = {}
        
        # Indexes
        self._by_location: dict[str, set[str]] = defaultdict(set)  # location -> hypothesis_ids
        self._by_observation: dict[str, set[str]] = defaultdict(set)  # obs_id -> hypothesis_ids
        self._by_status: dict[str, set[str]] = defaultdict(set)  # status -> hypothesis_ids
        
        # Position index for fast lookup: location_id -> list of (obs_id, position)
        self._position_index: dict[str, list[tuple[str, tuple[float, float, float], str, datetime]]] = defaultdict(list)
    
    def record_observation(
        self,
        observation_id: str,
        location_id: str,
        position: tuple[float, float, float],
        label: str | None = None,
        visual_embedding: list[float] | None = None,
        timestamp: datetime | None = None,
    ) -> list[EntityHypothesis]:
        """Record an observation and check for same-entity hypotheses.
        
        Args:
            observation_id: Unique ID of the observation
            location_id: Location where observed
            position: Relative position (agent-centric)
            label: Optional label for the observation
            visual_embedding: Optional visual features
            timestamp: Observation timestamp
            
        Returns:
            List of hypotheses generated
        """
        timestamp = timestamp or datetime.now()
        hypotheses = []
        
        # Check existing observations at this location
        existing = self._position_index.get(location_id, [])
        
        for obs_id, obs_pos, obs_label, obs_time in existing:
            if obs_id == observation_id:
                continue
            
            # Compute position distance
            distance = self._compute_distance(position, obs_pos)
            
            if distance <= self._position_threshold:
                # Compute visual similarity (if available)
                visual_sim = 0.5  # Default if no embeddings
                
                # Compute time gap
                time_gap = (timestamp - obs_time).total_seconds()
                
                # Create hypothesis
                hypothesis = EntityHypothesis.from_observations(
                    obs_a_id=obs_id,
                    obs_b_id=observation_id,
                    location_id=location_id,
                    pos_a=obs_pos,
                    pos_b=position,
                    label_a=obs_label,
                    label_b=label,
                    visual_sim=visual_sim,
                    time_gap=abs(time_gap),
                )
                
                # Only keep if confidence meets threshold
                if hypothesis.confidence >= self._confidence_threshold:
                    self._add_hypothesis(hypothesis)
                    hypotheses.append(hypothesis)
        
        # Add this observation to index
        self._position_index[location_id].append(
            (observation_id, position, label, timestamp)
        )
        
        return hypotheses
    
    def _add_hypothesis(self, hypothesis: EntityHypothesis) -> None:
        """Add a hypothesis to storage."""
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        self._by_location[hypothesis.location_id].add(hypothesis.hypothesis_id)
        self._by_observation[hypothesis.observation_a_id].add(hypothesis.hypothesis_id)
        self._by_observation[hypothesis.observation_b_id].add(hypothesis.hypothesis_id)
        self._by_status[hypothesis.status].add(hypothesis.hypothesis_id)
    
    def get_hypothesis(self, hypothesis_id: str) -> EntityHypothesis | None:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)
    
    def get_hypotheses_for_location(self, location_id: str) -> list[EntityHypothesis]:
        """Get all hypotheses for a location."""
        hyp_ids = self._by_location.get(location_id, set())
        return [self._hypotheses[hid] for hid in hyp_ids if hid in self._hypotheses]
    
    def get_hypotheses_for_observation(self, observation_id: str) -> list[EntityHypothesis]:
        """Get all hypotheses involving an observation."""
        hyp_ids = self._by_observation.get(observation_id, set())
        return [self._hypotheses[hid] for hid in hyp_ids if hid in self._hypotheses]
    
    def get_pending_hypotheses(self) -> list[EntityHypothesis]:
        """Get all pending hypotheses."""
        hyp_ids = self._by_status.get("pending", set())
        return [self._hypotheses[hid] for hid in hyp_ids if hid in self._hypotheses]
    
    def confirm_hypothesis(self, hypothesis_id: str, reason: str = "user confirmed") -> bool:
        """Confirm a hypothesis."""
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            return False
        
        old_status = hypothesis.status
        hypothesis.confirm(reason)
        
        # Update index
        self._by_status[old_status].discard(hypothesis_id)
        self._by_status["confirmed"].add(hypothesis_id)
        
        return True
    
    def reject_hypothesis(self, hypothesis_id: str, reason: str = "user rejected") -> bool:
        """Reject a hypothesis."""
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            return False
        
        old_status = hypothesis.status
        hypothesis.reject(reason)
        
        # Update index
        self._by_status[old_status].discard(hypothesis_id)
        self._by_status["rejected"].add(hypothesis_id)
        
        return True
    
    def _compute_distance(
        self,
        pos_a: tuple[float, float, float],
        pos_b: tuple[float, float, float],
    ) -> float:
        """Compute Euclidean distance between positions."""
        return math.sqrt(
            (pos_a[0] - pos_b[0]) ** 2 +
            (pos_a[1] - pos_b[1]) ** 2 +
            (pos_a[2] - pos_b[2]) ** 2
        )
    
    @property
    def total_hypotheses(self) -> int:
        """Total number of hypotheses."""
        return len(self._hypotheses)
    
    @property
    def pending_count(self) -> int:
        """Number of pending hypotheses."""
        return len(self._by_status.get("pending", set()))
    
    @property
    def confirmed_count(self) -> int:
        """Number of confirmed hypotheses."""
        return len(self._by_status.get("confirmed", set()))


# =============================================================================
# CUED RECALL MODULE
# =============================================================================

class CuedRecallModule:
    """Multi-cue memory retrieval with salience weighting.
    
    ARCHITECTURAL INVARIANT: Memory retrieval combines multiple cue types
    with tunable weights at query time. High-salience moments surface first.
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        episode_store: EpisodeStore,
        cue_store: RedundantCueStore | None = None,
        hypothesis_tracker: EntityHypothesisTracker | None = None,
    ):
        """Initialize the cued recall module.
        
        Args:
            graph_store: Graph memory storage
            episode_store: Episode memory storage
            cue_store: Redundant cue storage (created if not provided)
            hypothesis_tracker: Entity hypothesis tracker (created if not provided)
        """
        self._graph = graph_store
        self._episodes = episode_store
        self._cues = cue_store or RedundantCueStore()
        self._hypotheses = hypothesis_tracker or EntityHypothesisTracker()
        
        # Location visit tracking for revisit detection
        self._last_visit: dict[str, datetime] = {}
        self._visit_count: dict[str, int] = defaultdict(int)
        
        # Callbacks for recall events
        self._on_recall_callbacks: list[Callable[[RecallResult], None]] = []
        self._on_revisit_callbacks: list[Callable[[LocationRevisit], None]] = []
    
    def recall(self, query: CuedRecallQuery) -> RecallResult:
        """Perform cued recall based on query.
        
        Combines salience weights from multiple cue types to retrieve
        relevant memories, with high-salience moments surfacing first.
        
        Args:
            query: The recall query with cues and weights
            
        Returns:
            RecallResult with retrieved memories and salience scores
        """
        start_time = time.time()
        
        # Get effective weights for this query
        weights = query.get_effective_weights()
        active_cues = query.get_active_cue_types()
        
        # Collect candidate memories with salience scores
        candidates: dict[str, float] = {}  # node_id -> combined_score
        cue_contributions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # 1. Location cue retrieval
        if CueType.LOCATION in active_cues:
            self._retrieve_by_location_cue(
                query, weights, candidates, cue_contributions
            )
        
        # 2. Entity cue retrieval
        if CueType.ENTITY in active_cues:
            self._retrieve_by_entity_cue(
                query, weights, candidates, cue_contributions
            )
        
        # 3. Semantic cue retrieval (labels)
        if CueType.SEMANTIC in active_cues:
            self._retrieve_by_semantic_cue(
                query, weights, candidates, cue_contributions
            )
        
        # 4. Visual cue retrieval
        if CueType.VISUAL in active_cues:
            self._retrieve_by_visual_cue(
                query, weights, candidates, cue_contributions
            )
        
        # 5. Event cue retrieval
        if CueType.EVENT in active_cues:
            self._retrieve_by_event_cue(
                query, weights, candidates, cue_contributions
            )
        
        # Filter by minimum salience
        candidates = {
            nid: score for nid, score in candidates.items()
            if score >= query.min_salience
        }
        
        # Sort by salience if requested
        if query.prioritize_salient:
            sorted_candidates = sorted(
                candidates.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            sorted_candidates = list(candidates.items())
        
        # Limit results
        sorted_candidates = sorted_candidates[:query.max_results]
        
        # Separate nodes and episodes
        node_ids = []
        node_salience = []
        episode_ids = []
        episode_salience = []
        contributions = []
        
        for node_id, score in sorted_candidates:
            node = self._graph.get_node(node_id)
            if node:
                if node.node_type == NODE_TYPE_EPISODE:
                    # Extract episode_id from node_id
                    ep_id = node.source_id or node_id.replace(f"{NODE_TYPE_EPISODE}_", "")
                    episode_ids.append(ep_id)
                    episode_salience.append(score)
                else:
                    node_ids.append(node_id)
                    node_salience.append(score)
                contributions.append(dict(cue_contributions.get(node_id, {})))
        
        query_time_ms = (time.time() - start_time) * 1000
        
        result = RecallResult(
            query_id=query.query_id,
            node_ids=node_ids,
            episode_ids=episode_ids,
            node_salience=node_salience,
            episode_salience=episode_salience,
            cue_contributions=contributions,
            query_time_ms=query_time_ms,
            total_candidates=len(candidates),
        )
        
        # Notify callbacks
        for callback in self._on_recall_callbacks:
            callback(result)
        
        return result
    
    def on_location_enter(
        self,
        location_id: str,
        location_label: str | None = None,
    ) -> LocationRevisit | None:
        """Handle entering a location.
        
        If this is a revisit (previously visited), triggers cued recall
        to surface memories from prior visits.
        
        Args:
            location_id: ID of the location entered
            location_label: Optional label for the location
            
        Returns:
            LocationRevisit if this is a revisit, None otherwise
        """
        now = datetime.now()
        last_visit = self._last_visit.get(location_id)
        visit_count = self._visit_count[location_id]
        
        # Update visit tracking
        self._visit_count[location_id] += 1
        
        # Check if this is a revisit
        if last_visit is None or (now - last_visit).total_seconds() < REVISIT_TIME_THRESHOLD:
            # First visit or too recent, not a revisit
            self._last_visit[location_id] = now
            return None
        
        # This is a revisit - trigger cued recall
        self._last_visit[location_id] = now
        
        # Recall memories from this location
        query = CuedRecallQuery(
            location_id=location_id,
            location_label=location_label,
            max_results=MAX_RECALLED_EPISODES,
            prioritize_salient=True,
            min_salience=MIN_SALIENCE_TO_RECALL,
        )
        
        recall_result = self.recall(query)
        
        # Find high-salience moments
        salient_moments = []
        for i, salience in enumerate(recall_result.episode_salience):
            if salience >= HIGH_SALIENCE_THRESHOLD:
                salient_moments.append(recall_result.episode_ids[i])
        
        # Get entity hypotheses for this location
        hypotheses = self._hypotheses.get_hypotheses_for_location(location_id)
        hypothesis_ids = [h.hypothesis_id for h in hypotheses if h.status == "pending"]
        
        # Create revisit record
        revisit = LocationRevisit(
            location_id=location_id,
            location_label=location_label,
            current_timestamp=now,
            prior_visit_count=visit_count,
            recalled_episode_ids=recall_result.episode_ids,
            recalled_salience=recall_result.episode_salience,
            salient_moments=salient_moments,
            entity_hypotheses=hypothesis_ids,
        )
        
        # Notify callbacks
        for callback in self._on_revisit_callbacks:
            callback(revisit)
        
        return revisit
    
    def record_entity_observation(
        self,
        observation_id: str,
        location_id: str,
        position: tuple[float, float, float],
        label: str | None = None,
        visual_embedding: list[float] | None = None,
    ) -> list[EntityHypothesis]:
        """Record an entity observation for same-entity tracking.
        
        Args:
            observation_id: Unique observation ID
            location_id: Location where observed
            position: Relative position (agent-centric)
            label: Optional label
            visual_embedding: Optional visual features
            
        Returns:
            Any same-entity hypotheses generated
        """
        return self._hypotheses.record_observation(
            observation_id=observation_id,
            location_id=location_id,
            position=position,
            label=label,
            visual_embedding=visual_embedding,
        )
    
    def create_memory_cues(
        self,
        memory_node_id: str,
        location_id: str | None = None,
        entity_ids: list[str] | None = None,
        labels: list[str] | None = None,
        episode_id: str | None = None,
    ) -> list[RedundantCue]:
        """Create redundant cues for a memory.
        
        Args:
            memory_node_id: The memory node to create cues for
            location_id: Associated location
            entity_ids: Associated entities
            labels: Associated labels
            episode_id: Associated episode
            
        Returns:
            Created cues
        """
        return self._cues.create_cues_for_memory(
            target_node_id=memory_node_id,
            location_id=location_id,
            entity_ids=entity_ids,
            labels=labels,
            timestamp=datetime.now(),
            episode_id=episode_id,
        )
    
    def boost_edge_salience(
        self,
        edge_id: str,
        cue_type: CueType,
        amount: float = 0.1,
    ) -> bool:
        """Boost salience of an edge for a cue type.
        
        Args:
            edge_id: Edge to boost
            cue_type: Which cue type to boost
            amount: Amount to boost (0.0-1.0)
            
        Returns:
            True if boosted, False if edge not found
        """
        edge = self._graph.get_edge(edge_id)
        if not edge:
            return False
        
        edge.boost_salience(cue_type.value, amount)
        self._graph.update_edge(edge)
        return True
    
    def register_recall_callback(
        self,
        callback: Callable[[RecallResult], None],
    ) -> None:
        """Register callback for recall events."""
        self._on_recall_callbacks.append(callback)
    
    def register_revisit_callback(
        self,
        callback: Callable[[LocationRevisit], None],
    ) -> None:
        """Register callback for revisit events."""
        self._on_revisit_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # PRIVATE: Retrieval by cue type
    # -------------------------------------------------------------------------
    
    def _retrieve_by_location_cue(
        self,
        query: CuedRecallQuery,
        weights: dict[CueType, float],
        candidates: dict[str, float],
        contributions: dict[str, dict[str, float]],
    ) -> None:
        """Retrieve memories cued by location."""
        location_weight = weights.get(CueType.LOCATION, 0.25)
        
        location_id = query.location_id
        if not location_id and query.location_label:
            # Try to find location by label
            nodes = self._graph.get_nodes_by_label(query.location_label)
            for node in nodes:
                if node.node_type == NODE_TYPE_LOCATION:
                    location_id = node.source_id or node.node_id
                    break
        
        if not location_id:
            return
        
        # Get cues from location
        cues = self._cues.get_cues_by_location(location_id)
        for cue in cues:
            score = cue.strength * location_weight
            candidates[cue.target_node_id] = (
                candidates.get(cue.target_node_id, 0) + score
            )
            contributions[cue.target_node_id]["location"] += score
            cue.access()
        
        # Also get directly connected nodes
        location_node_id = f"{NODE_TYPE_LOCATION}_{location_id}"
        edges = self._graph.get_edges_from_node(location_node_id)
        for edge in edges:
            if edge.edge_type in (EDGE_TYPE_CONTAINS, EDGE_TYPE_OCCURRED_IN):
                salience = edge.get_salience_score("location")
                score = max(salience, 0.1) * location_weight
                candidates[edge.target_node_id] = (
                    candidates.get(edge.target_node_id, 0) + score
                )
                contributions[edge.target_node_id]["location"] += score
    
    def _retrieve_by_entity_cue(
        self,
        query: CuedRecallQuery,
        weights: dict[CueType, float],
        candidates: dict[str, float],
        contributions: dict[str, dict[str, float]],
    ) -> None:
        """Retrieve memories cued by entities."""
        entity_weight = weights.get(CueType.ENTITY, 0.20)
        
        for entity_id in query.entity_ids:
            # Get cues from entity
            cues = self._cues.get_cues_by_entity(entity_id)
            for cue in cues:
                score = cue.strength * entity_weight
                candidates[cue.target_node_id] = (
                    candidates.get(cue.target_node_id, 0) + score
                )
                contributions[cue.target_node_id]["entity"] += score
                cue.access()
            
            # Also get directly connected nodes
            entity_node_id = f"{NODE_TYPE_ENTITY}_{entity_id}"
            edges = self._graph.get_edges_from_node(entity_node_id)
            for edge in edges:
                if edge.edge_type in (EDGE_TYPE_IN_CONTEXT, EDGE_TYPE_SIMILAR_TO):
                    salience = edge.get_salience_score("entity")
                    score = max(salience, 0.1) * entity_weight
                    candidates[edge.target_node_id] = (
                        candidates.get(edge.target_node_id, 0) + score
                    )
                    contributions[edge.target_node_id]["entity"] += score
    
    def _retrieve_by_semantic_cue(
        self,
        query: CuedRecallQuery,
        weights: dict[CueType, float],
        candidates: dict[str, float],
        contributions: dict[str, dict[str, float]],
    ) -> None:
        """Retrieve memories cued by semantic labels."""
        semantic_weight = weights.get(CueType.SEMANTIC, 0.20)
        
        for label in query.labels:
            # Get cues from label
            cues = self._cues.get_cues_by_label(label)
            for cue in cues:
                score = cue.strength * semantic_weight
                candidates[cue.target_node_id] = (
                    candidates.get(cue.target_node_id, 0) + score
                )
                contributions[cue.target_node_id]["semantic"] += score
                cue.access()
            
            # Also get nodes by label
            nodes = self._graph.get_nodes_by_label(label)
            for node in nodes:
                # Get edges from this node
                edges = self._graph.get_edges_from_node(node.node_id)
                for edge in edges:
                    salience = edge.get_salience_score("semantic")
                    score = max(salience, 0.1) * semantic_weight
                    candidates[edge.target_node_id] = (
                        candidates.get(edge.target_node_id, 0) + score
                    )
                    contributions[edge.target_node_id]["semantic"] += score
    
    def _retrieve_by_visual_cue(
        self,
        query: CuedRecallQuery,
        weights: dict[CueType, float],
        candidates: dict[str, float],
        contributions: dict[str, dict[str, float]],
    ) -> None:
        """Retrieve memories cued by visual similarity."""
        visual_weight = weights.get(CueType.VISUAL, 0.10)
        
        if not query.visual_embedding:
            return
        
        # Get similar nodes by embedding
        similar = self._graph.get_similar_nodes(query.visual_embedding, max_results=20)
        for node, similarity in similar:
            score = similarity * visual_weight
            candidates[node.node_id] = (
                candidates.get(node.node_id, 0) + score
            )
            contributions[node.node_id]["visual"] += score
    
    def _retrieve_by_event_cue(
        self,
        query: CuedRecallQuery,
        weights: dict[CueType, float],
        candidates: dict[str, float],
        contributions: dict[str, dict[str, float]],
    ) -> None:
        """Retrieve memories cued by event type."""
        event_weight = weights.get(CueType.EVENT, 0.10)
        
        if not query.event_type:
            return
        
        # Get cues by event type (stored as label)
        cues = self._cues.get_cues_by_label(query.event_type)
        for cue in cues:
            if cue.cue_type == CueType.EVENT:
                score = cue.strength * event_weight
                candidates[cue.target_node_id] = (
                    candidates.get(cue.target_node_id, 0) + score
                )
                contributions[cue.target_node_id]["event"] += score
                cue.access()
    
    # -------------------------------------------------------------------------
    # PROPERTIES
    # -------------------------------------------------------------------------
    
    @property
    def cue_store(self) -> RedundantCueStore:
        """Access the cue store."""
        return self._cues
    
    @property
    def hypothesis_tracker(self) -> EntityHypothesisTracker:
        """Access the hypothesis tracker."""
        return self._hypotheses
    
    @property
    def total_cues(self) -> int:
        """Total redundant cues."""
        return self._cues.total_cues
    
    @property
    def total_hypotheses(self) -> int:
        """Total entity hypotheses."""
        return self._hypotheses.total_hypotheses


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "MIN_SALIENCE_TO_RECALL",
    "HIGH_SALIENCE_THRESHOLD",
    "SAME_POSITION_THRESHOLD",
    "VISUAL_SIMILARITY_THRESHOLD",
    "HYPOTHESIS_CONFIDENCE_THRESHOLD",
    "REVISIT_TIME_THRESHOLD",
    "MAX_RECALLED_EPISODES",
    "MAX_CUES_PER_TARGET",
    # Classes
    "RedundantCueStore",
    "EntityHypothesisTracker",
    "CuedRecallModule",
]
