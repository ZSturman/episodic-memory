"""Memory integration module for emergent knowledge architecture.

ARCHITECTURAL INVARIANT: Memory retrieval uses learned labels and spatial context.
No predefined semantic categories - all meaning emerges from user interaction.

This module provides:
- MemoryIntegrator: Combines label learning, spatial context, and graph memory
- Enhanced retrieval using spreading activation with learned labels
- Spatial context-aware memory queries
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from episodic_agent.core.interfaces import EpisodeStore, GraphStore
from episodic_agent.modules.label_learner import LabelLearner
from episodic_agent.modules.landmark_manager import LandmarkManager
from episodic_agent.schemas import (
    Episode,
    GraphEdge,
    GraphNode,
    RetrievalResult,
)
from episodic_agent.schemas.graph import (
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_IN_CONTEXT,
    EDGE_TYPE_OCCURRED_IN,
    EDGE_TYPE_PART_OF,
    EDGE_TYPE_SIMILAR_TO,
    EDGE_TYPE_SPATIAL,
    EDGE_TYPE_TEMPORAL,
    EDGE_TYPE_TYPICAL_IN,
    NODE_TYPE_ENTITY,
    NODE_TYPE_EPISODE,
    NODE_TYPE_EVENT,
    NODE_TYPE_LOCATION,
    EdgeType,
    NodeType,
)
from episodic_agent.schemas.learning import (
    CATEGORY_ENTITY,
    CATEGORY_EVENT,
    CATEGORY_LOCATION,
    LearnedLabel,
    RecognitionResult,
)
from episodic_agent.schemas.spatial import (
    LandmarkReference,
    RelativePosition,
    SpatialRelation,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Spreading activation parameters
DEFAULT_DECAY = 0.7              # Activation decay per hop
MAX_ACTIVATION_DEPTH = 3         # Maximum hops for spreading
MIN_ACTIVATION = 0.01            # Minimum activation to continue
RECENCY_WEIGHT = 0.3             # Weight for recency in retrieval
FREQUENCY_WEIGHT = 0.2           # Weight for access frequency
SIMILARITY_WEIGHT = 0.3          # Weight for embedding similarity
ACTIVATION_WEIGHT = 0.2          # Weight for spreading activation

# Spatial retrieval parameters
SPATIAL_BOOST = 1.5              # Boost for spatially relevant results
SAME_LOCATION_BOOST = 2.0        # Boost for same location results


# =============================================================================
# MEMORY QUERY
# =============================================================================

class MemoryQuery:
    """A query for retrieving from memory.
    
    Supports multiple retrieval cues:
    - Labels: Match against learned labels
    - Location: Filter by spatial context
    - Embedding: Similarity search
    - Temporal: Recency filtering
    """
    
    def __init__(
        self,
        query_id: str | None = None,
        labels: list[str] | None = None,
        location_id: str | None = None,
        location_label: str | None = None,
        relative_position: RelativePosition | None = None,
        embedding: list[float] | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        max_results: int = 10,
        include_episodes: bool = True,
        include_nodes: bool = True,
        node_types: list[str] | None = None,
    ):
        self.query_id = query_id or str(uuid.uuid4())
        self.labels = labels or []
        self.location_id = location_id
        self.location_label = location_label
        self.relative_position = relative_position
        self.embedding = embedding
        self.time_start = time_start
        self.time_end = time_end
        self.max_results = max_results
        self.include_episodes = include_episodes
        self.include_nodes = include_nodes
        self.node_types = node_types


# =============================================================================
# MEMORY INTEGRATOR
# =============================================================================

class MemoryIntegrator:
    """Integrates label learning, spatial context, and graph memory.
    
    This is the core component that brings together:
    - LabelLearner: For emergent vocabulary
    - LandmarkManager: For spatial context
    - GraphStore: For associative memory
    - EpisodeStore: For episodic memory
    
    Provides enhanced retrieval that uses learned labels and spatial
    context to find relevant memories.
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        episode_store: EpisodeStore,
        label_learner: LabelLearner | None = None,
        landmark_manager: LandmarkManager | None = None,
    ):
        """Initialize the memory integrator.
        
        Args:
            graph_store: Graph memory storage
            episode_store: Episode memory storage
            label_learner: Label learning component (optional)
            landmark_manager: Spatial context component (optional)
        """
        self._graph = graph_store
        self._episodes = episode_store
        self._learner = label_learner or LabelLearner()
        self._landmarks = landmark_manager or LandmarkManager()
        
        # Index: label -> node IDs
        self._label_to_nodes: dict[str, set[str]] = defaultdict(set)
        
        # Index: location_id -> episode IDs
        self._location_to_episodes: dict[str, set[str]] = defaultdict(set)
        
        # Build indexes from existing data
        self._build_indexes()
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Integration
    # -------------------------------------------------------------------------
    
    def integrate_observation(
        self,
        observation_type: str,
        observation_id: str,
        label: str | None = None,
        location_id: str | None = None,
        position: tuple[float, float, float] | None = None,
        features: dict[str, Any] | None = None,
    ) -> GraphNode | None:
        """Integrate a new observation into memory.
        
        This is the primary entry point for new observations. It:
        1. Attempts to recognize using learned labels
        2. Creates or updates graph node
        3. Records spatial context
        
        Args:
            observation_type: Type of observation (entity, location, event)
            observation_id: Unique ID of the observation
            label: Optional user-provided label
            location_id: Optional current location
            position: Optional absolute position
            features: Optional features for recognition
            
        Returns:
            The created/updated graph node
        """
        # Map observation type to category and node type
        category = self._type_to_category(observation_type)
        node_type = self._type_to_node_type(observation_type)
        
        # Try to recognize using learned labels
        recognition: RecognitionResult | None = None
        if features and not label:
            recognition = self._learner.recognize(
                instance_id=observation_id,
                category=category,
                features=features or {},
            )
            if recognition.recognized and recognition.confidence > 0.7:
                label = recognition.best_label
        
        # Use provided label or recognition result
        final_label = label or (recognition.best_label if recognition else None) or "unknown"
        
        # Learn the label if user provided it
        if label:
            self._learner.learn_label(
                label,
                category=category,
                instance_id=observation_id,
                features=features,
            )
        
        # Create or get graph node
        node = self._get_or_create_node(
            node_id=f"{node_type}_{observation_id}",
            node_type=node_type,
            label=final_label,
            source_id=observation_id,
        )
        
        # Record spatial context
        if location_id and position:
            self._record_spatial_context(
                node_id=node.node_id,
                location_id=location_id,
                position=position,
            )
        
        # Update indexes
        self._label_to_nodes[final_label.lower()].add(node.node_id)
        
        return node
    
    def integrate_episode(
        self,
        episode: Episode,
        location_id: str | None = None,
    ) -> GraphNode:
        """Integrate an episode into memory.
        
        Creates an episode node and links it to:
        - Location (if known)
        - Entities mentioned
        - Events that occurred
        
        Args:
            episode: The episode to integrate
            location_id: Optional location ID
            
        Returns:
            The episode graph node
        """
        # Store the episode
        self._episodes.store(episode)
        
        # Create episode node
        episode_node = self._get_or_create_node(
            node_id=f"{NODE_TYPE_EPISODE}_{episode.episode_id}",
            node_type=NODE_TYPE_EPISODE,
            label=episode.location_label or "unknown",
            source_id=episode.episode_id,
            embedding=episode.episode_embedding,
        )
        
        # Link to location
        if location_id:
            self._create_edge(
                source_id=episode_node.node_id,
                target_id=f"{NODE_TYPE_LOCATION}_{location_id}",
                edge_type=EDGE_TYPE_OCCURRED_IN,
            )
            self._location_to_episodes[location_id].add(episode.episode_id)
        
        # Link to entities
        for entity in episode.entities:
            entity_node_id = f"{NODE_TYPE_ENTITY}_{entity.guid}"
            self._create_edge(
                source_id=episode_node.node_id,
                target_id=entity_node_id,
                edge_type=EDGE_TYPE_IN_CONTEXT,
            )
        
        # Learn location label if provided
        if episode.location_label and episode.location_label != "unknown":
            self._learner.learn_label(
                episode.location_label,
                category=CATEGORY_LOCATION,
                instance_id=location_id or episode.episode_id,
            )
        
        return episode_node
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Retrieval
    # -------------------------------------------------------------------------
    
    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve memories matching a query.
        
        Uses multiple retrieval strategies:
        1. Label matching (against learned vocabulary)
        2. Spatial filtering (by location/position)
        3. Embedding similarity
        4. Spreading activation
        
        Args:
            query: The memory query
            
        Returns:
            RetrievalResult with matched memories
        """
        # Collect candidate nodes
        candidate_nodes: dict[str, float] = {}
        
        # 1. Label-based retrieval
        if query.labels:
            self._retrieve_by_labels(query.labels, candidate_nodes)
        
        # 2. Location-based retrieval
        if query.location_id or query.location_label:
            self._retrieve_by_location(
                query.location_id,
                query.location_label,
                candidate_nodes,
            )
        
        # 3. Spatial context retrieval
        if query.relative_position:
            self._retrieve_by_spatial(query.relative_position, candidate_nodes)
        
        # 4. Embedding similarity
        if query.embedding:
            self._retrieve_by_embedding(query.embedding, candidate_nodes)
        
        # 5. If no specific cues, use recent/frequent
        if not candidate_nodes:
            self._retrieve_recent_frequent(candidate_nodes)
        
        # 6. Spreading activation from candidates
        if candidate_nodes:
            self._spread_activation(candidate_nodes)
        
        # Filter by node types if specified
        if query.node_types:
            candidate_nodes = {
                nid: score for nid, score in candidate_nodes.items()
                if self._graph.get_node(nid) and 
                   self._graph.get_node(nid).node_type in query.node_types
            }
        
        # Sort by score
        sorted_nodes = sorted(
            candidate_nodes.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:query.max_results]
        
        # Build result
        nodes = []
        node_scores = []
        for node_id, score in sorted_nodes:
            node = self._graph.get_node(node_id)
            if node and query.include_nodes:
                nodes.append(node)
                node_scores.append(score)
        
        # Collect episodes
        episodes = []
        episode_scores = []
        if query.include_episodes:
            episode_ids = self._get_related_episodes(
                [n[0] for n in sorted_nodes],
                query,
            )
            for ep_id, score in episode_ids[:query.max_results]:
                ep = self._episodes.get(ep_id)
                if ep:
                    episodes.append(ep)
                    episode_scores.append(score)
        
        return RetrievalResult(
            query_id=query.query_id,
            nodes=nodes,
            node_scores=node_scores,
            episodes=episodes,
            episode_scores=episode_scores,
        )
    
    def retrieve_by_label(
        self,
        label: str,
        max_results: int = 10,
    ) -> RetrievalResult:
        """Convenience method to retrieve by a single label.
        
        Args:
            label: The label to search for
            max_results: Maximum results to return
            
        Returns:
            RetrievalResult with matched memories
        """
        return self.retrieve(MemoryQuery(
            labels=[label],
            max_results=max_results,
        ))
    
    def retrieve_by_location(
        self,
        location_id: str | None = None,
        location_label: str | None = None,
        max_results: int = 10,
    ) -> RetrievalResult:
        """Convenience method to retrieve by location.
        
        Args:
            location_id: Location GUID
            location_label: Location label
            max_results: Maximum results to return
            
        Returns:
            RetrievalResult with matched memories
        """
        return self.retrieve(MemoryQuery(
            location_id=location_id,
            location_label=location_label,
            max_results=max_results,
        ))
    
    def retrieve_similar(
        self,
        node_id: str,
        max_results: int = 10,
    ) -> RetrievalResult:
        """Retrieve nodes similar to a given node.
        
        Uses spreading activation from the given node.
        
        Args:
            node_id: ID of the seed node
            max_results: Maximum results to return
            
        Returns:
            RetrievalResult with similar memories
        """
        node = self._graph.get_node(node_id)
        if not node:
            return RetrievalResult(query_id=str(uuid.uuid4()))
        
        return self.retrieve(MemoryQuery(
            labels=[node.label] if node.label != "unknown" else [],
            embedding=node.embedding,
            max_results=max_results,
        ))
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Spatial Context
    # -------------------------------------------------------------------------
    
    def get_nearby_memories(
        self,
        location_id: str,
        position: tuple[float, float, float],
        radius: float = 5.0,
        max_results: int = 10,
    ) -> RetrievalResult:
        """Get memories near a spatial position.
        
        Args:
            location_id: Current location
            position: Current position
            radius: Search radius
            max_results: Maximum results
            
        Returns:
            RetrievalResult with nearby memories
        """
        # Get landmarks in location
        landmarks = self._landmarks.get_landmarks_in_location(location_id)
        
        candidate_nodes: dict[str, float] = {}
        
        for landmark in landmarks:
            # Compute distance
            if not landmark.internal_position:
                continue
            lm_pos = landmark.internal_position
            dist = self._compute_distance(position, lm_pos)
            
            if dist <= radius:
                # Score inversely proportional to distance
                score = 1.0 / (1.0 + dist)
                
                # Find associated node
                node_id = f"{NODE_TYPE_ENTITY}_{landmark.landmark_id}"
                if self._graph.get_node(node_id):
                    candidate_nodes[node_id] = max(
                        candidate_nodes.get(node_id, 0),
                        score * SPATIAL_BOOST,
                    )
        
        # Spread activation
        self._spread_activation(candidate_nodes)
        
        # Sort and return
        sorted_nodes = sorted(
            candidate_nodes.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_results]
        
        nodes = []
        scores = []
        for node_id, score in sorted_nodes:
            node = self._graph.get_node(node_id)
            if node:
                nodes.append(node)
                scores.append(score)
        
        return RetrievalResult(
            query_id=str(uuid.uuid4()),
            nodes=nodes,
            node_scores=scores,
        )
    
    def what_is_here(
        self,
        location_id: str,
        position: tuple[float, float, float] | None = None,
    ) -> list[tuple[str, RecognitionResult]]:
        """Get recognized entities at a location.
        
        Uses learned labels to describe what's here.
        
        Args:
            location_id: Current location
            position: Optional position for relative descriptions
            
        Returns:
            List of (description, recognition) tuples
        """
        results = []
        
        # Get entities in this location
        location_node_id = f"{NODE_TYPE_LOCATION}_{location_id}"
        edges = self._graph.get_edges(location_node_id)
        
        for edge in edges:
            if edge.edge_type in [EDGE_TYPE_CONTAINS, EDGE_TYPE_TYPICAL_IN]:
                entity_id = edge.source_node_id if edge.target_node_id == location_node_id else edge.target_node_id
                entity_node = self._graph.get_node(entity_id)
                
                if entity_node:
                    # Get recognition result
                    recognition = RecognitionResult(
                        instance_id=entity_node.source_id or entity_id,
                        category=CATEGORY_ENTITY,
                        recognized=entity_node.label != "unknown",
                        best_label=entity_node.label,
                        confidence=entity_node.confidence,
                    )
                    
                    # Build description
                    description = entity_node.label
                    
                    results.append((description, recognition))
        
        return results
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Learning Integration
    # -------------------------------------------------------------------------
    
    def learn_label_for_node(
        self,
        node_id: str,
        label: str,
        features: dict[str, Any] | None = None,
    ) -> bool:
        """Learn a label for an existing node.
        
        Updates the node's label and records the learning.
        
        Args:
            node_id: ID of the node
            label: Label to learn
            features: Optional features for the label
            
        Returns:
            True if successful
        """
        node = self._graph.get_node(node_id)
        if not node:
            return False
        
        # Determine category from node type
        category = self._node_type_to_category(node.node_type)
        
        # Learn the label
        self._learner.learn_label(
            label,
            category=category,
            instance_id=node.source_id or node_id,
            features=features,
        )
        
        # Update node label
        old_label = node.label
        node.label = label
        
        # Update indexes
        if old_label and old_label.lower() in self._label_to_nodes:
            self._label_to_nodes[old_label.lower()].discard(node_id)
        self._label_to_nodes[label.lower()].add(node_id)
        
        return True
    
    def confirm_recognition(self, node_id: str) -> bool:
        """Confirm that a node's label is correct.
        
        Args:
            node_id: ID of the node
            
        Returns:
            True if successful
        """
        node = self._graph.get_node(node_id)
        if not node or node.label == "unknown":
            return False
        
        self._learner.confirm_label(node.label)
        
        # Increase node confidence
        node.confidence = min(1.0, node.confidence + 0.1)
        
        return True
    
    def correct_recognition(
        self,
        node_id: str,
        correct_label: str,
        features: dict[str, Any] | None = None,
    ) -> bool:
        """Correct a node's label.
        
        Args:
            node_id: ID of the node
            correct_label: The correct label
            features: Optional features
            
        Returns:
            True if successful
        """
        node = self._graph.get_node(node_id)
        if not node:
            return False
        
        old_label = node.label
        
        # Correct in learner
        self._learner.correct_label(
            old_label,
            correct_label,
            instance_id=node.source_id or node_id,
            features=features,
        )
        
        # Update node
        node.label = correct_label
        
        # Update indexes
        if old_label and old_label.lower() in self._label_to_nodes:
            self._label_to_nodes[old_label.lower()].discard(node_id)
        self._label_to_nodes[correct_label.lower()].add(node_id)
        
        return True
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Statistics
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> dict[str, Any]:
        """Get integrated memory statistics.
        
        Returns:
            Dictionary of statistics
        """
        learner_stats = self._learner.get_statistics()
        
        return {
            "graph": {
                "node_count": len(self._graph.get_all_nodes()),
                "edge_count": len(self._graph.get_all_edges()),
            },
            "episodes": {
                "count": self._episodes.count(),
            },
            "learning": learner_stats,
            "spatial": {
                "landmark_count": len(self._landmarks.get_all_landmarks()),
            },
            "indexes": {
                "labels_indexed": len(self._label_to_nodes),
                "locations_indexed": len(self._location_to_episodes),
            },
        }
    
    # -------------------------------------------------------------------------
    # INTERNAL: Retrieval helpers
    # -------------------------------------------------------------------------
    
    def _retrieve_by_labels(
        self,
        labels: list[str],
        candidates: dict[str, float],
    ) -> None:
        """Retrieve nodes matching labels."""
        for label in labels:
            # Check learned labels
            learned = self._learner.get_label(label)
            if learned:
                # Boost score by learner confidence
                boost = 1.0 + learned.confidence
            else:
                boost = 1.0
            
            # Find nodes with this label
            node_ids = self._label_to_nodes.get(label.lower(), set())
            for node_id in node_ids:
                candidates[node_id] = max(
                    candidates.get(node_id, 0),
                    boost,
                )
            
            # Also check graph store's label lookup
            for node in self._graph.get_all_nodes():
                if node.label.lower() == label.lower():
                    candidates[node.node_id] = max(
                        candidates.get(node.node_id, 0),
                        boost,
                    )
                if label.lower() in [l.lower() for l in node.labels]:
                    candidates[node.node_id] = max(
                        candidates.get(node.node_id, 0),
                        boost * 0.8,  # Slightly lower for alternative labels
                    )
    
    def _retrieve_by_location(
        self,
        location_id: str | None,
        location_label: str | None,
        candidates: dict[str, float],
    ) -> None:
        """Retrieve nodes in a location."""
        # Find location node
        loc_node_id = None
        if location_id:
            loc_node_id = f"{NODE_TYPE_LOCATION}_{location_id}"
        elif location_label:
            # Find by label
            for node in self._graph.get_all_nodes():
                if (node.node_type == NODE_TYPE_LOCATION and 
                    node.label.lower() == location_label.lower()):
                    loc_node_id = node.node_id
                    break
        
        if not loc_node_id:
            return
        
        # Find connected nodes
        edges = self._graph.get_edges(loc_node_id)
        for edge in edges:
            other_id = (
                edge.source_node_id 
                if edge.target_node_id == loc_node_id 
                else edge.target_node_id
            )
            candidates[other_id] = max(
                candidates.get(other_id, 0),
                SAME_LOCATION_BOOST,
            )
        
        # Also add the location node itself
        candidates[loc_node_id] = max(
            candidates.get(loc_node_id, 0),
            SAME_LOCATION_BOOST,
        )
    
    def _retrieve_by_spatial(
        self,
        relative_position: RelativePosition,
        candidates: dict[str, float],
    ) -> None:
        """Retrieve by spatial context."""
        # Find landmarks matching the relative position description
        if relative_position.reference_landmark:
            lm = self._landmarks.get_landmark(relative_position.reference_landmark)
            if lm:
                node_id = f"{NODE_TYPE_ENTITY}_{lm.landmark_id}"
                if self._graph.get_node(node_id):
                    candidates[node_id] = max(
                        candidates.get(node_id, 0),
                        SPATIAL_BOOST,
                    )
    
    def _retrieve_by_embedding(
        self,
        embedding: list[float],
        candidates: dict[str, float],
    ) -> None:
        """Retrieve by embedding similarity."""
        for node in self._graph.get_all_nodes():
            if node.embedding:
                similarity = self._cosine_similarity(embedding, node.embedding)
                if similarity > 0.5:
                    candidates[node.node_id] = max(
                        candidates.get(node.node_id, 0),
                        similarity * SIMILARITY_WEIGHT,
                    )
    
    def _retrieve_recent_frequent(
        self,
        candidates: dict[str, float],
    ) -> None:
        """Retrieve recent and frequently accessed nodes."""
        now = datetime.now()
        
        for node in self._graph.get_all_nodes():
            # Recency score
            age_hours = (now - node.last_accessed).total_seconds() / 3600
            recency = 1.0 / (1.0 + age_hours)
            
            # Frequency score
            frequency = math.log(1 + node.access_count) / 10
            
            score = recency * RECENCY_WEIGHT + frequency * FREQUENCY_WEIGHT
            if score > 0.1:
                candidates[node.node_id] = max(
                    candidates.get(node.node_id, 0),
                    score,
                )
    
    def _spread_activation(
        self,
        candidates: dict[str, float],
        depth: int = 0,
    ) -> None:
        """Spread activation through the graph."""
        if depth >= MAX_ACTIVATION_DEPTH:
            return
        
        # Get nodes to spread from (above threshold)
        to_spread = [
            (nid, score) for nid, score in candidates.items()
            if score > MIN_ACTIVATION
        ]
        
        new_activations: dict[str, float] = {}
        
        for node_id, activation in to_spread:
            # Get connected nodes
            edges = self._graph.get_edges(node_id)
            
            for edge in edges:
                # Determine neighbor
                neighbor_id = (
                    edge.source_node_id 
                    if edge.target_node_id == node_id 
                    else edge.target_node_id
                )
                
                # Compute spread activation
                spread = activation * DEFAULT_DECAY * edge.weight
                
                if spread > MIN_ACTIVATION:
                    new_activations[neighbor_id] = max(
                        new_activations.get(neighbor_id, 0),
                        spread,
                    )
        
        # Add new activations
        for node_id, activation in new_activations.items():
            candidates[node_id] = max(
                candidates.get(node_id, 0),
                activation * ACTIVATION_WEIGHT,
            )
        
        # Recurse if we found new nodes
        if new_activations and depth < MAX_ACTIVATION_DEPTH - 1:
            self._spread_activation(candidates, depth + 1)
    
    def _get_related_episodes(
        self,
        node_ids: list[str],
        query: MemoryQuery,
    ) -> list[tuple[str, float]]:
        """Get episodes related to the given nodes."""
        episode_scores: dict[str, float] = {}
        
        # Get episodes from location index
        if query.location_id:
            for ep_id in self._location_to_episodes.get(query.location_id, set()):
                episode_scores[ep_id] = max(
                    episode_scores.get(ep_id, 0),
                    SAME_LOCATION_BOOST,
                )
        
        # Get episodes connected to nodes
        for node_id in node_ids:
            node = self._graph.get_node(node_id)
            if node and node.node_type == NODE_TYPE_EPISODE:
                episode_scores[node.source_id or node_id] = max(
                    episode_scores.get(node.source_id or node_id, 0),
                    1.0,
                )
            
            # Check edges for episode connections
            edges = self._graph.get_edges(node_id)
            for edge in edges:
                if edge.edge_type == EDGE_TYPE_PART_OF:
                    ep_node_id = edge.target_node_id
                    ep_node = self._graph.get_node(ep_node_id)
                    if ep_node and ep_node.node_type == NODE_TYPE_EPISODE:
                        episode_scores[ep_node.source_id or ep_node_id] = max(
                            episode_scores.get(ep_node.source_id or ep_node_id, 0),
                            edge.weight,
                        )
        
        # Filter by time if specified
        if query.time_start or query.time_end:
            filtered = {}
            for ep_id, score in episode_scores.items():
                ep = self._episodes.get(ep_id)
                if ep:
                    if query.time_start and ep.end_time < query.time_start:
                        continue
                    if query.time_end and ep.start_time > query.time_end:
                        continue
                    filtered[ep_id] = score
            episode_scores = filtered
        
        # Sort by score
        return sorted(
            episode_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    
    # -------------------------------------------------------------------------
    # INTERNAL: Node/edge management
    # -------------------------------------------------------------------------
    
    def _get_or_create_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        source_id: str | None = None,
        embedding: list[float] | None = None,
    ) -> GraphNode:
        """Get existing node or create new one."""
        existing = self._graph.get_node(node_id)
        if existing:
            # Update access
            existing.last_accessed = datetime.now()
            existing.access_count += 1
            return existing
        
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            source_id=source_id,
            embedding=embedding,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
        )
        self._graph.add_node(node)
        return node
    
    def _create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
    ) -> GraphEdge | None:
        """Create an edge if both nodes exist."""
        # Check nodes exist
        if not self._graph.get_node(source_id):
            return None
        if not self._graph.get_node(target_id):
            return None
        
        edge = GraphEdge(
            edge_id=f"edge_{uuid.uuid4().hex[:12]}",
            edge_type=edge_type,
            source_node_id=source_id,
            target_node_id=target_id,
            weight=weight,
        )
        self._graph.add_edge(edge)
        return edge
    
    def _record_spatial_context(
        self,
        node_id: str,
        location_id: str,
        position: tuple[float, float, float],
    ) -> None:
        """Record spatial context for a node."""
        node = self._graph.get_node(node_id)
        if not node or not node.source_id:
            return
        
        # Record observation in landmark manager
        self._landmarks.record_entity_observation(
            entity_id=node.source_id,
            location_id=location_id,
            position=position,
        )
        
        # Create spatial edge to location
        loc_node_id = f"{NODE_TYPE_LOCATION}_{location_id}"
        if self._graph.get_node(loc_node_id):
            self._create_edge(
                source_id=node_id,
                target_id=loc_node_id,
                edge_type=EDGE_TYPE_TYPICAL_IN,
            )
    
    def _build_indexes(self) -> None:
        """Build indexes from existing data."""
        # Index nodes by label
        for node in self._graph.get_all_nodes():
            if node.label:
                self._label_to_nodes[node.label.lower()].add(node.node_id)
            for label in node.labels:
                self._label_to_nodes[label.lower()].add(node.node_id)
        
        # Index episodes by location
        for episode in self._episodes.get_all():
            if episode.location_label:
                # Find location ID from label
                for node in self._graph.get_all_nodes():
                    if (node.node_type == NODE_TYPE_LOCATION and 
                        node.label == episode.location_label):
                        loc_id = node.source_id or node.node_id
                        self._location_to_episodes[loc_id].add(episode.episode_id)
                        break
    
    # -------------------------------------------------------------------------
    # INTERNAL: Type mappings
    # -------------------------------------------------------------------------
    
    def _type_to_category(self, obs_type: str) -> str:
        """Map observation type to learning category."""
        mapping = {
            "entity": CATEGORY_ENTITY,
            "location": CATEGORY_LOCATION,
            "event": CATEGORY_EVENT,
        }
        return mapping.get(obs_type, CATEGORY_ENTITY)
    
    def _type_to_node_type(self, obs_type: str) -> str:
        """Map observation type to node type."""
        mapping = {
            "entity": NODE_TYPE_ENTITY,
            "location": NODE_TYPE_LOCATION,
            "event": NODE_TYPE_EVENT,
        }
        return mapping.get(obs_type, NODE_TYPE_ENTITY)
    
    def _node_type_to_category(self, node_type: str) -> str:
        """Map node type to learning category."""
        mapping = {
            NODE_TYPE_ENTITY: CATEGORY_ENTITY,
            NODE_TYPE_LOCATION: CATEGORY_LOCATION,
            NODE_TYPE_EVENT: CATEGORY_EVENT,
        }
        return mapping.get(node_type, CATEGORY_ENTITY)
    
    # -------------------------------------------------------------------------
    # INTERNAL: Math helpers
    # -------------------------------------------------------------------------
    
    def _compute_distance(
        self,
        pos1: tuple[float, float, float],
        pos2: tuple[float, float, float],
    ) -> float:
        """Compute Euclidean distance between positions."""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )
    
    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Compute cosine similarity between vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
