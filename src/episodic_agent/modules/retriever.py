"""Spreading activation retriever for associative memory retrieval.

Implements spreading activation over the memory graph:
1. Build cue tokens from ACF (location, entities, events, goals, prediction errors)
2. Activate cue and goal nodes
3. Propagate activation through graph for N hops with decay
4. Return top-K episodes/nodes with scores and retrieval certainty
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from episodic_agent.core.interfaces import Retriever
from episodic_agent.schemas import (
    Episode,
    GraphNode,
    NodeType,
    RetrievalResult,
)

if TYPE_CHECKING:
    from episodic_agent.memory.episode_store import PersistentEpisodeStore
    from episodic_agent.memory.graph_store import LabeledGraphStore
    from episodic_agent.schemas import ActiveContextFrame

logger = logging.getLogger(__name__)


class CueToken:
    """Represents a retrieval cue derived from context."""
    
    def __init__(
        self,
        token_type: str,
        value: str,
        weight: float = 1.0,
        source: str = "unknown",
    ) -> None:
        """Initialize a cue token.
        
        Args:
            token_type: Type of cue (location, entity, event, goal, prediction_error).
            value: The token value (label, id, etc.).
            weight: Importance weight for this cue.
            source: Where this cue came from.
        """
        self.token_type = token_type
        self.value = value
        self.weight = weight
        self.source = source
    
    def __repr__(self) -> str:
        return f"CueToken({self.token_type}:{self.value}, w={self.weight:.2f})"


class ActivatedNode:
    """Tracks activation for a single node during spreading."""
    
    def __init__(self, node_id: str, initial_activation: float = 0.0) -> None:
        self.node_id = node_id
        self.activation = initial_activation
        self.activation_history: list[tuple[int, float]] = [(0, initial_activation)]
        self.source_cues: list[str] = []
    
    def add_activation(self, amount: float, hop: int, source: str = "") -> None:
        """Add activation from spreading."""
        self.activation += amount
        self.activation_history.append((hop, self.activation))
        if source and source not in self.source_cues:
            self.source_cues.append(source)


class SpreadingActivationResult:
    """Detailed result from spreading activation."""
    
    def __init__(self) -> None:
        self.activated_nodes: dict[str, ActivatedNode] = {}
        self.episode_rankings: list[tuple[str, float]] = []
        self.cues_used: list[CueToken] = []
        self.hops_performed: int = 0
        self.total_activation: float = 0.0
        self.retrieval_certainty: float = 0.0


class RetrieverSpreadingActivation(Retriever):
    """Retriever using spreading activation over the memory graph.
    
    Features:
    - Builds cue tokens from ACF (location, entities, events, prediction errors)
    - Activates cue nodes and propagates through graph edges
    - Configurable number of hops and decay factor
    - Returns ranked episodes and nodes with activation scores
    - Inspectable: exposes activated nodes and episode rankings
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        episode_store: "PersistentEpisodeStore",
        max_hops: int = 3,
        decay_factor: float = 0.5,
        initial_activation: float = 1.0,
        prediction_error_weight: float = 1.5,
        goal_weight: float = 1.2,
        top_k_default: int = 5,
        min_activation_threshold: float = 0.01,
        log_retrievals: bool = True,
    ) -> None:
        """Initialize the spreading activation retriever.
        
        Args:
            graph_store: Graph store for node/edge access.
            episode_store: Episode store for episode retrieval.
            max_hops: Maximum propagation hops (2-3 recommended).
            decay_factor: Activation decay per hop (0-1).
            initial_activation: Initial activation for cue nodes.
            prediction_error_weight: Extra weight for prediction error cues.
            goal_weight: Extra weight for goal cues.
            top_k_default: Default number of results to return.
            min_activation_threshold: Minimum activation to track.
            log_retrievals: Whether to log top episodes each step.
        """
        self._graph_store = graph_store
        self._episode_store = episode_store
        self._max_hops = max_hops
        self._decay_factor = decay_factor
        self._initial_activation = initial_activation
        self._prediction_error_weight = prediction_error_weight
        self._goal_weight = goal_weight
        self._top_k_default = top_k_default
        self._min_activation_threshold = min_activation_threshold
        self._log_retrievals = log_retrievals
        
        # Last retrieval result for inspection
        self._last_result: SpreadingActivationResult | None = None
        
        # Statistics
        self._retrievals_performed = 0
        self._total_cues_used = 0
        self._total_nodes_activated = 0

    @property
    def last_result(self) -> SpreadingActivationResult | None:
        """Get the last spreading activation result for inspection."""
        return self._last_result

    @property
    def retrievals_performed(self) -> int:
        """Get count of retrievals performed."""
        return self._retrievals_performed

    def retrieve(
        self,
        acf: "ActiveContextFrame",
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve relevant memories using spreading activation.
        
        Args:
            acf: Current active context frame as query.
            top_k: Maximum number of results to return.
            
        Returns:
            RetrievalResult with ranked episodes and nodes.
        """
        import time
        start_time = time.perf_counter()
        
        # Build cue tokens from ACF
        cues = self._build_cues(acf)
        
        # Perform spreading activation
        result = self._spread_activation(cues)
        result.cues_used = cues
        
        # Rank episodes by activation of their associated nodes
        episode_rankings = self._rank_episodes(result)
        result.episode_rankings = episode_rankings
        
        # Compute retrieval certainty (based on activation distribution)
        result.retrieval_certainty = self._compute_certainty(result)
        
        # Store for inspection
        self._last_result = result
        
        # Get top episodes
        top_k = top_k or self._top_k_default
        episodes = []
        episode_scores = []
        
        for episode_id, score in episode_rankings[:top_k]:
            episode = self._episode_store.get(episode_id)
            if episode:
                episodes.append(episode)
                episode_scores.append(score)
        
        # Get top activated nodes (excluding episode nodes)
        nodes = []
        node_scores = []
        sorted_nodes = sorted(
            result.activated_nodes.values(),
            key=lambda n: n.activation,
            reverse=True,
        )
        
        for activated in sorted_nodes[:top_k]:
            node = self._graph_store.get_node(activated.node_id)
            if node and node.node_type != NodeType.EPISODE:
                nodes.append(node)
                node_scores.append(activated.activation)
        
        # Log if enabled
        if self._log_retrievals and episodes:
            logger.debug(f"Top retrieved episodes: {[e.episode_id for e in episodes[:3]]}")
            if cues:
                logger.debug(f"Cues used: {[str(c) for c in cues[:5]]}")
        
        # Update statistics
        self._retrievals_performed += 1
        self._total_cues_used += len(cues)
        self._total_nodes_activated += len(result.activated_nodes)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return RetrievalResult(
            query_id=f"ret_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(),
            episodes=episodes,
            episode_scores=episode_scores,
            nodes=nodes,
            node_scores=node_scores,
            retrieval_method="spreading_activation",
            retrieval_time_ms=elapsed_ms,
            extras={
                "cues_count": len(cues),
                "nodes_activated": len(result.activated_nodes),
                "hops_performed": result.hops_performed,
                "retrieval_certainty": result.retrieval_certainty,
                "total_activation": result.total_activation,
                "activated_node_ids": list(result.activated_nodes.keys())[:20],
            },
        )

    def _build_cues(self, acf: "ActiveContextFrame") -> list[CueToken]:
        """Build cue tokens from ACF state.
        
        Extracts cues from:
        - Current location (label and ID)
        - Visible entities (IDs)
        - Recent events (IDs)
        - Goals (if any)
        - Prediction errors (if any)
        
        Args:
            acf: Active context frame to extract cues from.
            
        Returns:
            List of CueToken objects.
        """
        cues = []
        
        # Location cue
        if acf.location_label and acf.location_label != "unknown":
            weight = acf.location_confidence
            cues.append(CueToken(
                token_type="location",
                value=acf.location_label,
                weight=weight,
                source="acf_location",
            ))
        
        # Entity cues
        for entity in acf.entities[:10]:  # Limit to top 10 entities
            if entity.candidate_id:
                cues.append(CueToken(
                    token_type="entity",
                    value=entity.candidate_id,
                    weight=entity.confidence,
                    source=f"entity_{entity.label}",
                ))
            if entity.label and entity.label != "unknown":
                cues.append(CueToken(
                    token_type="entity_label",
                    value=entity.label,
                    weight=entity.confidence * 0.8,  # Slightly lower for label
                    source=f"entity_label_{entity.label}",
                ))
        
        # Event cues (from recent events)
        for event in acf.events[-5:]:  # Last 5 events
            event_id = event.get("event_id") or event.get("id")
            if event_id:
                cues.append(CueToken(
                    token_type="event",
                    value=event_id,
                    weight=event.get("confidence", 0.5),
                    source=f"event_{event.get('label', '?')}",
                ))
        
        # Goal cues (from extras if present)
        goals = acf.extras.get("goals", [])
        for goal in goals:
            goal_id = goal.get("goal_id") or goal.get("id")
            if goal_id:
                cues.append(CueToken(
                    token_type="goal",
                    value=goal_id,
                    weight=self._goal_weight,
                    source=f"goal_{goal.get('label', '?')}",
                ))
        
        # Prediction error cues (high priority)
        prediction_errors = acf.extras.get("prediction_errors", [])
        for error in prediction_errors:
            error_type = error.get("type", "unknown")
            entity_id = error.get("entity_id")
            if entity_id:
                cues.append(CueToken(
                    token_type="prediction_error",
                    value=entity_id,
                    weight=self._prediction_error_weight * error.get("magnitude", 1.0),
                    source=f"pred_error_{error_type}",
                ))
        
        return cues

    def _spread_activation(self, cues: list[CueToken]) -> SpreadingActivationResult:
        """Perform spreading activation from cue nodes.
        
        Args:
            cues: List of cue tokens to activate.
            
        Returns:
            SpreadingActivationResult with activated nodes.
        """
        result = SpreadingActivationResult()
        
        if not cues:
            return result
        
        # Initialize activation for cue nodes
        for cue in cues:
            node_ids = self._find_nodes_for_cue(cue)
            for node_id in node_ids:
                if node_id not in result.activated_nodes:
                    result.activated_nodes[node_id] = ActivatedNode(
                        node_id, self._initial_activation * cue.weight
                    )
                else:
                    result.activated_nodes[node_id].add_activation(
                        self._initial_activation * cue.weight, 0, cue.value
                    )
                result.activated_nodes[node_id].source_cues.append(str(cue))
        
        # Spread activation for max_hops
        for hop in range(1, self._max_hops + 1):
            new_activations: dict[str, float] = defaultdict(float)
            
            for node_id, activated in result.activated_nodes.items():
                if activated.activation < self._min_activation_threshold:
                    continue
                
                # Get edges from this node
                edges = self._graph_store.get_edges(node_id)
                for edge in edges:
                    # Determine neighbor
                    neighbor_id = (
                        edge.target_node_id
                        if edge.source_node_id == node_id
                        else edge.source_node_id
                    )
                    
                    # Calculate spread activation
                    spread = activated.activation * self._decay_factor * edge.weight
                    if spread >= self._min_activation_threshold:
                        new_activations[neighbor_id] += spread
            
            # Apply new activations
            for node_id, activation in new_activations.items():
                if node_id not in result.activated_nodes:
                    result.activated_nodes[node_id] = ActivatedNode(node_id, 0)
                result.activated_nodes[node_id].add_activation(activation, hop)
            
            result.hops_performed = hop
            
            # Early termination if no significant new activations
            if not new_activations:
                break
        
        # Calculate total activation
        result.total_activation = sum(
            n.activation for n in result.activated_nodes.values()
        )
        
        return result

    def _find_nodes_for_cue(self, cue: CueToken) -> list[str]:
        """Find graph nodes matching a cue token.
        
        Args:
            cue: Cue token to match.
            
        Returns:
            List of matching node IDs.
        """
        node_ids = []
        
        if cue.token_type == "location":
            # Find by location label
            nodes = self._graph_store.get_nodes_by_label(cue.value)
            for node in nodes:
                if node.node_type == NodeType.LOCATION:
                    node_ids.append(node.node_id)
        
        elif cue.token_type in ("entity", "entity_label"):
            # Find by entity label or ID
            nodes = self._graph_store.get_nodes_by_label(cue.value)
            for node in nodes:
                if node.node_type == NodeType.ENTITY:
                    node_ids.append(node.node_id)
            
            # Also check by source_id
            for node in self._graph_store.get_nodes_by_type(NodeType.ENTITY):
                if node.source_id == cue.value:
                    node_ids.append(node.node_id)
        
        elif cue.token_type == "event":
            # Find by event label or ID
            nodes = self._graph_store.get_nodes_by_label(cue.value)
            for node in nodes:
                if node.node_type == NodeType.EVENT:
                    node_ids.append(node.node_id)
        
        elif cue.token_type == "goal":
            nodes = self._graph_store.get_nodes_by_label(cue.value)
            for node in nodes:
                if node.node_type == NodeType.GOAL:
                    node_ids.append(node.node_id)
        
        elif cue.token_type == "prediction_error":
            # For prediction errors, activate the entity involved
            nodes = self._graph_store.get_nodes_by_label(cue.value)
            node_ids.extend(n.node_id for n in nodes)
        
        return list(set(node_ids))

    def _rank_episodes(
        self,
        result: SpreadingActivationResult,
    ) -> list[tuple[str, float]]:
        """Rank episodes by accumulated activation.
        
        An episode's score is the sum of activation of its related nodes
        (location, entities, events).
        
        Args:
            result: Spreading activation result.
            
        Returns:
            List of (episode_id, score) tuples sorted by score.
        """
        episode_scores: dict[str, float] = defaultdict(float)
        
        # Score from episode nodes directly
        for node_id, activated in result.activated_nodes.items():
            node = self._graph_store.get_node(node_id)
            if node and node.node_type == NodeType.EPISODE:
                episode_id = node.source_id or node_id
                episode_scores[episode_id] += activated.activation
        
        # Also score based on matching location/entities
        for episode in self._episode_store.get_all():
            # Check if episode's location is activated
            loc_nodes = self._graph_store.get_nodes_by_label(episode.location_label)
            for loc_node in loc_nodes:
                if loc_node.node_id in result.activated_nodes:
                    episode_scores[episode.episode_id] += (
                        result.activated_nodes[loc_node.node_id].activation * 0.3
                    )
            
            # Check if episode's entities are activated
            for entity in episode.entities:
                if entity.candidate_id:
                    for node_id, activated in result.activated_nodes.items():
                        node = self._graph_store.get_node(node_id)
                        if node and node.source_id == entity.candidate_id:
                            episode_scores[episode.episode_id] += activated.activation * 0.2
        
        # Sort by score descending
        ranked = sorted(episode_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _compute_certainty(self, result: SpreadingActivationResult) -> float:
        """Compute retrieval certainty based on activation distribution.
        
        High certainty when activation is concentrated (clear winner).
        Low certainty when activation is spread evenly (ambiguous).
        
        Args:
            result: Spreading activation result.
            
        Returns:
            Certainty score between 0 and 1.
        """
        if not result.episode_rankings:
            return 0.0
        
        scores = [s for _, s in result.episode_rankings]
        if not scores:
            return 0.0
        
        total = sum(scores)
        if total == 0:
            return 0.0
        
        # Compute concentration (entropy-based)
        top_score = scores[0]
        concentration = top_score / total
        
        # Adjust by margin between top-1 and top-2
        if len(scores) > 1:
            margin = (scores[0] - scores[1]) / max(scores[0], 0.001)
            return min(1.0, concentration * 0.5 + margin * 0.5)
        
        return concentration

    def get_activated_nodes_summary(self) -> dict[str, Any]:
        """Get a summary of activated nodes from last retrieval.
        
        Returns:
            Summary dict with node counts and top activations.
        """
        if not self._last_result:
            return {"error": "No retrieval performed yet"}
        
        result = self._last_result
        
        # Group by node type
        by_type: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for node_id, activated in result.activated_nodes.items():
            node = self._graph_store.get_node(node_id)
            if node:
                # node_type is now a string, not enum with .value
                node_type_str = node.node_type if isinstance(node.node_type, str) else str(node.node_type)
                by_type[node_type_str].append((node.label, activated.activation))
        
        summary = {
            "total_nodes": len(result.activated_nodes),
            "total_activation": result.total_activation,
            "hops_performed": result.hops_performed,
            "cues_used": len(result.cues_used),
            "certainty": result.retrieval_certainty,
            "by_type": {},
        }
        
        for node_type, nodes in by_type.items():
            nodes_sorted = sorted(nodes, key=lambda x: x[1], reverse=True)[:5]
            summary["by_type"][node_type] = nodes_sorted
        
        return summary
