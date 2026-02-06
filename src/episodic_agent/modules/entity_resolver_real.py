"""Real entity resolver using embedding similarity.

Identifies and tracks entities from perception data without relying
on Unity GUIDs. Uses embedding similarity for re-identification
and the dialog manager for label learning.

ARCHITECTURAL INVARIANT: No pre-wired semantics. Entity identity is
determined by visual/perceptual similarity, not by privileged IDs.
Labels come from user interaction only.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from episodic_agent.core.interfaces import EntityResolver
from episodic_agent.schemas import (
    ActiveContextFrame,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    ObjectCandidate,
    Percept,
)
from episodic_agent.utils.confidence import ConfidenceHelper, ConfidenceSignal
from episodic_agent.utils.config import (
    CONFIDENCE_T_HIGH,
    CONFIDENCE_T_LOW,
    DEFAULT_EMBEDDING_DIM,
)

if TYPE_CHECKING:
    from episodic_agent.memory.graph_store import LabeledGraphStore
    from episodic_agent.modules.dialog import DialogManager
    from episodic_agent.core.interfaces import LocationResolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# EntityResolverReal
# ---------------------------------------------------------------------------

class EntityResolverReal(EntityResolver):
    """Entity resolver using embedding-based re-identification.

    Instead of matching by Unity GUID, this resolver:
    1. Compares each candidate's embedding against known entity nodes
    2. If similarity exceeds a threshold → re-identification
    3. Otherwise → new entity, optionally prompt for label
    4. Links entities to locations via typical_in edges

    Constructor parameters:
        graph_store:         Graph store for persistence.
        dialog_manager:      For label prompting.
        location_resolver:   For linking entities to locations.
        match_threshold:     Cosine similarity above which an entity
                             is considered a re-identification (default 0.80).
        embedding_dim:       Dimensionality of entity embeddings.
        auto_label:          If True, auto-generates labels without prompt.
        prompt_for_new:      If True, prompt user for every new entity.
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        dialog_manager: "DialogManager",
        location_resolver: "LocationResolver",
        match_threshold: float = 0.80,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        auto_label: bool = True,
        prompt_for_new: bool = False,
    ) -> None:
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        self._location_resolver = location_resolver
        self._match_threshold = match_threshold
        self._embedding_dim = embedding_dim
        self._auto_label = auto_label
        self._prompt_for_new = prompt_for_new

        self._confidence_helper = ConfidenceHelper()

        # Local cache: entity_node_id → GraphNode
        self._entity_cache: dict[str, GraphNode] = {}
        # Track visibility set for change detection
        self._visible_entity_ids: set[str] = set()

    # ------------------------------------------------------------------
    # EntityResolver interface
    # ------------------------------------------------------------------

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[ObjectCandidate]:
        """Resolve entities from perception.

        Args:
            percept: Current perception with object candidates.
            acf:     Active context frame.

        Returns:
            Updated list of ObjectCandidates with resolved labels.
        """
        resolved: list[ObjectCandidate] = []
        current_visible: set[str] = set()

        for candidate in percept.candidates:
            embedding = candidate.embedding
            if not embedding:
                resolved.append(candidate)
                continue

            # Find best matching known entity
            match_node, similarity = self._find_best_entity_match(embedding)

            if match_node and similarity >= self._match_threshold:
                # Re-identification: update existing node
                entity_node = match_node
                entity_node.last_accessed = datetime.now()
                entity_node.access_count += 1
                # Update centroid embedding with running average
                if entity_node.embedding:
                    n = entity_node.access_count
                    entity_node.embedding = [
                        (old * (n - 1) + new) / n
                        for old, new in zip(entity_node.embedding, embedding)
                    ]
            else:
                # New entity
                entity_node = self._create_entity_node(candidate)
                self._entity_cache[entity_node.node_id] = entity_node

            current_visible.add(entity_node.node_id)

            # Build resolved candidate
            resolved_candidate = self._build_resolved_candidate(
                candidate, entity_node, similarity if match_node else 0.0,
            )
            resolved.append(resolved_candidate)

            # Link to current location
            self._update_location_link(entity_node, percept)

        # Detect visibility transitions
        entered = current_visible - self._visible_entity_ids
        left = self._visible_entity_ids - current_visible

        if entered:
            labels = [self._entity_cache.get(eid, GraphNode(node_id="?", node_type="?")).label for eid in entered]
            logger.debug("Entities entered view: %s", labels)
        if left:
            labels = [self._entity_cache.get(eid, GraphNode(node_id="?", node_type="?")).label for eid in left]
            logger.debug("Entities left view: %s", labels)

        self._visible_entity_ids = current_visible
        return resolved

    # ------------------------------------------------------------------
    # Entity matching
    # ------------------------------------------------------------------

    def _find_best_entity_match(
        self,
        embedding: list[float],
    ) -> tuple[GraphNode | None, float]:
        """Find the best matching known entity by embedding similarity.

        Searches both the local cache and the graph store.
        """
        best_node: GraphNode | None = None
        best_sim = -1.0

        # Search cache first (fast path)
        for node in self._entity_cache.values():
            if not node.embedding:
                continue
            sim = _cosine_similarity(embedding, node.embedding)
            if sim > best_sim:
                best_sim = sim
                best_node = node

        # Also search graph store for entities not in cache
        try:
            entity_nodes = self._graph_store.get_nodes_by_type(NodeType.ENTITY)
            for node in entity_nodes:
                if node.node_id in self._entity_cache:
                    continue  # already checked
                if not node.embedding:
                    continue
                sim = _cosine_similarity(embedding, node.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_node = node
                    # Cache it
                    self._entity_cache[node.node_id] = node
        except Exception:
            pass

        return (best_node, best_sim) if best_sim >= 0 else (None, 0.0)

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------

    def _create_entity_node(self, candidate: ObjectCandidate) -> GraphNode:
        """Create a new entity node from a perception candidate."""
        extras = candidate.extras or {}
        initial_label = candidate.label if candidate.label != "unknown" else None
        category = extras.get("category", "object")

        if self._prompt_for_new and not self._auto_label:
            suggestions = []
            if initial_label:
                suggestions.append(initial_label)
            suggestions.append(f"{category}_{uuid.uuid4().hex[:6]}")

            self._dialog_manager.notify(f"🆕 New entity detected!")
            label = self._dialog_manager.ask_label(
                f"What should this {category} be called?",
                suggestions=suggestions,
            )
        else:
            label = initial_label or f"{category}_{uuid.uuid4().hex[:6]}"

        node = GraphNode(
            node_id=f"ent_fp_{uuid.uuid4().hex[:12]}",
            node_type=NodeType.ENTITY,
            label=label,
            labels=[initial_label] if initial_label and initial_label != label else [],
            embedding=list(candidate.embedding) if candidate.embedding else [],
            source_id=f"fp_{uuid.uuid4().hex[:8]}",
            confidence=CONFIDENCE_T_LOW,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            extras={
                "category": category,
                "source": "fingerprint",
            },
        )

        self._graph_store.add_node(node)
        logger.info("Created entity node: %s (%s)", label, category)
        return node

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def _build_resolved_candidate(
        self,
        original: ObjectCandidate,
        entity_node: GraphNode,
        similarity: float,
    ) -> ObjectCandidate:
        """Build a resolved ObjectCandidate from original + entity node."""
        signals = [
            ConfidenceSignal("embedding_match", max(0.0, similarity), weight=2.0),
            ConfidenceSignal(
                "visibility",
                1.0 if (original.extras or {}).get("visible", True) else 0.5,
                weight=1.0,
            ),
            ConfidenceSignal(
                "visits",
                min(1.0, entity_node.access_count / 5.0),
                weight=0.5,
            ),
        ]
        confidence = self._confidence_helper.combine_weighted(signals)

        updated_extras = dict(original.extras or {})
        updated_extras["entity_node_id"] = entity_node.node_id
        updated_extras["resolved"] = True
        updated_extras["match_similarity"] = similarity

        return ObjectCandidate(
            candidate_id=original.candidate_id,
            label=entity_node.label,
            labels=entity_node.labels + original.labels,
            confidence=confidence,
            embedding=entity_node.embedding,
            position=original.position,
            bounding_box=original.bounding_box,
            extras=updated_extras,
        )

    # ------------------------------------------------------------------
    # Location linking
    # ------------------------------------------------------------------

    def _update_location_link(
        self,
        entity_node: GraphNode,
        percept: Percept,
    ) -> None:
        """Link entity to its current location via typical_in edge."""
        # Get current location from the percept's extras (set by orchestrator)
        extras = percept.extras or {}
        location_label = extras.get("resolved_location")
        if not location_label:
            return

        # Find the location node
        try:
            location_nodes = self._graph_store.get_nodes_by_type(NodeType.LOCATION)
        except Exception:
            return

        location_node = None
        for ln in location_nodes:
            if ln.label == location_label or ln.source_id == location_label:
                location_node = ln
                break

        if not location_node:
            return

        # Check for existing edge
        try:
            edges = self._graph_store.get_outgoing_edges(entity_node.node_id)
        except Exception:
            edges = []

        for edge in edges:
            if (
                edge.edge_type == EdgeType.TYPICAL_IN
                and edge.target_node_id == location_node.node_id
            ):
                edge.weight += 1.0
                edge.last_accessed = datetime.now()
                return

        # Create new edge
        edge = GraphEdge(
            edge_id=f"edge_{uuid.uuid4().hex[:12]}",
            edge_type=EdgeType.TYPICAL_IN,
            source_node_id=entity_node.node_id,
            target_node_id=location_node.node_id,
            weight=1.0,
            confidence=CONFIDENCE_T_LOW,
            extras={"source": "fingerprint"},
        )
        self._graph_store.add_edge(edge)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_visible_entity_summary(self) -> dict[str, int]:
        """Summary of currently visible entities by category."""
        summary: dict[str, int] = {}
        for eid in self._visible_entity_ids:
            node = self._entity_cache.get(eid)
            cat = node.extras.get("category", "unknown") if node else "unknown"
            summary[cat] = summary.get(cat, 0) + 1
        return summary

    def get_entity_inventory(self) -> list[dict[str, Any]]:
        """Full inventory of known entities."""
        return [
            {
                "node_id": node.node_id,
                "label": node.label,
                "category": node.extras.get("category", "unknown"),
                "access_count": node.access_count,
                "created_at": node.created_at.isoformat(),
                "last_seen": node.last_accessed.isoformat(),
            }
            for node in self._entity_cache.values()
        ]
