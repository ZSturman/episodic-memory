"""Cheat resolvers for Unity integration.

Uses Unity's ground-truth GUIDs for perfect location and entity
resolution without real perception. Integrates with graph memory
and dialog manager for label learning.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from episodic_agent.core.interfaces import EntityResolver, LocationResolver
from episodic_agent.modules.unity.perception import guid_to_embedding
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
from episodic_agent.utils.config import CONFIDENCE_T_HIGH, CONFIDENCE_T_LOW, DEFAULT_EMBEDDING_DIM

if TYPE_CHECKING:
    from episodic_agent.modules.dialog import DialogManager
    from episodic_agent.memory.graph_store import LabeledGraphStore

logger = logging.getLogger(__name__)


class LocationResolverCheat(LocationResolver):
    """Location resolver using Unity's room GUIDs.
    
    Features:
    - Matches locations by room GUID (perfect identification)
    - Persists learned locations to graph memory
    - Prompts for labels via dialog manager when needed
    - Tracks visit counts and updates confidence over time
    
    After labeling a room once, revisits resolve automatically
    with high confidence.
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        dialog_manager: "DialogManager",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        auto_label: bool = False,
    ) -> None:
        """Initialize the location resolver.
        
        Args:
            graph_store: Graph store for location persistence.
            dialog_manager: Dialog manager for label requests.
            embedding_dim: Embedding dimension for location nodes.
            auto_label: If True, auto-generate labels without prompting.
        """
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        self._embedding_dim = embedding_dim
        self._auto_label = auto_label
        
        self._confidence_helper = ConfidenceHelper()
        
        # Track current location for change detection
        self._current_room_guid: str | None = None
        self._pending_label_request: str | None = None

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> tuple[str, float]:
        """Resolve current location from percept.
        
        Uses room GUID from percept.extras to identify location.
        Creates new location node if unknown, prompts for label.
        
        Args:
            percept: Current perception with room info in extras.
            acf: Current active context frame.
            
        Returns:
            Tuple of (location_label, confidence).
        """
        extras = percept.extras or {}
        room_guid = extras.get("room_guid")
        room_label_hint = extras.get("room_label", "")
        
        # No room = unknown location
        if not room_guid:
            return ("unknown", 0.0)
        
        # Track room changes
        room_changed = room_guid != self._current_room_guid
        self._current_room_guid = room_guid
        
        # Look up existing location node by GUID
        location_node = self._find_location_by_guid(room_guid)
        
        if location_node:
            # Known location - update and return
            return self._handle_known_location(location_node, room_changed)
        else:
            # Unknown location - create and possibly label
            return self._handle_unknown_location(room_guid, room_label_hint, room_changed)

    def _find_location_by_guid(self, room_guid: str) -> GraphNode | None:
        """Find a location node by its room GUID.
        
        Args:
            room_guid: The Unity room GUID.
            
        Returns:
            The location GraphNode if found, None otherwise.
        """
        # Look for nodes with matching GUID in extras or source_id
        location_nodes = self._graph_store.get_nodes_by_type(NodeType.LOCATION)
        
        for node in location_nodes:
            # Check source_id first (canonical)
            if node.source_id == room_guid:
                return node
            # Check extras for backward compatibility
            if node.extras.get("room_guid") == room_guid:
                return node
        
        return None

    def _handle_known_location(
        self,
        node: GraphNode,
        room_changed: bool,
    ) -> tuple[str, float]:
        """Handle resolution of a known location.
        
        Args:
            node: The existing location node.
            room_changed: Whether we just entered this room.
            
        Returns:
            Tuple of (location_label, confidence).
        """
        # Update access info
        node.last_accessed = datetime.now()
        node.access_count += 1
        
        # Build confidence from signals
        signals = [
            ConfidenceSignal("guid_match", 1.0, weight=2.0),  # Perfect GUID match
            ConfidenceSignal(
                "visit_count", 
                min(1.0, node.access_count / 10.0),  # Caps at 10 visits
                weight=0.5,
            ),
        ]
        
        confidence = self._confidence_helper.combine_weighted(signals)
        
        # Notify on room change
        if room_changed:
            self._dialog_manager.notify(f"📍 Entered: {node.label}")
        
        return (node.label, confidence)

    def _handle_unknown_location(
        self,
        room_guid: str,
        room_label_hint: str,
        room_changed: bool,
    ) -> tuple[str, float]:
        """Handle resolution of an unknown location.
        
        Creates a new location node and optionally prompts for label.
        
        Args:
            room_guid: The Unity room GUID.
            room_label_hint: Label hint from Unity (may be empty).
            room_changed: Whether we just entered this room.
            
        Returns:
            Tuple of (location_label, confidence).
        """
        # Generate embedding from GUID
        embedding = guid_to_embedding(room_guid, self._embedding_dim)
        
        # Determine label
        if self._auto_label:
            # Auto-generate label
            if room_label_hint:
                label = room_label_hint
            else:
                label = f"location_{room_guid[:8]}"
        else:
            # Prompt for label via dialog
            suggestions = [room_label_hint] if room_label_hint else []
            suggestions.append(f"room_{room_guid[:8]}")
            
            self._dialog_manager.notify(f"🆕 New location detected!")
            label = self._dialog_manager.ask_label(
                "What should this location be called?",
                suggestions=suggestions,
            )
        
        # Create location node
        node = GraphNode(
            node_id=f"loc_{uuid.uuid4().hex[:12]}",
            node_type=NodeType.LOCATION,
            label=label,
            labels=[room_label_hint] if room_label_hint and room_label_hint != label else [],
            embedding=embedding,
            source_id=room_guid,
            confidence=CONFIDENCE_T_HIGH,  # High confidence from GUID
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            extras={
                "room_guid": room_guid,
                "source": "unity_cheat",
            },
        )
        
        # Persist to graph
        self._graph_store.add_node(node)
        logger.info(f"Created location node: {label} (GUID: {room_guid[:8]}...)")
        
        self._dialog_manager.notify(f"✅ Learned location: {label}")
        
        return (label, CONFIDENCE_T_HIGH)

    def get_location_node(self, room_guid: str) -> GraphNode | None:
        """Get the location node for a room GUID (public accessor).
        
        Args:
            room_guid: The Unity room GUID.
            
        Returns:
            The location GraphNode if found.
        """
        return self._find_location_by_guid(room_guid)


class EntityResolverCheat(EntityResolver):
    """Entity resolver using Unity's entity GUIDs.
    
    Features:
    - Matches entities by GUID (perfect identification)
    - Creates/stores new entity nodes when unknown
    - Prompts for labels via dialog manager
    - Links entities to locations (typical_in edges)
    - Links entities to contexts (in_context edges)
    - Tracks entity "inventory" of what's been seen
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        dialog_manager: "DialogManager",
        location_resolver: LocationResolverCheat,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        auto_label: bool = True,  # Default to auto for entities
        prompt_for_new_entities: bool = False,
    ) -> None:
        """Initialize the entity resolver.
        
        Args:
            graph_store: Graph store for entity persistence.
            dialog_manager: Dialog manager for label requests.
            location_resolver: Location resolver for linking.
            embedding_dim: Embedding dimension for entity nodes.
            auto_label: If True, auto-generate labels without prompting.
            prompt_for_new_entities: If True, prompt for every new entity.
        """
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        self._location_resolver = location_resolver
        self._embedding_dim = embedding_dim
        self._auto_label = auto_label
        self._prompt_for_new_entities = prompt_for_new_entities
        
        self._confidence_helper = ConfidenceHelper()
        
        # Track entity visibility for change detection
        self._visible_entities: set[str] = set()
        self._entity_nodes_cache: dict[str, GraphNode] = {}

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[ObjectCandidate]:
        """Resolve entities from percept.
        
        Matches candidates by GUID, creates new entity nodes for
        unknown entities, updates typical_in links.
        
        Args:
            percept: Current perception with entity candidates.
            acf: Current active context frame.
            
        Returns:
            List of resolved ObjectCandidates with updated labels.
        """
        extras = percept.extras or {}
        room_guid = extras.get("room_guid")
        
        resolved_candidates = []
        current_visible: set[str] = set()
        
        for candidate in percept.candidates:
            guid = candidate.extras.get("guid") if candidate.extras else candidate.candidate_id
            if not guid:
                resolved_candidates.append(candidate)
                continue
            
            current_visible.add(guid)
            
            # Look up or create entity node
            entity_node = self._find_or_create_entity(candidate, guid)
            
            # Update candidate with resolved info
            resolved_candidate = self._update_candidate(candidate, entity_node)
            resolved_candidates.append(resolved_candidate)
            
            # Link to current location if known
            if room_guid:
                self._update_location_link(entity_node, room_guid)
        
        # Detect entity changes (entered/left visibility)
        entered = current_visible - self._visible_entities
        left = self._visible_entities - current_visible
        
        if entered:
            entered_labels = [self._get_entity_label(g) for g in entered]
            logger.debug(f"Entities entered view: {entered_labels}")
        
        if left:
            left_labels = [self._get_entity_label(g) for g in left]
            logger.debug(f"Entities left view: {left_labels}")
        
        self._visible_entities = current_visible
        
        return resolved_candidates

    def _find_or_create_entity(
        self,
        candidate: ObjectCandidate,
        guid: str,
    ) -> GraphNode:
        """Find existing entity node or create a new one.
        
        Args:
            candidate: The object candidate from perception.
            guid: The entity GUID.
            
        Returns:
            The entity GraphNode (existing or new).
        """
        # Check cache first
        if guid in self._entity_nodes_cache:
            node = self._entity_nodes_cache[guid]
            node.last_accessed = datetime.now()
            node.access_count += 1
            return node
        
        # Search graph store
        entity_node = self._find_entity_by_guid(guid)
        
        if entity_node:
            # Cache and update
            self._entity_nodes_cache[guid] = entity_node
            entity_node.last_accessed = datetime.now()
            entity_node.access_count += 1
            return entity_node
        
        # Create new entity node
        entity_node = self._create_entity_node(candidate, guid)
        self._entity_nodes_cache[guid] = entity_node
        
        return entity_node

    def _find_entity_by_guid(self, guid: str) -> GraphNode | None:
        """Find an entity node by its GUID.
        
        Args:
            guid: The Unity entity GUID.
            
        Returns:
            The entity GraphNode if found, None otherwise.
        """
        entity_nodes = self._graph_store.get_nodes_by_type(NodeType.ENTITY)
        
        for node in entity_nodes:
            if node.source_id == guid:
                return node
            if node.extras.get("guid") == guid:
                return node
        
        return None

    def _create_entity_node(
        self,
        candidate: ObjectCandidate,
        guid: str,
    ) -> GraphNode:
        """Create a new entity node from a candidate.
        
        Args:
            candidate: The object candidate with entity info.
            guid: The entity GUID.
            
        Returns:
            The newly created entity GraphNode.
        """
        extras = candidate.extras or {}
        unity_label = candidate.label
        category = extras.get("category", "unknown")
        
        # Determine label
        if self._prompt_for_new_entities and not self._auto_label:
            suggestions = [unity_label] if unity_label != "unknown" else []
            suggestions.append(f"{category}_{guid[:8]}")
            
            self._dialog_manager.notify(f"🆕 New entity detected: {unity_label}")
            label = self._dialog_manager.ask_label(
                f"What should this {category} be called?",
                suggestions=suggestions,
            )
        else:
            # Use Unity's label or generate one
            label = unity_label if unity_label != "unknown" else f"{category}_{guid[:8]}"
        
        # Generate embedding from GUID
        embedding = guid_to_embedding(guid, self._embedding_dim)
        
        # Create node
        node = GraphNode(
            node_id=f"ent_{uuid.uuid4().hex[:12]}",
            node_type=NodeType.ENTITY,
            label=label,
            labels=[unity_label] if unity_label and unity_label != label else [],
            embedding=embedding,
            source_id=guid,
            confidence=CONFIDENCE_T_HIGH,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            extras={
                "guid": guid,
                "category": category,
                "initial_state": extras.get("state", "default"),
                "source": "unity_cheat",
            },
        )
        
        # Persist to graph
        self._graph_store.add_node(node)
        logger.info(f"Created entity node: {label} ({category}, GUID: {guid[:8]}...)")
        
        return node

    def _update_candidate(
        self,
        candidate: ObjectCandidate,
        entity_node: GraphNode,
    ) -> ObjectCandidate:
        """Update a candidate with resolved entity information.
        
        Args:
            candidate: The original object candidate.
            entity_node: The resolved entity node.
            
        Returns:
            Updated ObjectCandidate with resolved label and confidence.
        """
        # Build confidence signals
        signals = [
            ConfidenceSignal("guid_match", 1.0, weight=2.0),
            ConfidenceSignal("visibility", 1.0 if candidate.extras.get("visible", True) else 0.5, weight=1.0),
            ConfidenceSignal("visits", min(1.0, entity_node.access_count / 5.0), weight=0.5),
        ]
        
        confidence = self._confidence_helper.combine_weighted(signals)
        
        # Update extras with resolution info
        updated_extras = dict(candidate.extras or {})
        updated_extras["entity_node_id"] = entity_node.node_id
        updated_extras["resolved"] = True
        
        return ObjectCandidate(
            candidate_id=candidate.candidate_id,
            label=entity_node.label,
            labels=entity_node.labels + candidate.labels,
            confidence=confidence,
            embedding=entity_node.embedding,
            position=candidate.position,
            bounding_box=candidate.bounding_box,
            extras=updated_extras,
        )

    def _update_location_link(
        self,
        entity_node: GraphNode,
        room_guid: str,
    ) -> None:
        """Update the typical_in link between entity and location.
        
        Increments the edge weight each time entity is seen in location.
        
        Args:
            entity_node: The entity node.
            room_guid: The current room GUID.
        """
        # Get location node
        location_node = self._location_resolver.get_location_node(room_guid)
        if not location_node:
            return
        
        # Look for existing edge
        existing_edge = self._find_typical_in_edge(
            entity_node.node_id,
            location_node.node_id,
        )
        
        if existing_edge:
            # Increment weight (observation count)
            existing_edge.weight += 1.0
            existing_edge.last_accessed = datetime.now()
        else:
            # Create new typical_in edge
            edge = GraphEdge(
                edge_id=f"edge_{uuid.uuid4().hex[:12]}",
                edge_type=EdgeType.TYPICAL_IN,
                source_node_id=entity_node.node_id,
                target_node_id=location_node.node_id,
                weight=1.0,
                confidence=CONFIDENCE_T_HIGH,
                extras={
                    "observation_count": 1,
                    "source": "unity_cheat",
                },
            )
            self._graph_store.add_edge(edge)
            logger.debug(f"Created typical_in edge: {entity_node.label} -> {location_node.label}")

    def _find_typical_in_edge(
        self,
        entity_node_id: str,
        location_node_id: str,
    ) -> GraphEdge | None:
        """Find an existing typical_in edge between entity and location.
        
        Args:
            entity_node_id: The entity node ID.
            location_node_id: The location node ID.
            
        Returns:
            The GraphEdge if found, None otherwise.
        """
        edges = self._graph_store.get_outgoing_edges(entity_node_id)
        
        for edge in edges:
            if (edge.edge_type == EdgeType.TYPICAL_IN and 
                edge.target_node_id == location_node_id):
                return edge
        
        return None

    def _get_entity_label(self, guid: str) -> str:
        """Get the label for an entity by GUID.
        
        Args:
            guid: The entity GUID.
            
        Returns:
            The entity label or GUID if unknown.
        """
        node = self._entity_nodes_cache.get(guid)
        if node:
            return node.label
        
        node = self._find_entity_by_guid(guid)
        if node:
            return node.label
        
        return f"entity_{guid[:8]}"

    def get_visible_entity_summary(self) -> dict[str, int]:
        """Get a summary of currently visible entities by category.
        
        Returns:
            Dict mapping category to count of visible entities.
        """
        summary: dict[str, int] = {}
        
        for guid in self._visible_entities:
            node = self._entity_nodes_cache.get(guid)
            if node:
                category = node.extras.get("category", "unknown")
            else:
                category = "unknown"
            
            summary[category] = summary.get(category, 0) + 1
        
        return summary

    def get_entity_inventory(self) -> list[dict[str, Any]]:
        """Get the full inventory of known entities.
        
        Returns:
            List of dicts with entity info (label, category, visits, etc.).
        """
        inventory = []
        
        for guid, node in self._entity_nodes_cache.items():
            inventory.append({
                "guid": guid,
                "label": node.label,
                "category": node.extras.get("category", "unknown"),
                "access_count": node.access_count,
                "created_at": node.created_at.isoformat(),
                "last_seen": node.last_accessed.isoformat(),
            })
        
        return inventory
