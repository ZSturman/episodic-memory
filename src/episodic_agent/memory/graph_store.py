"""Graph memory store with label indexing and neighbor queries."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from episodic_agent.core.interfaces import GraphStore
from episodic_agent.schemas import EdgeType, GraphEdge, GraphNode, NodeType


def _node_to_dict(node: GraphNode) -> dict[str, Any]:
    """Convert a GraphNode to a JSON-serializable dictionary."""
    # node_type is now a string, not an enum with .value
    node_type_str = node.node_type if isinstance(node.node_type, str) else node.node_type
    return {
        "node_id": node.node_id,
        "node_type": node_type_str,
        "label": node.label,
        "labels": node.labels,
        "embedding": node.embedding,
        "activation": node.activation,
        "base_activation": node.base_activation,
        "created_at": node.created_at.isoformat(),
        "last_accessed": node.last_accessed.isoformat(),
        "access_count": node.access_count,
        "source_id": node.source_id,
        "confidence": node.confidence,
        "extras": node.extras,
    }


def _dict_to_node(data: dict[str, Any]) -> GraphNode:
    """Convert a dictionary back to a GraphNode."""
    # node_type is now a string (not NodeType enum)
    node_type_val = data["node_type"]
    return GraphNode(
        node_id=data["node_id"],
        node_type=node_type_val,  # Pass string directly
        label=data.get("label", "unknown"),
        labels=data.get("labels", []),
        embedding=data.get("embedding"),
        activation=data.get("activation", 0.0),
        base_activation=data.get("base_activation", 0.0),
        created_at=datetime.fromisoformat(data["created_at"]),
        last_accessed=datetime.fromisoformat(data["last_accessed"]),
        access_count=data.get("access_count", 0),
        source_id=data.get("source_id"),
        confidence=data.get("confidence", 0.0),
        extras=data.get("extras", {}),
    )


def _edge_to_dict(edge: GraphEdge) -> dict[str, Any]:
    """Convert a GraphEdge to a JSON-serializable dictionary."""
    # edge_type is now a string, not an enum with .value
    edge_type_str = edge.edge_type if isinstance(edge.edge_type, str) else edge.edge_type
    return {
        "edge_id": edge.edge_id,
        "edge_type": edge_type_str,
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "weight": edge.weight,
        "confidence": edge.confidence,
        "created_at": edge.created_at.isoformat(),
        "last_accessed": edge.last_accessed.isoformat(),
        "extras": edge.extras,
    }


def _dict_to_edge(data: dict[str, Any]) -> GraphEdge:
    """Convert a dictionary back to a GraphEdge."""
    # edge_type is now a string (not EdgeType enum)
    edge_type_val = data["edge_type"]
    return GraphEdge(
        edge_id=data["edge_id"],
        edge_type=edge_type_val,  # Pass string directly
        source_node_id=data["source_node_id"],
        target_node_id=data["target_node_id"],
        weight=data.get("weight", 1.0),
        confidence=data.get("confidence", 0.0),
        created_at=datetime.fromisoformat(data["created_at"]),
        last_accessed=datetime.fromisoformat(data["last_accessed"]),
        extras=data.get("extras", {}),
    )


class LabeledGraphStore(GraphStore):
    """Graph store with efficient label indexing and neighbor queries.
    
    Supports:
    - Multiple labels per node (primary + alternatives)
    - Fast lookup by label
    - Neighbor queries for spreading activation
    - Optional persistence to JSONL files
    """

    def __init__(
        self,
        nodes_path: Path | None = None,
        edges_path: Path | None = None,
    ) -> None:
        """Initialize the graph store.
        
        Args:
            nodes_path: Optional path for node persistence.
            edges_path: Optional path for edge persistence.
        """
        self._nodes_path = nodes_path
        self._edges_path = edges_path
        
        # Primary storage
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        
        # Indexes
        self._label_to_nodes: dict[str, set[str]] = defaultdict(set)
        self._edges_by_source: dict[str, set[str]] = defaultdict(set)
        self._edges_by_target: dict[str, set[str]] = defaultdict(set)
        self._nodes_by_type: dict[NodeType, set[str]] = defaultdict(set)
        
        # Load existing data if paths provided
        if nodes_path:
            nodes_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_nodes()
        if edges_path:
            edges_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_edges()

    def _load_nodes(self) -> None:
        """Load nodes from storage file."""
        if not self._nodes_path or not self._nodes_path.exists():
            return
            
        with open(self._nodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    node = _dict_to_node(data)
                    self._index_node(node)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    def _load_edges(self) -> None:
        """Load edges from storage file."""
        if not self._edges_path or not self._edges_path.exists():
            return
            
        with open(self._edges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    edge = _dict_to_edge(data)
                    self._index_edge(edge)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    def _index_node(self, node: GraphNode) -> None:
        """Add node to storage and all indexes."""
        self._nodes[node.node_id] = node
        
        # Index by primary label
        if node.label:
            self._label_to_nodes[node.label.lower()].add(node.node_id)
        
        # Index by alternative labels
        for label in node.labels:
            self._label_to_nodes[label.lower()].add(node.node_id)
        
        # Index by type
        self._nodes_by_type[node.node_type].add(node.node_id)

    def _index_edge(self, edge: GraphEdge) -> None:
        """Add edge to storage and indexes."""
        self._edges[edge.edge_id] = edge
        self._edges_by_source[edge.source_node_id].add(edge.edge_id)
        self._edges_by_target[edge.target_node_id].add(edge.edge_id)

    def _persist_node(self, node: GraphNode) -> None:
        """Persist a node to storage if path is set."""
        if not self._nodes_path:
            return
        with open(self._nodes_path, "a", encoding="utf-8") as f:
            line = json.dumps(_node_to_dict(node), separators=(",", ":"))
            f.write(line + "\n")

    def _persist_edge(self, edge: GraphEdge) -> None:
        """Persist an edge to storage if path is set."""
        if not self._edges_path:
            return
        with open(self._edges_path, "a", encoding="utf-8") as f:
            line = json.dumps(_edge_to_dict(edge), separators=(",", ":"))
            f.write(line + "\n")

    # =========================================================================
    # GraphStore interface implementation
    # =========================================================================

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph.
        
        Args:
            node: Node to add.
        """
        self._index_node(node)
        self._persist_node(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph.
        
        Args:
            edge: Edge to add.
        """
        self._index_edge(edge)
        self._persist_edge(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a node by ID.
        
        Args:
            node_id: ID of the node to retrieve.
            
        Returns:
            The GraphNode if found, None otherwise.
        """
        return self._nodes.get(node_id)

    def get_edges(self, node_id: str) -> list[GraphEdge]:
        """Get all edges connected to a node.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            List of edges where node is source or target.
        """
        edge_ids = (
            self._edges_by_source.get(node_id, set()) |
            self._edges_by_target.get(node_id, set())
        )
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_all_nodes(self) -> list[GraphNode]:
        """Retrieve all nodes in the graph.
        
        Returns:
            List of all nodes.
        """
        return list(self._nodes.values())

    # =========================================================================
    # Extended functionality
    # =========================================================================

    def get_all_edges(self) -> list[GraphEdge]:
        """Retrieve all edges in the graph.
        
        Returns:
            List of all edges.
        """
        return list(self._edges.values())

    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return len(self._edges)

    def get_nodes_by_label(self, label: str) -> list[GraphNode]:
        """Get all nodes with a specific label (primary or alternative).
        
        Args:
            label: Label to search for (case-insensitive).
            
        Returns:
            List of nodes with that label.
        """
        node_ids = self._label_to_nodes.get(label.lower(), set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve.
            
        Returns:
            List of nodes of that type.
        """
        node_ids = self._nodes_by_type.get(node_type, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        """Get edges where node is the source.
        
        Args:
            node_id: ID of the source node.
            
        Returns:
            List of outgoing edges.
        """
        edge_ids = self._edges_by_source.get(node_id, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_incoming_edges(self, node_id: str) -> list[GraphEdge]:
        """Get edges where node is the target.
        
        Args:
            node_id: ID of the target node.
            
        Returns:
            List of incoming edges.
        """
        edge_ids = self._edges_by_target.get(node_id, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
    ) -> list[GraphNode]:
        """Get all neighbor nodes connected by edges.
        
        Args:
            node_id: ID of the central node.
            edge_types: Optional filter for edge types.
            
        Returns:
            List of connected neighbor nodes.
        """
        edges = self.get_edges(node_id)
        
        if edge_types:
            edges = [e for e in edges if e.edge_type in edge_types]
        
        neighbor_ids = set()
        for edge in edges:
            if edge.source_node_id == node_id:
                neighbor_ids.add(edge.target_node_id)
            else:
                neighbor_ids.add(edge.source_node_id)
        
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def add_label_to_node(self, node_id: str, label: str) -> bool:
        """Add an alternative label to a node.
        
        Args:
            node_id: ID of the node.
            label: Label to add.
            
        Returns:
            True if added, False if node not found.
        """
        node = self._nodes.get(node_id)
        if not node:
            return False
        
        if label not in node.labels and label != node.label:
            node.labels.append(label)
            self._label_to_nodes[label.lower()].add(node_id)
        
        return True

    def remove_label_from_node(self, node_id: str, label: str) -> bool:
        """Remove an alternative label from a node.
        
        Args:
            node_id: ID of the node.
            label: Label to remove.
            
        Returns:
            True if removed, False if not found.
        """
        node = self._nodes.get(node_id)
        if not node or label == node.label:
            return False  # Can't remove primary label
        
        if label in node.labels:
            node.labels.remove(label)
            self._label_to_nodes[label.lower()].discard(node_id)
            return True
        
        return False

    def label_exists(self, label: str) -> bool:
        """Check if a label is already in use.
        
        Args:
            label: Label to check.
            
        Returns:
            True if label exists on any node.
        """
        return len(self._label_to_nodes.get(label.lower(), set())) > 0

    def create_alias_edge(
        self,
        alias_node_id: str,
        canonical_node_id: str,
    ) -> GraphEdge:
        """Create an alias relationship between nodes.
        
        Args:
            alias_node_id: The alias node.
            canonical_node_id: The canonical/main node.
            
        Returns:
            The created edge.
        """
        edge = GraphEdge(
            edge_id=f"edge_{uuid.uuid4().hex[:12]}",
            edge_type=EdgeType.ALIAS_OF,
            source_node_id=alias_node_id,
            target_node_id=canonical_node_id,
            weight=1.0,
            confidence=1.0,
        )
        self.add_edge(edge)
        return edge

    def create_merge_edge(
        self,
        merged_node_id: str,
        target_node_id: str,
    ) -> GraphEdge:
        """Create a merge relationship (node A merged into node B).
        
        Args:
            merged_node_id: The node that was merged.
            target_node_id: The node it was merged into.
            
        Returns:
            The created edge.
        """
        edge = GraphEdge(
            edge_id=f"edge_{uuid.uuid4().hex[:12]}",
            edge_type=EdgeType.MERGED_INTO,
            source_node_id=merged_node_id,
            target_node_id=target_node_id,
            weight=1.0,
            confidence=1.0,
        )
        self.add_edge(edge)
        return edge

    def update_node_access(self, node_id: str) -> None:
        """Update last_accessed timestamp and access_count for a node.
        
        Args:
            node_id: ID of the node to update.
        """
        node = self._nodes.get(node_id)
        if node:
            node.last_accessed = datetime.now()
            node.access_count += 1

    def clear(self) -> None:
        """Clear all nodes and edges (in-memory only)."""
        self._nodes.clear()
        self._edges.clear()
        self._label_to_nodes.clear()
        self._edges_by_source.clear()
        self._edges_by_target.clear()
        self._nodes_by_type.clear()
