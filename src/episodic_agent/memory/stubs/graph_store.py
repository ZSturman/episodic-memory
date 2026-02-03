"""In-memory graph store for Phase 1 testing."""

from __future__ import annotations

from episodic_agent.core.interfaces import GraphStore
from episodic_agent.schemas import GraphEdge, GraphNode


class InMemoryGraphStore(GraphStore):
    """Simple in-memory graph storage.
    
    Stores nodes and edges in dictionaries. Not persisted across runs.
    Suitable for testing and Phase 1 demonstrations.
    """

    def __init__(self) -> None:
        """Initialize an empty graph store."""
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        # Index for fast edge lookups by node
        self._edges_by_node: dict[str, list[str]] = {}

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph.
        
        Args:
            node: Node to add.
        """
        self._nodes[node.node_id] = node
        if node.node_id not in self._edges_by_node:
            self._edges_by_node[node.node_id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph.
        
        Args:
            edge: Edge to add.
        """
        self._edges[edge.edge_id] = edge
        
        # Update edge index for both endpoints
        if edge.source_node_id not in self._edges_by_node:
            self._edges_by_node[edge.source_node_id] = []
        self._edges_by_node[edge.source_node_id].append(edge.edge_id)
        
        if edge.target_node_id not in self._edges_by_node:
            self._edges_by_node[edge.target_node_id] = []
        self._edges_by_node[edge.target_node_id].append(edge.edge_id)

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
        edge_ids = self._edges_by_node.get(node_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_all_nodes(self) -> list[GraphNode]:
        """Retrieve all nodes in the graph.
        
        Returns:
            List of all nodes.
        """
        return list(self._nodes.values())

    def get_all_edges(self) -> list[GraphEdge]:
        """Retrieve all edges in the graph.
        
        Returns:
            List of all edges.
        """
        return list(self._edges.values())

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._nodes.clear()
        self._edges.clear()
        self._edges_by_node.clear()

    def node_count(self) -> int:
        """Get the number of nodes in the graph.
        
        Returns:
            Count of nodes.
        """
        return len(self._nodes)

    def edge_count(self) -> int:
        """Get the number of edges in the graph.
        
        Returns:
            Count of edges.
        """
        return len(self._edges)
