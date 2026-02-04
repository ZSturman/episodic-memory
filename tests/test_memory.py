"""Tests for memory stores (episode and graph)."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from episodic_agent.memory.episode_store import PersistentEpisodeStore
from episodic_agent.memory.graph_store import LabeledGraphStore
from episodic_agent.memory.stubs import InMemoryEpisodeStore, InMemoryGraphStore
from episodic_agent.schemas import Episode, GraphEdge, GraphNode
from episodic_agent.schemas.graph import EdgeType, NodeType


# =============================================================================
# InMemoryEpisodeStore Tests
# =============================================================================

class TestInMemoryEpisodeStore:
    """Tests for in-memory episode store."""

    def _make_episode(self, episode_id: str, location: str = "test") -> Episode:
        """Helper to create test episodes."""
        return Episode(
            episode_id=episode_id,
            source_acf_id=f"acf_{episode_id}",
            location_label=location,
            start_time=datetime.now(),
            end_time=datetime.now(),
            step_count=10,
        )

    def test_store_and_get(self):
        """Store and retrieve episode."""
        store = InMemoryEpisodeStore()
        episode = self._make_episode("ep_001", "Kitchen")
        
        store.store(episode)
        retrieved = store.get("ep_001")
        
        assert retrieved is not None
        assert retrieved.episode_id == "ep_001"
        assert retrieved.location_label == "Kitchen"

    def test_get_nonexistent(self):
        """Get returns None for missing episode."""
        store = InMemoryEpisodeStore()
        
        result = store.get("nonexistent")
        
        assert result is None

    def test_list_all(self):
        """List all stored episodes."""
        store = InMemoryEpisodeStore()
        
        for i in range(5):
            store.store(self._make_episode(f"ep_{i:03d}"))
        
        episodes = store.list_all()
        
        assert len(episodes) == 5

    def test_count(self):
        """Count returns correct number."""
        store = InMemoryEpisodeStore()
        
        assert store.count() == 0
        
        store.store(self._make_episode("ep_001"))
        assert store.count() == 1
        
        store.store(self._make_episode("ep_002"))
        assert store.count() == 2


# =============================================================================
# PersistentEpisodeStore Tests
# =============================================================================

class TestPersistentEpisodeStore:
    """Tests for persistent JSONL episode store."""

    def _make_episode(self, episode_id: str, location: str = "test") -> Episode:
        """Helper to create test episodes."""
        return Episode(
            episode_id=episode_id,
            source_acf_id=f"acf_{episode_id}",
            location_label=location,
            start_time=datetime.now(),
            end_time=datetime.now(),
            step_count=10,
        )

    def test_store_creates_file(self):
        """Storing episode creates JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episodes.jsonl"
            store = PersistentEpisodeStore(path)
            
            store.store(self._make_episode("ep_001"))
            
            assert path.exists()

    def test_persistence(self):
        """Episodes persist to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episodes.jsonl"
            store = PersistentEpisodeStore(path)
            
            store.store(self._make_episode("ep_001", "Kitchen"))
            store.store(self._make_episode("ep_002", "Bedroom"))
            
            # Read file directly
            with open(path) as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            assert "Kitchen" in lines[0]
            assert "Bedroom" in lines[1]

    def test_load_on_init(self):
        """Existing episodes loaded on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episodes.jsonl"
            
            # Store with first instance
            store1 = PersistentEpisodeStore(path)
            store1.store(self._make_episode("ep_001", "Kitchen"))
            store1.store(self._make_episode("ep_002", "Bedroom"))
            
            # Load with new instance
            store2 = PersistentEpisodeStore(path)
            
            assert store2.count() == 2
            assert store2.get("ep_001").location_label == "Kitchen"

    def test_list_by_location(self):
        """Filter episodes by location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episodes.jsonl"
            store = PersistentEpisodeStore(path)
            
            store.store(self._make_episode("ep_001", "Kitchen"))
            store.store(self._make_episode("ep_002", "Bedroom"))
            store.store(self._make_episode("ep_003", "Kitchen"))
            
            kitchen = store.list_by_location("Kitchen")
            
            assert len(kitchen) == 2
            assert all(e.location_label == "Kitchen" for e in kitchen)

    def test_list_recent(self):
        """Get most recent episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "episodes.jsonl"
            store = PersistentEpisodeStore(path)
            
            for i in range(10):
                store.store(self._make_episode(f"ep_{i:03d}"))
            
            recent = store.list_recent(3)
            
            assert len(recent) == 3


# =============================================================================
# InMemoryGraphStore Tests
# =============================================================================

class TestInMemoryGraphStore:
    """Tests for in-memory graph store."""

    def _make_node(self, node_id: str, node_type: NodeType, label: str) -> GraphNode:
        """Helper to create test nodes."""
        return GraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def _make_edge(self, edge_id: str, source: str, target: str) -> GraphEdge:
        """Helper to create test edges."""
        return GraphEdge(
            edge_id=edge_id,
            edge_type=EdgeType.TYPICAL_IN,
            source_id=source,
            target_id=target,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def test_add_and_get_node(self):
        """Add and retrieve node."""
        store = InMemoryGraphStore()
        node = self._make_node("loc_001", NodeType.LOCATION, "Kitchen")
        
        store.add_node(node)
        retrieved = store.get_node("loc_001")
        
        assert retrieved is not None
        assert retrieved.label == "Kitchen"

    def test_get_nodes_by_type(self):
        """Filter nodes by type."""
        store = InMemoryGraphStore()
        
        store.add_node(self._make_node("loc_001", NodeType.LOCATION, "Kitchen"))
        store.add_node(self._make_node("loc_002", NodeType.LOCATION, "Bedroom"))
        store.add_node(self._make_node("ent_001", NodeType.ENTITY, "Door"))
        
        locations = store.get_nodes_by_type(NodeType.LOCATION)
        
        assert len(locations) == 2

    def test_add_and_get_edges(self):
        """Add and retrieve edges."""
        store = InMemoryGraphStore()
        
        store.add_node(self._make_node("ent_001", NodeType.ENTITY, "Door"))
        store.add_node(self._make_node("loc_001", NodeType.LOCATION, "Kitchen"))
        
        edge = self._make_edge("e_001", "ent_001", "loc_001")
        store.add_edge(edge)
        
        edges_from = store.get_edges_from("ent_001")
        edges_to = store.get_edges_to("loc_001")
        
        assert len(edges_from) == 1
        assert len(edges_to) == 1


# =============================================================================
# LabeledGraphStore Tests
# =============================================================================

class TestLabeledGraphStore:
    """Tests for persistent labeled graph store."""

    def _make_node(self, node_id: str, node_type: NodeType, label: str) -> GraphNode:
        """Helper to create test nodes."""
        return GraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def _make_edge(self, edge_id: str, source: str, target: str) -> GraphEdge:
        """Helper to create test edges."""
        return GraphEdge(
            edge_id=edge_id,
            edge_type=EdgeType.TYPICAL_IN,
            source_id=source,
            target_id=target,
            weight=1.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    def test_persistence(self):
        """Nodes and edges persist to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_path = Path(tmpdir) / "nodes.jsonl"
            edges_path = Path(tmpdir) / "edges.jsonl"
            
            store = LabeledGraphStore(nodes_path, edges_path)
            
            store.add_node(self._make_node("loc_001", NodeType.LOCATION, "Kitchen"))
            store.add_edge(self._make_edge("e_001", "ent_001", "loc_001"))
            
            assert nodes_path.exists()
            assert edges_path.exists()

    def test_load_on_init(self):
        """Existing data loaded on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_path = Path(tmpdir) / "nodes.jsonl"
            edges_path = Path(tmpdir) / "edges.jsonl"
            
            # Store with first instance
            store1 = LabeledGraphStore(nodes_path, edges_path)
            store1.add_node(self._make_node("loc_001", NodeType.LOCATION, "Kitchen"))
            store1.add_node(self._make_node("ent_001", NodeType.ENTITY, "Door"))
            store1.add_edge(self._make_edge("e_001", "ent_001", "loc_001"))
            
            # Load with new instance
            store2 = LabeledGraphStore(nodes_path, edges_path)
            
            assert store2.get_node("loc_001") is not None
            assert store2.get_node("ent_001") is not None
            assert len(store2.get_edges_from("ent_001")) == 1

    def test_update_edge_weight(self):
        """Edge weight can be updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_path = Path(tmpdir) / "nodes.jsonl"
            edges_path = Path(tmpdir) / "edges.jsonl"
            
            store = LabeledGraphStore(nodes_path, edges_path)
            
            store.add_edge(self._make_edge("e_001", "src", "tgt"))
            store.update_edge_weight("e_001", 5.0)
            
            edges = store.get_edges_from("src")
            assert edges[0].weight == 5.0

    def test_get_or_create_node(self):
        """Get existing node or create new one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nodes_path = Path(tmpdir) / "nodes.jsonl"
            edges_path = Path(tmpdir) / "edges.jsonl"
            
            store = LabeledGraphStore(nodes_path, edges_path)
            
            # First call creates
            node1 = store.get_or_create_node("loc_001", NodeType.LOCATION, "Kitchen")
            assert node1.label == "Kitchen"
            
            # Second call returns existing
            node2 = store.get_or_create_node("loc_001", NodeType.LOCATION, "Different")
            assert node2.label == "Kitchen"  # Original label preserved
