"""Abstract base classes for all swappable modules.

These interfaces define the contract between the orchestrator and
module implementations. All modules should depend only on these
interfaces and the schemas - never on concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episodic_agent.schemas import (
        ActiveContextFrame,
        Episode,
        GraphEdge,
        GraphNode,
        ObjectCandidate,
        Percept,
        RetrievalResult,
        SensorFrame,
    )


# =============================================================================
# Sensor and Perception Interfaces
# =============================================================================


class SensorProvider(ABC):
    """Abstract interface for sensor data sources.
    
    Implementations might include:
    - Unity WebSocket stream
    - File replay
    - Synthetic test data
    """

    @abstractmethod
    def get_frame(self) -> SensorFrame:
        """Get the next sensor frame.
        
        Returns:
            The next available SensorFrame from the sensor source.
        """
        ...

    @abstractmethod
    def has_frames(self) -> bool:
        """Check if more frames are available.
        
        Returns:
            True if more frames can be retrieved.
        """
        ...

    def reset(self) -> None:
        """Reset the sensor to its initial state.
        
        Optional - implementations may choose not to support reset.
        """
        pass


class PerceptionModule(ABC):
    """Abstract interface for perception processing.
    
    Converts raw sensor frames into structured percepts with
    embeddings and object candidates.
    """

    @abstractmethod
    def process(self, frame: SensorFrame) -> Percept:
        """Process a sensor frame into a percept.
        
        Args:
            frame: Raw sensor frame to process.
            
        Returns:
            Processed Percept with embeddings and candidates.
        """
        ...


# =============================================================================
# Context and Resolution Interfaces
# =============================================================================


class ACFBuilder(ABC):
    """Abstract interface for Active Context Frame management.
    
    Responsible for creating, updating, and managing the mutable
    working memory that accumulates episode information.
    """

    @abstractmethod
    def create_acf(self) -> ActiveContextFrame:
        """Create a new empty ACF.
        
        Returns:
            A fresh ActiveContextFrame instance.
        """
        ...

    @abstractmethod
    def update_acf(
        self,
        acf: ActiveContextFrame,
        percept: Percept,
    ) -> ActiveContextFrame:
        """Update ACF with new perception data.
        
        Args:
            acf: Current ACF to update.
            percept: New percept to incorporate.
            
        Returns:
            Updated ACF (may be same instance, mutated).
        """
        ...


class LocationResolver(ABC):
    """Abstract interface for location resolution.
    
    Determines the current location from perception data,
    matching against known locations or identifying new ones.
    """

    @abstractmethod
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> tuple[str, float]:
        """Resolve current location from percept.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            Tuple of (location_label, confidence).
        """
        ...


class EntityResolver(ABC):
    """Abstract interface for entity resolution.
    
    Identifies and tracks entities from perception data,
    matching against known entities or creating new candidates.
    """

    @abstractmethod
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[ObjectCandidate]:
        """Resolve entities from percept.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            List of resolved ObjectCandidates.
        """
        ...


class EventResolver(ABC):
    """Abstract interface for event detection.
    
    Detects state changes and events by comparing current
    perception to previous context.
    """

    @abstractmethod
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[dict]:
        """Detect events from perception changes.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            List of detected event dictionaries.
        """
        ...


# =============================================================================
# Memory and Retrieval Interfaces
# =============================================================================


class Retriever(ABC):
    """Abstract interface for memory retrieval.
    
    Queries episodic and graph memory to find relevant
    past experiences and associations.
    """

    @abstractmethod
    def retrieve(
        self,
        acf: ActiveContextFrame,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve relevant memories for current context.
        
        Args:
            acf: Current active context frame as query.
            top_k: Maximum number of results to return.
            
        Returns:
            RetrievalResult with ranked episodes and nodes.
        """
        ...


class EpisodeStore(ABC):
    """Abstract interface for episode storage.
    
    Persists frozen episodes as append-only records.
    """

    @abstractmethod
    def store(self, episode: Episode) -> None:
        """Store a frozen episode.
        
        Args:
            episode: Episode to persist.
        """
        ...

    @abstractmethod
    def get(self, episode_id: str) -> Episode | None:
        """Retrieve an episode by ID.
        
        Args:
            episode_id: ID of the episode to retrieve.
            
        Returns:
            The Episode if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_all(self) -> list[Episode]:
        """Retrieve all stored episodes.
        
        Returns:
            List of all episodes in storage.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Get the number of stored episodes.
        
        Returns:
            Count of episodes in storage.
        """
        ...


class GraphStore(ABC):
    """Abstract interface for graph memory storage.
    
    Manages nodes and edges for associative retrieval.
    """

    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph.
        
        Args:
            node: Node to add.
        """
        ...

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph.
        
        Args:
            edge: Edge to add.
        """
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a node by ID.
        
        Args:
            node_id: ID of the node to retrieve.
            
        Returns:
            The GraphNode if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_edges(self, node_id: str) -> list[GraphEdge]:
        """Get all edges connected to a node.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            List of edges where node is source or target.
        """
        ...

    @abstractmethod
    def get_all_nodes(self) -> list[GraphNode]:
        """Retrieve all nodes in the graph.
        
        Returns:
            List of all nodes.
        """
        ...


class VectorIndex(ABC):
    """Abstract interface for vector similarity search.
    
    Optional component for fast approximate nearest neighbor
    search on embeddings.
    """

    @abstractmethod
    def add(self, id: str, embedding: list[float]) -> None:
        """Add an embedding to the index.
        
        Args:
            id: Identifier for this embedding.
            embedding: Vector to index.
        """
        ...

    @abstractmethod
    def search(
        self,
        query: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings.
        
        Args:
            query: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of (id, score) tuples ranked by similarity.
        """
        ...


# =============================================================================
# Boundary and Dialog Interfaces
# =============================================================================


class BoundaryDetector(ABC):
    """Abstract interface for episode boundary detection.
    
    Determines when to freeze the current ACF into an episode
    based on various signals (location change, time, prediction error).
    """

    @abstractmethod
    def check(self, acf: ActiveContextFrame) -> tuple[bool, str | None]:
        """Check if an episode boundary should be triggered.
        
        Args:
            acf: Current active context frame.
            
        Returns:
            Tuple of (should_freeze, reason) where reason is None
            if no boundary detected.
        """
        ...


class DialogManager(ABC):
    """Abstract interface for user interaction.
    
    Handles label confirmation, conflict resolution, and
    other user-driven interactions. Can operate in CLI mode
    or auto-accept mode for automated testing.
    """

    @abstractmethod
    def request_label(
        self,
        prompt: str,
        candidates: list[str],
    ) -> str | None:
        """Request a label from the user.
        
        Args:
            prompt: Description of what needs labeling.
            candidates: Suggested label options.
            
        Returns:
            Selected label or None if canceled.
        """
        ...

    @abstractmethod
    def confirm(self, prompt: str, default: bool = True) -> bool:
        """Request confirmation from the user.
        
        Args:
            prompt: Yes/no question to ask.
            default: Default response if skipped.
            
        Returns:
            True for yes, False for no.
        """
        ...

    @abstractmethod
    def resolve_conflict(
        self,
        prompt: str,
        options: list[str],
    ) -> int:
        """Resolve a conflict between options.
        
        Args:
            prompt: Description of the conflict.
            options: Available resolution options.
            
        Returns:
            Index of the selected option.
        """
        ...


# =============================================================================
# Optional Consolidation Interface
# =============================================================================


class Consolidator(ABC):
    """Abstract interface for memory consolidation.
    
    Optional component that performs background processing
    on episodic memory (compression, abstraction, forgetting).
    """

    @abstractmethod
    def consolidate(
        self,
        episode_store: EpisodeStore,
        graph_store: GraphStore,
    ) -> None:
        """Run consolidation on memory stores.
        
        Args:
            episode_store: Episode storage to consolidate.
            graph_store: Graph storage to update.
        """
        ...
