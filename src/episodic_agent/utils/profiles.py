"""Profile configuration system for episodic memory agent.

Profiles define which module implementations to use for different
deployment scenarios (testing, Unity integration, production, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from episodic_agent.core.interfaces import (
        ACFBuilder,
        BoundaryDetector,
        DialogManager,
        EntityResolver,
        EpisodeStore,
        EventResolver,
        GraphStore,
        LocationResolver,
        PerceptionModule,
        Retriever,
        SensorProvider,
    )


class ProfileName(str, Enum):
    """Available profile names."""
    
    STUB = "stub"           # All stub modules (Phase 1 testing)
    UNITY_CHEAT = "unity_cheat"  # Unity with cheat perception
    UNITY_FULL = "unity_full"    # Phase 6: Full features with spreading activation
    # Future profiles:
    # UNITY_REAL = "unity_real"  # Unity with real perception
    # FILE_REPLAY = "file_replay"  # Replay from recorded data


@dataclass
class ProfileConfig:
    """Configuration for a specific profile.
    
    Specifies which module implementations to use and their parameters.
    """
    
    name: str
    description: str
    
    # Module class names/factories (will be instantiated by CLI)
    sensor_provider: str
    perception: str
    acf_builder: str
    location_resolver: str
    entity_resolver: str
    event_resolver: str
    retriever: str
    boundary_detector: str
    dialog_manager: str
    episode_store: str
    graph_store: str
    
    # Profile-specific parameters
    parameters: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Profile Definitions
# ============================================================================

STUB_PROFILE = ProfileConfig(
    name="stub",
    description="All stub modules for Phase 1 testing",
    sensor_provider="StubSensorProvider",
    perception="StubPerception",
    acf_builder="StubACFBuilder",
    location_resolver="StubLocationResolver",
    entity_resolver="StubEntityResolver",
    event_resolver="StubEventResolver",
    retriever="StubRetriever",
    boundary_detector="StubBoundaryDetector",
    dialog_manager="StubDialogManager",
    episode_store="InMemoryEpisodeStore",
    graph_store="InMemoryGraphStore",
    parameters={
        "auto_accept_dialogs": True,
    },
)

UNITY_CHEAT_PROFILE = ProfileConfig(
    name="unity_cheat",
    description="Unity integration with cheat perception (uses GUIDs)",
    sensor_provider="UnityWebSocketSensorProvider",
    perception="PerceptionUnityCheat",
    acf_builder="StubACFBuilder",  # ACF builder unchanged
    location_resolver="LocationResolverCheat",
    entity_resolver="EntityResolverCheat",
    event_resolver="EventResolverStateChange",  # Phase 5: State change events
    retriever="StubRetriever",  # Will be upgraded in Phase 6
    boundary_detector="StubBoundaryDetector",
    dialog_manager="CLIDialogManager",  # Interactive CLI
    episode_store="PersistentEpisodeStore",
    graph_store="LabeledGraphStore",
    parameters={
        "ws_url": "ws://localhost:8765",
        "auto_label_locations": False,
        "auto_label_entities": True,
        "use_persistent_storage": True,
    },
)


UNITY_FULL_PROFILE = ProfileConfig(
    name="unity_full",
    description="Phase 6: Full features with spreading activation and prediction",
    sensor_provider="UnityWebSocketSensorProvider",
    perception="PerceptionUnityCheat",
    acf_builder="StubACFBuilder",
    location_resolver="LocationResolverCheat",
    entity_resolver="EntityResolverCheat",
    event_resolver="EventResolverStateChange",
    retriever="SpreadingActivationRetriever",  # Phase 6: Spreading activation
    boundary_detector="HysteresisBoundaryDetector",  # Phase 6: Hysteresis with prediction error
    dialog_manager="CLIDialogManager",
    episode_store="PersistentEpisodeStore",
    graph_store="LabeledGraphStore",
    parameters={
        "ws_url": "ws://localhost:8765",
        "auto_label_locations": False,
        "auto_label_entities": True,
        "use_persistent_storage": True,
        # Spreading activation parameters
        "spreading_decay": 0.85,
        "spreading_max_hops": 3,
        "spreading_min_activation": 0.1,
        "spreading_top_k": 5,
        # Hysteresis boundary parameters
        "boundary_high_threshold": 0.7,
        "boundary_prediction_error_threshold": 0.6,
    },
)


# Profile registry
PROFILES: dict[str, ProfileConfig] = {
    "stub": STUB_PROFILE,
    "unity_cheat": UNITY_CHEAT_PROFILE,
    "unity_full": UNITY_FULL_PROFILE,
}


def get_profile(name: str) -> ProfileConfig:
    """Get a profile configuration by name.
    
    Args:
        name: Profile name (case-insensitive).
        
    Returns:
        The ProfileConfig for the specified profile.
        
    Raises:
        ValueError: If profile name is not found.
    """
    name_lower = name.lower()
    if name_lower not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile: {name}. Available: {available}")
    
    return PROFILES[name_lower]


def list_profiles() -> list[tuple[str, str]]:
    """List available profiles with descriptions.
    
    Returns:
        List of (name, description) tuples.
    """
    return [(p.name, p.description) for p in PROFILES.values()]


# ============================================================================
# Module Factory
# ============================================================================

class ModuleFactory:
    """Factory for creating module instances from profile configuration.
    
    Handles module instantiation with proper dependency injection.
    """

    def __init__(
        self,
        profile: ProfileConfig,
        run_dir: Path,
        seed: int = 42,
        **overrides: Any,
    ) -> None:
        """Initialize the module factory.
        
        Args:
            profile: Profile configuration to use.
            run_dir: Directory for persistent storage.
            seed: Random seed for deterministic behavior.
            **overrides: Override profile parameters.
        """
        self.profile = profile
        self.run_dir = run_dir
        self.seed = seed
        
        # Merge parameters with overrides
        self.params = dict(profile.parameters)
        self.params.update(overrides)
        
        # Track created modules for dependency injection
        self._modules: dict[str, Any] = {}
        
        # Ensure run directory exists
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def create_modules(self) -> dict[str, Any]:
        """Create all modules for the profile.
        
        Returns:
            Dict with all instantiated modules ready for orchestrator.
        """
        # Create stores first (others may depend on them)
        episode_store = self._create_episode_store()
        graph_store = self._create_graph_store()
        
        # Store for dependency injection
        self._modules["episode_store"] = episode_store
        self._modules["graph_store"] = graph_store
        
        # Create dialog manager
        dialog_manager = self._create_dialog_manager()
        
        # Create sensor and perception
        sensor = self._create_sensor_provider()
        perception = self._create_perception()
        
        # Create resolvers (may depend on stores/dialog)
        location_resolver = self._create_location_resolver(graph_store, dialog_manager)
        entity_resolver = self._create_entity_resolver(
            graph_store, dialog_manager, location_resolver
        )
        
        # Create remaining modules
        acf_builder = self._create_acf_builder()
        event_resolver = self._create_event_resolver(graph_store, dialog_manager)
        retriever = self._create_retriever()
        boundary_detector = self._create_boundary_detector()
        
        return {
            "sensor": sensor,
            "perception": perception,
            "acf_builder": acf_builder,
            "location_resolver": location_resolver,
            "entity_resolver": entity_resolver,
            "event_resolver": event_resolver,
            "retriever": retriever,
            "boundary_detector": boundary_detector,
            "dialog_manager": dialog_manager,
            "episode_store": episode_store,
            "graph_store": graph_store,
        }

    def _create_sensor_provider(self) -> "SensorProvider":
        """Create the sensor provider."""
        name = self.profile.sensor_provider
        
        if name == "StubSensorProvider":
            from episodic_agent.modules.stubs import StubSensorProvider
            return StubSensorProvider(
                max_frames=self.params.get("max_frames", 1000),
                seed=self.seed,
            )
        
        elif name == "UnityWebSocketSensorProvider":
            from episodic_agent.modules.unity import UnityWebSocketSensorProvider
            return UnityWebSocketSensorProvider(
                ws_url=self.params.get("ws_url", "ws://localhost:8765"),
                buffer_size=self.params.get("buffer_size", 100),
                reconnect_delay=self.params.get("reconnect_delay", 2.0),
            )
        
        raise ValueError(f"Unknown sensor provider: {name}")

    def _create_perception(self) -> "PerceptionModule":
        """Create the perception module."""
        name = self.profile.perception
        
        if name == "StubPerception":
            from episodic_agent.modules.stubs import StubPerception
            return StubPerception(seed=self.seed)
        
        elif name == "PerceptionUnityCheat":
            from episodic_agent.modules.unity import PerceptionUnityCheat
            return PerceptionUnityCheat(
                include_invisible=self.params.get("include_invisible", False),
            )
        
        raise ValueError(f"Unknown perception module: {name}")

    def _create_acf_builder(self) -> "ACFBuilder":
        """Create the ACF builder."""
        name = self.profile.acf_builder
        
        if name == "StubACFBuilder":
            from episodic_agent.modules.stubs import StubACFBuilder
            return StubACFBuilder(seed=self.seed)
        
        raise ValueError(f"Unknown ACF builder: {name}")

    def _create_location_resolver(
        self,
        graph_store: "GraphStore",
        dialog_manager: "DialogManager",
    ) -> "LocationResolver":
        """Create the location resolver."""
        name = self.profile.location_resolver
        
        if name == "StubLocationResolver":
            from episodic_agent.modules.stubs import StubLocationResolver
            return StubLocationResolver(seed=self.seed)
        
        elif name == "LocationResolverCheat":
            from episodic_agent.modules.unity import LocationResolverCheat
            return LocationResolverCheat(
                graph_store=graph_store,
                dialog_manager=dialog_manager,
                auto_label=self.params.get("auto_label_locations", False),
            )
        
        raise ValueError(f"Unknown location resolver: {name}")

    def _create_entity_resolver(
        self,
        graph_store: "GraphStore",
        dialog_manager: "DialogManager",
        location_resolver: "LocationResolver",
    ) -> "EntityResolver":
        """Create the entity resolver."""
        name = self.profile.entity_resolver
        
        if name == "StubEntityResolver":
            from episodic_agent.modules.stubs import StubEntityResolver
            return StubEntityResolver(seed=self.seed)
        
        elif name == "EntityResolverCheat":
            from episodic_agent.modules.unity import EntityResolverCheat
            return EntityResolverCheat(
                graph_store=graph_store,
                dialog_manager=dialog_manager,
                location_resolver=location_resolver,
                auto_label=self.params.get("auto_label_entities", True),
            )
        
        raise ValueError(f"Unknown entity resolver: {name}")

    def _create_event_resolver(
        self,
        graph_store: "GraphStore" = None,
        dialog_manager: "DialogManager" = None,
    ) -> "EventResolver":
        """Create the event resolver."""
        name = self.profile.event_resolver
        
        if name == "StubEventResolver":
            from episodic_agent.modules.stubs import StubEventResolver
            return StubEventResolver(seed=self.seed)
        
        elif name == "EventResolverStateChange":
            from episodic_agent.modules.event_resolver import EventResolverStateChange
            return EventResolverStateChange(
                graph_store=graph_store,
                dialog_manager=dialog_manager,
                auto_label_events=self.params.get("auto_label_events", False),
                prompt_for_unknown_events=self.params.get("prompt_for_unknown_events", True),
            )
        
        raise ValueError(f"Unknown event resolver: {name}")

    def _create_retriever(self) -> "Retriever":
        """Create the retriever."""
        name = self.profile.retriever
        
        if name == "StubRetriever":
            from episodic_agent.modules.stubs import StubRetriever
            return StubRetriever(seed=self.seed)
        
        elif name == "SpreadingActivationRetriever":
            from episodic_agent.modules.retriever import RetrieverSpreadingActivation
            return RetrieverSpreadingActivation(
                graph_store=self._modules.get("graph_store"),
                episode_store=self._modules.get("episode_store"),
                decay_factor=self.params.get("spreading_decay", 0.85),
                max_hops=self.params.get("spreading_max_hops", 3),
                min_activation_threshold=self.params.get("spreading_min_activation", 0.1),
                top_k_default=self.params.get("spreading_top_k", 5),
            )
        
        raise ValueError(f"Unknown retriever: {name}")

    def _create_boundary_detector(self) -> "BoundaryDetector":
        """Create the boundary detector."""
        name = self.profile.boundary_detector
        
        if name == "StubBoundaryDetector":
            from episodic_agent.modules.stubs import StubBoundaryDetector
            return StubBoundaryDetector(
                freeze_interval=self.params.get("freeze_interval", 50),
                seed=self.seed,
            )
        
        elif name == "HysteresisBoundaryDetector":
            from episodic_agent.modules.boundary import HysteresisBoundaryDetector
            return HysteresisBoundaryDetector(
                location_confidence_threshold=self.params.get("boundary_high_threshold", 0.7),
                prediction_error_threshold=self.params.get("boundary_prediction_error_threshold", 0.6),
            )
        
        raise ValueError(f"Unknown boundary detector: {name}")

    def _create_dialog_manager(self) -> "DialogManager":
        """Create the dialog manager."""
        name = self.profile.dialog_manager
        
        if name == "StubDialogManager":
            from episodic_agent.modules.stubs import StubDialogManager
            return StubDialogManager(
                auto_accept=self.params.get("auto_accept_dialogs", True),
            )
        
        elif name == "CLIDialogManager":
            from episodic_agent.modules.dialog import CLIDialogManager
            return CLIDialogManager()
        
        elif name == "AutoAcceptDialogManager":
            from episodic_agent.modules.dialog import AutoAcceptDialogManager
            return AutoAcceptDialogManager()
        
        raise ValueError(f"Unknown dialog manager: {name}")

    def _create_episode_store(self) -> "EpisodeStore":
        """Create the episode store."""
        name = self.profile.episode_store
        
        if name == "InMemoryEpisodeStore":
            from episodic_agent.memory.stubs import InMemoryEpisodeStore
            return InMemoryEpisodeStore()
        
        elif name == "PersistentEpisodeStore":
            from episodic_agent.memory.episode_store import PersistentEpisodeStore
            storage_path = self.run_dir / "episodes.jsonl"
            return PersistentEpisodeStore(storage_path=storage_path)
        
        raise ValueError(f"Unknown episode store: {name}")

    def _create_graph_store(self) -> "GraphStore":
        """Create the graph store."""
        name = self.profile.graph_store
        
        if name == "InMemoryGraphStore":
            from episodic_agent.memory.stubs import InMemoryGraphStore
            return InMemoryGraphStore()
        
        elif name == "LabeledGraphStore":
            from episodic_agent.memory.graph_store import LabeledGraphStore
            
            if self.params.get("use_persistent_storage", True):
                nodes_path = self.run_dir / "nodes.jsonl"
                edges_path = self.run_dir / "edges.jsonl"
                return LabeledGraphStore(
                    nodes_path=nodes_path,
                    edges_path=edges_path,
                )
            else:
                return LabeledGraphStore()
        
        raise ValueError(f"Unknown graph store: {name}")
