# Module Interfaces

All modules implement abstract base classes defined in `src/episodic_agent/core/interfaces.py`. This enables dependency injection and module swapping without changing the orchestrator.

---

## Architectural Invariants

These invariants are **non-negotiable** and must be enforced by all module implementations.

### ACF Stability Invariant

**Rule:** A location label persists across perceptual variation unless strong contradictory evidence accumulates.

| Scenario | Expected Behavior |
|----------|------------------|
| Lighting/shadow change | Identity PERSISTS - not a new location |
| Temporary clutter | Identity PERSISTS - objects don't define location |
| Sudden visual change WITHOUT motion | ANOMALY - investigate, don't auto-relabel |
| Sustained mismatch + spatial contradiction | Enter UNCERTAINTY state |

**Implementation:** `ACFStabilityGuard` in `modules/acf_stability.py`

```python
from episodic_agent.modules import ACFStabilityGuard, StabilityState

guard = ACFStabilityGuard()
decision = guard.evaluate_stability(percept, acf, motion_detected=False)

if not decision.identity_stable:
    # Only after sustained mismatch WITH motion
    allow_location_transition()
elif decision.new_state == StabilityState.UNCERTAIN:
    # Flag for investigation or user confirmation
    request_clarification()
```

### Motion Advisory / Perception Authoritative Invariant

**Rule:** If motion implies new location but visual ACF matches previous location, enter uncertainty state and investigate. Never trust motion alone.

| Scenario | Motion Says | Perception Says | Outcome |
|----------|-------------|-----------------|---------|
| Normal walk | New location | New location | Allow transition |
| Elevator | New floor | Same appearance | INVESTIGATE - don't auto-relabel |
| Teleport | No movement | Different location | ANOMALY - investigate |
| Stationary | Same location | Same location | Stable |

**Implementation:** `MotionPerceptionArbitrator` in `modules/arbitrator.py`

```python
from episodic_agent.modules import MotionPerceptionArbitrator, ArbitrationOutcome

arbitrator = MotionPerceptionArbitrator()
decision = arbitrator.arbitrate(motion_signal, perception_signal)

if decision.outcome == ArbitrationOutcome.TRUST_PERCEPTION:
    # Perception is authoritative
    location = decision.resolved_location
elif decision.outcome == ArbitrationOutcome.INVESTIGATING:
    # Conflict detected - gathering evidence
    continue_investigation()
elif decision.outcome == ArbitrationOutcome.UNCERTAIN:
    # Needs user confirmation
    request_user_label()
```

### No Absolute Coordinates Invariant

**Rule:** Absolute world coordinates must never cross the wire and never be stored in semantic memory.

| Allowed | Not Allowed |
|---------|-------------|
| Agent-relative positions | World coordinates in protocol |
| Landmark-relative offsets | Absolute positions in graph |
| Qualitative relations ("near", "left of") | Raw (x, y, z) in persistent storage |

### Backend-Owned Cognition Invariant

**Rule:** Python owns ALL labeling logic, recognition reasoning, and memory. Unity/client is stateless.

| Backend (Python) Responsibilities | Client (Unity) Responsibilities |
|----------------------------------|--------------------------------|
| Label assignment | Render visuals |
| Entity recognition | Stream sensor data |
| Location resolution | Relay user input |
| Event interpretation | Display backend-provided labels |
| Memory storage | NOTHING semantic |

---

## Interface Summary

| Interface | Methods | Purpose |
|-----------|---------|---------|
| `SensorProvider` | `get_frame()`, `has_frames()` | Raw sensor data source |
| `PerceptionModule` | `process(frame)` | Frame to percept conversion |
| `ACFBuilder` | `create_acf()`, `update_acf()` | Working memory management |
| `LocationResolver` | `resolve(percept, acf)` | Location identification |
| `EntityResolver` | `resolve(percept, acf)` | Entity recognition |
| `EventResolver` | `resolve(percept, acf)` | Event detection |
| `BoundaryDetector` | `check(acf)` | Episode segmentation |
| `Retriever` | `retrieve(acf)` | Memory queries |
| `DialogManager` | `ask_label()`, `notify()` | User interaction |
| `EpisodeStore` | `store()`, `get()`, `list_all()` | Episode persistence |
| `GraphStore` | `add_node()`, `add_edge()`, etc. | Graph persistence |
| **`ACFStabilityGuard`** | `evaluate_stability()` | **Identity stability enforcement** |
| **`MotionPerceptionArbitrator`** | `arbitrate()` | **Motion vs perception resolution** |

---

## SensorProvider

Provides raw sensor data from any source.

```python
class SensorProvider(ABC):
    @abstractmethod
    def get_frame(self) -> SensorFrame:
        """Get the next sensor frame."""
        ...

    @abstractmethod
    def has_frames(self) -> bool:
        """Check if more frames are available."""
        ...

    def reset(self) -> None:
        """Optional: Reset to initial state."""
        pass
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubSensorProvider` | `modules.stubs` | Synthetic test data |
| `UnityWebSocketSensorProvider` | `modules.unity.sensor_provider` | Unity WebSocket stream |
| `ReplaySensorProvider` | `modules.unity.sensor_provider` | JSONL file replay |

---

## PerceptionModule

Converts raw sensor frames into structured percepts with embeddings.

```python
class PerceptionModule(ABC):
    @abstractmethod
    def process(self, frame: SensorFrame) -> Percept:
        """Process a sensor frame into a percept."""
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubPerception` | `modules.stubs` | Returns empty percept |
| `PerceptionUnityCheat` | `modules.unity.perception` | GUID-to-embedding conversion |

---

## ACFBuilder

Manages the Active Context Frame (working memory).

```python
class ACFBuilder(ABC):
    @abstractmethod
    def create_acf(self) -> ActiveContextFrame:
        """Create a new empty ACF."""
        ...

    @abstractmethod
    def update_acf(
        self,
        acf: ActiveContextFrame,
        percept: Percept,
    ) -> ActiveContextFrame:
        """Update ACF with new perception data."""
        ...
```

---

## LocationResolver

Determines the current location from perception data.

```python
class LocationResolver(ABC):
    @abstractmethod
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> tuple[str, float]:
        """Resolve current location.
        
        Returns:
            Tuple of (location_label, confidence).
        """
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubLocationResolver` | `modules.stubs` | Returns "unknown" |
| `LocationResolverCheat` | `modules.unity.resolvers` | Uses Unity room GUID |

---

## EntityResolver

Identifies entities in the current scene.

```python
class EntityResolver(ABC):
    @abstractmethod
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[ObjectCandidate]:
        """Resolve entities from percept.
        
        Returns:
            List of resolved ObjectCandidates.
        """
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubEntityResolver` | `modules.stubs` | Returns empty list |
| `EntityResolverCheat` | `modules.unity.resolvers` | Uses Unity entity GUIDs |

---

## EventResolver

Detects state changes and events.

```python
class EventResolver(ABC):
    @abstractmethod
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[dict]:
        """Detect events from perception changes.
        
        Returns:
            List of detected event dictionaries.
        """
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubEventResolver` | `modules.stubs` | Returns empty list |
| `EventResolverStateChange` | `modules.event_resolver` | Detects state-change events |

---

## BoundaryDetector

Determines when to segment episodes.

```python
class BoundaryDetector(ABC):
    @abstractmethod
    def check(
        self,
        acf: ActiveContextFrame,
    ) -> tuple[bool, str | None]:
        """Check if episode boundary should be triggered.
        
        Returns:
            Tuple of (should_freeze, reason).
        """
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubBoundaryDetector` | `modules.stubs` | Time-based only |
| `HysteresisBoundaryDetector` | `modules.boundary` | Multi-factor with hysteresis |

---

## Retriever

Queries memory stores for relevant information.

```python
class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        acf: ActiveContextFrame,
    ) -> RetrievalResult:
        """Query memory based on current context.
        
        Returns:
            RetrievalResult with relevant episodes and nodes.
        """
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubRetriever` | `modules.stubs` | Returns empty result |
| `SpreadingActivationRetriever` | `modules.retriever` | Graph-based spreading activation |

---

## DialogManager

Handles user interactions for labeling and notifications.

```python
class DialogManager(ABC):
    @abstractmethod
    def ask_label(
        self,
        prompt: str,
        suggestions: list[str] | None = None,
    ) -> str:
        """Ask user for a label.
        
        Returns:
            User-provided label string.
        """
        ...

    @abstractmethod
    def notify(self, message: str) -> None:
        """Display a notification to user."""
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `StubDialogManager` | `modules.stubs` | Auto-generates labels |
| `CLIDialogManager` | `modules.dialog` | Interactive CLI prompts |
| `AutoAcceptDialogManager` | `modules.dialog` | Auto-accepts suggestions |

---

## EpisodeStore

Persists frozen episodes.

```python
class EpisodeStore(ABC):
    @abstractmethod
    def store(self, episode: Episode) -> None:
        """Store a frozen episode."""
        ...

    @abstractmethod
    def get(self, episode_id: str) -> Episode | None:
        """Retrieve episode by ID."""
        ...

    @abstractmethod
    def list_all(self) -> list[Episode]:
        """List all stored episodes."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Get total episode count."""
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `InMemoryEpisodeStore` | `memory.stubs` | In-memory only |
| `PersistentEpisodeStore` | `memory.episode_store` | JSONL persistence |

---

## GraphStore

Maintains associative graph memory.

```python
class GraphStore(ABC):
    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        """Add or update a node."""
        ...

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        """Add or update an edge."""
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None:
        """Get node by ID."""
        ...

    @abstractmethod
    def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        """Get all edges from a node."""
        ...

    @abstractmethod
    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Get all nodes of a type."""
        ...
```

### Implementations

| Class | Module | Description |
|-------|--------|-------------|
| `InMemoryGraphStore` | `memory.stubs` | In-memory only |
| `LabeledGraphStore` | `memory.graph_store` | JSONL persistence |

---

## Implementing Custom Modules

To create a custom module:

1. Inherit from the appropriate interface
2. Implement all abstract methods
3. Register in profiles or inject via CLI

Example:

```python
from episodic_agent.core.interfaces import LocationResolver
from episodic_agent.schemas import Percept, ActiveContextFrame

class MyLocationResolver(LocationResolver):
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> tuple[str, float]:
        # Custom location logic
        label = self._determine_location(percept)
        confidence = self._compute_confidence(percept)
        return label, confidence
```

See [Creating Custom Profiles](../modules/README.md#custom-profiles) for integration details.
