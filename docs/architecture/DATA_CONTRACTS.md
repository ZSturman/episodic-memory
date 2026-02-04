# Data Contracts

All data flows through Pydantic v2 models defined in `src/episodic_agent/schemas/`. These contracts ensure type safety and enable validation at module boundaries.

## Design Principles

1. **Immutability** - Use frozen models where appropriate
2. **Forward Compatibility** - All models include `extras: dict` field
3. **Validation** - Pydantic handles all validation
4. **Serialization** - JSON-serializable for logging and persistence

---

## Core Schemas

### SensorFrame

Raw sensor data from any source.

```python
class SensorFrame(BaseModel):
    """Raw sensor input from any source."""
    
    frame_id: int                    # Unique frame identifier
    timestamp: datetime              # Frame timestamp
    raw_data: dict[str, Any]         # Raw sensor payload
    sensor_type: str                 # Source type (e.g., "unity_websocket")
    extras: dict[str, Any] = {}      # Forward compatibility

# Unity-specific extras:
# - current_room: str (room GUID)
# - current_room_label: str (room label if known)
# - entities: list[dict] (visible entities)
# - camera_pose: dict (position, rotation, forward)
# - state_changes: list[dict] (state changes since last frame)
```

**Location**: `schemas/frames.py`

---

### Percept

Processed perception with embeddings and candidates.

```python
class Percept(BaseModel):
    """Processed perception from a sensor frame."""
    
    percept_id: str                           # Unique identifier
    timestamp: datetime                        # Processing timestamp
    scene_embedding: list[float]              # Scene-level embedding vector
    objects: list[ObjectCandidate] = []       # Recognized objects
    raw_frame: SensorFrame | None = None      # Optional: original frame
    extras: dict[str, Any] = {}               # Forward compatibility
```

**Location**: `schemas/frames.py`

---

### ObjectCandidate

A recognized object/entity.

```python
class ObjectCandidate(BaseModel):
    """A recognized object candidate."""
    
    candidate_id: str                  # Unique identifier (often GUID)
    label: str                         # Object label
    confidence: float = 1.0            # Recognition confidence [0, 1]
    embedding: list[float] = []        # Object embedding vector
    position: tuple[float, float, float] | None = None  # 3D position
    extras: dict[str, Any] = {}        # Forward compatibility

# Common extras:
# - guid: str (Unity GUID)
# - category: str (door, furniture, etc.)
# - state: str (Open, Closed, On, Off)
# - visible: bool
# - distance: float
```

**Location**: `schemas/frames.py`

---

### ActiveContextFrame (ACF)

Mutable working memory accumulating episode information.

```python
class ActiveContextFrame(BaseModel):
    """Active context frame - mutable working memory."""
    
    acf_id: str                                    # Unique identifier
    step_count: int = 0                            # Steps in this episode
    location_label: str = "unknown"                # Current location
    location_confidence: float = 0.0               # Location confidence
    entities: list[ObjectCandidate] = []           # Current entities
    events: list[dict] = []                        # Accumulated events
    deltas: list[Delta] = []                       # Recent changes
    start_time: datetime                           # Episode start
    last_update: datetime                          # Last update time
    extras: dict[str, Any] = {}                    # Forward compatibility

    def touch(self) -> None:
        """Update last_update timestamp."""
        self.last_update = datetime.now()
```

**Location**: `schemas/context.py`

---

### Episode

Frozen snapshot of an ACF at episode boundary.

```python
class Episode(BaseModel):
    """Frozen episode snapshot."""
    
    episode_id: str                    # Unique identifier
    source_acf_id: str                 # Original ACF ID
    location_label: str                # Primary location
    start_time: datetime               # Episode start
    end_time: datetime                 # Episode end
    step_count: int                    # Total steps
    entities: list[str] = []           # Entity IDs present
    events: list[dict] = []            # Events that occurred
    summary: str = ""                  # Optional text summary
    boundary_reason: str = ""          # Why episode ended
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/context.py`

---

### Delta

A detected change between steps.

```python
class DeltaType(str, Enum):
    """Types of deltas."""
    
    NEW = "new"                 # Entity appeared
    MISSING = "missing"         # Entity disappeared
    MOVED = "moved"             # Entity position changed
    STATE_CHANGED = "state_changed"  # Entity state changed


class Delta(BaseModel):
    """A detected change."""
    
    delta_type: DeltaType           # Type of change
    entity_id: str                  # Affected entity
    entity_label: str = ""          # Entity label
    old_value: Any = None           # Previous value
    new_value: Any = None           # New value
    extras: dict[str, Any] = {}     # Forward compatibility
```

**Location**: `schemas/events.py`

---

### EventCandidate

A detected event.

```python
class EventType(str, Enum):
    """Types of events."""
    
    STATE_CHANGE = "state_change"    # Entity state changed
    APPEARANCE = "appearance"         # Entity appeared
    DISAPPEARANCE = "disappearance"  # Entity disappeared
    MOVEMENT = "movement"            # Entity moved
    INTERACTION = "interaction"      # User interaction
    LOCATION_CHANGE = "location_change"  # Location changed


class EventCandidate(BaseModel):
    """A detected event."""
    
    event_type: EventType              # Type of event
    entity_id: str                     # Primary entity involved
    label: str = ""                    # Event label
    confidence: float = 1.0            # Detection confidence
    timestamp: datetime                # When event occurred
    details: dict[str, Any] = {}       # Event-specific details
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/events.py`

---

### GraphNode

A node in the associative memory graph.

```python
class NodeType(str, Enum):
    """Types of graph nodes."""
    
    LOCATION = "location"
    ENTITY = "entity"
    EVENT = "event"
    CONCEPT = "concept"


class GraphNode(BaseModel):
    """A node in the graph store."""
    
    node_id: str                       # Unique identifier
    node_type: NodeType                # Type of node
    label: str                         # Human-readable label
    embedding: list[float] = []        # Optional embedding vector
    properties: dict[str, Any] = {}    # Node-specific properties
    created_at: datetime               # Creation time
    updated_at: datetime               # Last update time
    extras: dict[str, Any] = {}        # Forward compatibility

# Common properties by type:
# - location: guid, room_label
# - entity: guid, category, typical_state
# - event: event_type, pattern_count
```

**Location**: `schemas/graph.py`

---

### GraphEdge

An edge connecting nodes in the graph.

```python
class EdgeType(str, Enum):
    """Types of graph edges."""
    
    TYPICAL_IN = "typical_in"          # Entity typically in location
    RELATED_TO = "related_to"          # General relation
    CAUSED_BY = "caused_by"            # Causal relation
    PART_OF = "part_of"                # Compositional
    CO_OCCURS = "co_occurs"            # Temporal co-occurrence


class GraphEdge(BaseModel):
    """An edge in the graph store."""
    
    edge_id: str                       # Unique identifier
    edge_type: EdgeType                # Type of edge
    source_id: str                     # Source node ID
    target_id: str                     # Target node ID
    weight: float = 1.0                # Edge weight/strength
    properties: dict[str, Any] = {}    # Edge-specific properties
    created_at: datetime               # Creation time
    updated_at: datetime               # Last update time
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/graph.py`

---

### RetrievalResult

Result from memory retrieval.

```python
class RetrievalResult(BaseModel):
    """Result from memory retrieval."""
    
    query_id: str                      # Query identifier
    episodes: list[Episode] = []       # Retrieved episodes
    nodes: list[GraphNode] = []        # Retrieved graph nodes
    edges: list[GraphEdge] = []        # Retrieved graph edges
    scores: dict[str, float] = {}      # Relevance scores by ID
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/results.py`

---

### StepResult

Complete result of one cognitive step (for logging).

```python
class StepResult(BaseModel):
    """Result of a single cognitive step."""
    
    step_number: int                   # Step counter
    timestamp: datetime                # Step timestamp
    location_label: str                # Resolved location
    location_confidence: float         # Location confidence
    entity_count: int                  # Number of entities
    event_count: int                   # Number of events
    episode_count: int                 # Total episodes so far
    boundary_triggered: bool           # Whether episode froze
    boundary_reason: str | None = None # Freeze reason if any
    frame_id: int | None = None        # Source frame ID
    extras: dict[str, Any] = {}        # Forward compatibility

    def to_log_dict(self) -> dict:
        """Convert to dictionary for JSONL logging."""
        return self.model_dump(mode="json")
```

**Location**: `schemas/results.py`

---

## Validation

Pydantic handles validation automatically:

```python
# This will raise ValidationError
frame = SensorFrame(
    frame_id="not_an_int",  # Error: expected int
    timestamp=datetime.now(),
    raw_data={},
    sensor_type="test",
)

# Correct usage
frame = SensorFrame(
    frame_id=1,
    timestamp=datetime.now(),
    raw_data={"example": "data"},
    sensor_type="test",
)
```

## Serialization

All models serialize to JSON:

```python
# To JSON string
json_str = frame.model_dump_json()

# To dict
data = frame.model_dump()

# From JSON
frame = SensorFrame.model_validate_json(json_str)

# From dict
frame = SensorFrame.model_validate(data)
```

## Forward Compatibility

The `extras` field allows adding data without schema changes:

```python
# Adding custom data
frame = SensorFrame(
    frame_id=1,
    timestamp=datetime.now(),
    raw_data={},
    sensor_type="test",
    extras={
        "custom_field": "custom_value",
        "experiment_id": "exp_001",
    },
)

# Accessing extras
if "custom_field" in frame.extras:
    value = frame.extras["custom_field"]
```
