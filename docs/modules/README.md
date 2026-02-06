# Module Implementations

This document provides an overview of all module implementations in the Episodic Memory Agent.

## Module Categories

| Category | Directory | Purpose |
|----------|-----------|---------|
| Core | `core/` | Interfaces and orchestration |
| Schemas | `schemas/` | Data models |
| Stubs | `modules/stubs/` | Testing implementations |
| Unity | `modules/unity/` | Unity integration |
| Memory | `memory/` | Persistent storage |
| Scenarios | `scenarios/` | Test automation |
| Metrics | `metrics/` | Logging and evaluation |

---

## Stub Modules (`modules/stubs/`)

Minimal implementations for testing without external dependencies.

### StubSensorProvider

Generates synthetic sensor frames.

```python
# Configuration
- Generates frames with incrementing IDs
- Random location changes
- No real sensor data
```

### StubPerception

Returns empty percepts.

```python
# Always returns
- Empty scene embedding
- No object candidates
```

### StubLocationResolver

Returns fixed "unknown" location.

### StubEntityResolver

Returns empty entity list.

### StubEventResolver

Returns empty event list.

### StubBoundaryDetector

Time-based boundary only.

```python
# Triggers boundary after N steps
- Default interval: 50 steps
- No event-based boundaries
```

### StubRetriever

Returns empty retrieval results.

### StubDialogManager

Auto-generates labels without user input.

---

## Unity Modules (`modules/unity/`)

Unity integration for real sensor data.

### UnityWebSocketSensorProvider

Receives frames via WebSocket.

```python
# Features
- Auto-reconnect on disconnect
- Frame buffering
- Protocol validation
- Configurable URL
```

### PerceptionUnityCheat

Converts Unity GUIDs to embeddings.

```python
# Algorithm
1. Hash GUID with SHA256
2. Convert to float vector
3. Normalize to unit length
4. Same GUID = same embedding (deterministic)
```

### LocationResolverCheat

Uses Unity room GUID for location.

```python
# Flow
1. Extract room GUID from frame
2. Check graph store for known location
3. If unknown, prompt for label
4. Store location node in graph
```

### EntityResolverCheat

Uses Unity entity GUIDs.

```python
# Features
- Tracks entities by GUID
- Creates typical_in edges to locations
- Updates visit counts
```

### CommandClient

Sends commands to Unity.

```python
# Supported commands
- teleport_player
- toggle_interactable
- spawn_ball / move_ball / despawn_ball
- reset_world
- get_world_state
- create_room_volume / update_room_volume / remove_room_volume
- set_entity_label
- clear_dynamic_volumes
```

---

## Real Resolvers (sensor-agnostic)

### LocationResolverReal (`modules/spatial_resolver.py`)

Fingerprint-based location resolution without Unity GUIDs.

```python
# Algorithm
1. Build scene fingerprint from percept embedding
2. Cosine distance to detect location transitions
3. Hysteresis (N consecutive frames) prevents flickering
4. Match against known fingerprints via centroid similarity
5. If no match, prompt user for label via DialogManager
6. Running average centroids adapt over time
```

### EntityResolverReal (`modules/entity_resolver_real.py`)

Embedding-similarity entity resolution without Unity GUIDs.

```python
# Algorithm
1. Match percept candidates by embedding cosine similarity
2. Threshold-based: above match_threshold → known entity
3. Below threshold → create new entity node in graph
4. Links entities to locations via typical_in edges
5. Optionally prompts user for labels on new entities
```

---

## Boundary Detection (`modules/boundary.py`)

### HysteresisBoundaryDetector

Multi-factor boundary detection with hysteresis.

```python
# Factors
1. Location change (immediate)
2. Time interval (configurable)
3. Prediction error (threshold)
4. Salient event (configurable)

# Hysteresis
- High threshold: 0.8 (trigger)
- Low threshold: 0.3 (reset)
- Prevents oscillation
```

---

## Event Resolution (`modules/event_resolver.py`)

### EventResolverStateChange

Detects state-change events.

```python
# Detection
1. Compare current state to previous
2. Detect: Open/Closed, On/Off changes
3. Learn event patterns in graph
4. Optionally prompt for labels

# Metrics tracked
- events_detected
- events_labeled
- events_recognized
- questions_asked
```

---

## Delta Detection (`modules/delta_detector.py`)

### DeltaDetector

Tracks changes between steps.

```python
# Delta types
- NEW: Entity appeared
- MISSING: Entity disappeared
- MOVED: Position changed (threshold)
- STATE_CHANGED: State changed

# Configuration
- move_threshold: 0.5 meters
- missing_window: 2 steps
```

---

## Retrieval (`modules/retriever.py`)

### SpreadingActivationRetriever

Graph-based memory retrieval.

```python
# Algorithm
1. Initialize activation from cues (current location, entities)
2. Spread activation along edges
3. Apply decay at each hop
4. Collect nodes above threshold

# Parameters
- initial_activation: 1.0
- decay: 0.9
- threshold: 0.1
- max_hops: 3
```

---

## Prediction (`modules/prediction.py`)

### PredictionModule

Generates expectations and computes errors.

```python
# Predictions generated
- Expected entities at location (from typical_in edges)
- Expected entity states (from history)

# Prediction error
- Missing expected entities
- Unexpected state changes
- Novel entities (low weight)
```

---

## Memory Stores

See [Memory Documentation](MEMORY.md) for details.

### InMemoryEpisodeStore / InMemoryGraphStore

In-memory only, for testing.

### PersistentEpisodeStore

JSONL-based persistence.

```python
# File: episodes.jsonl
- One JSON object per line
- Append-only writes
- Loads existing on init
```

### LabeledGraphStore

JSONL-based graph persistence.

```python
# Files
- nodes.jsonl: Graph nodes
- edges.jsonl: Graph edges
- Append-only with deduplication on load
```

---

## Custom Profiles

To create a custom profile, edit `utils/profiles.py`:

```python
MY_PROFILE = ProfileConfig(
    name="my_profile",
    description="Custom profile description",
    
    # Module class names
    sensor_provider="UnityWebSocketSensorProvider",
    perception="PerceptionUnityCheat",
    acf_builder="StubACFBuilder",
    location_resolver="LocationResolverCheat",
    entity_resolver="EntityResolverCheat",
    event_resolver="EventResolverStateChange",
    retriever="SpreadingActivationRetriever",
    boundary_detector="HysteresisBoundaryDetector",
    dialog_manager="CLIDialogManager",
    episode_store="PersistentEpisodeStore",
    graph_store="LabeledGraphStore",
    
    # Profile-specific parameters
    parameters={
        "spreading_decay": 0.9,
        "boundary_high_threshold": 0.8,
        "boundary_low_threshold": 0.3,
        "move_threshold": 0.5,
    },
)

# Register profile
PROFILES["my_profile"] = MY_PROFILE
```

Then use:

```bash
python -m episodic_agent run --profile my_profile
```

---

## Adding New Modules

1. **Implement interface** from `core/interfaces.py`
2. **Add to module directory** (`modules/` or `memory/`)
3. **Register in profiles** if needed
4. **Write tests** in `tests/`

Example:

```python
# modules/my_resolver.py
from episodic_agent.core.interfaces import LocationResolver
from episodic_agent.schemas import Percept, ActiveContextFrame

class MyLocationResolver(LocationResolver):
    def __init__(self, custom_param: float = 0.5):
        self._param = custom_param
    
    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> tuple[str, float]:
        # Custom logic here
        return "my_location", 0.95
```

Register in `__init__.py` and profile to use.
