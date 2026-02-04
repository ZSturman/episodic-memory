# Architecture Overview

The Episodic Memory Agent implements a cognitive architecture for building event-segmented episodic memories. This document describes the system design, data flow, and key components.

## Design Principles

1. **Strict Cognitive Order** - Each step follows an immutable sequence
2. **Dependency Injection** - All modules are swappable via interfaces
3. **Stable Data Contracts** - Pydantic models ensure type safety
4. **Forward Compatibility** - All schemas include `extras` dict for extensibility

## Cognitive Loop

The agent executes a fixed sequence of operations each step:

```
┌────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE STEP                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ SENSOR  │───>│ PERCEPTION  │───>│ ACF UPDATE  │                 │
│  └─────────┘    └─────────────┘    └──────┬──────┘                 │
│       │                                    │                        │
│       │              ┌─────────────────────┴────────────────┐      │
│       │              │                                      │      │
│       │              ▼                                      ▼      │
│       │    ┌─────────────────┐    ┌─────────────────────────┐     │
│       │    │    LOCATION     │    │       ENTITIES          │     │
│       │    │   "Where am I?" │    │     "What's here?"      │     │
│       │    └────────┬────────┘    └────────────┬────────────┘     │
│       │             │                          │                   │
│       │             └──────────┬───────────────┘                   │
│       │                        │                                   │
│       │                        ▼                                   │
│       │              ┌─────────────────┐                           │
│       │              │     EVENTS      │                           │
│       │              │ "What changed?" │                           │
│       │              └────────┬────────┘                           │
│       │                       │                                    │
│       │                       ▼                                    │
│       │    ┌─────────────────────────────────────────────────┐    │
│       │    │              RETRIEVAL & PREDICTION              │    │
│       │    │         (Query memory, generate predictions)     │    │
│       │    └─────────────────────┬───────────────────────────┘    │
│       │                          │                                 │
│       │                          ▼                                 │
│       │              ┌─────────────────┐                           │
│       │              │    BOUNDARY     │                           │
│       │              │    DETECTION    │                           │
│       │              └────────┬────────┘                           │
│       │                       │                                    │
│       │             ┌─────────┴─────────┐                          │
│       │             │                   │                          │
│       │     no      ▼                   ▼  yes                     │
│       │    ┌────────────┐      ┌────────────────┐                  │
│       │    │  Continue  │      │ FREEZE EPISODE │                  │
│       │    │   to next  │      │  (Store ACF)   │                  │
│       │    │    step    │      └────────────────┘                  │
│       │    └────────────┘                                          │
│       │                                                            │
└───────┴────────────────────────────────────────────────────────────┘
```

### Step Sequence (Immutable Order)

| Step | Operation | Input | Output |
|------|-----------|-------|--------|
| 1 | `sensor.get_frame()` | - | `SensorFrame` |
| 2 | `perception.process(frame)` | `SensorFrame` | `Percept` |
| 3 | `acf_builder.update_acf(acf, percept)` | `ACF`, `Percept` | `ACF` |
| 4 | `location_resolver.resolve(percept, acf)` | `Percept`, `ACF` | `(label, confidence)` |
| 5 | `entity_resolver.resolve(percept, acf)` | `Percept`, `ACF` | `list[ObjectCandidate]` |
| 6 | `event_resolver.resolve(percept, acf)` | `Percept`, `ACF` | `list[Event]` |
| 7 | `retriever.retrieve(acf)` | `ACF` | `RetrievalResult` |
| 8 | `boundary_detector.check(acf)` | `ACF` | `(bool, reason)` |
| 9 | `freeze_episode()` (if boundary) | `ACF` | `Episode` |

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Unity/Sensor                                                           │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────┐                                                        │
│   │ SensorFrame │  Raw sensor data with camera pose, room, entities      │
│   └──────┬──────┘                                                        │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                        │
│   │   Percept   │  Processed data with embeddings, object candidates     │
│   └──────┬──────┘                                                        │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────────────┐                                                │
│   │ ActiveContextFrame  │  Mutable working memory accumulating step data │
│   └──────────┬──────────┘                                                │
│              │                                                           │
│    ┌─────────┴─────────┐                                                 │
│    │                   │                                                 │
│    ▼                   ▼                                                 │
│ ┌─────────┐    ┌─────────────────────┐                                   │
│ │ Episode │    │ GraphNode/GraphEdge │                                   │
│ │ (frozen │    │ (persistent         │                                   │
│ │   ACF)  │    │  knowledge)         │                                   │
│ └────┬────┘    └──────────┬──────────┘                                   │
│      │                    │                                              │
│      ▼                    ▼                                              │
│ ┌──────────────────────────────────────┐                                 │
│ │           PERSISTENT STORAGE          │                                │
│ │  episodes.jsonl    nodes.jsonl        │                                │
│ │                    edges.jsonl        │                                │
│ └──────────────────────────────────────┘                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Memory Systems

### Working Memory (Active Context Frame)

The ACF is mutable working memory that accumulates information within an episode:

- Current location and confidence
- Visible entities
- Detected events
- Step count and timing
- Deltas (changes from previous step)

### Episodic Memory (Episodes)

When a boundary is detected, the ACF is "frozen" into an Episode:

- Immutable snapshot of the ACF
- Stored in `episodes.jsonl`
- Can be retrieved for reminiscing

### Semantic Memory (Graph Store)

Long-term knowledge stored as a graph:

- **Location Nodes** - Known places with labels
- **Entity Nodes** - Known objects with categories
- **Event Nodes** - Learned event patterns
- **Edges** - Relationships (e.g., `typical_in` links entities to locations)

## Boundary Detection

Episodes are segmented at natural boundaries:

| Boundary Type | Trigger | Example |
|---------------|---------|---------|
| Location Change | Enter new room | Walking from kitchen to bedroom |
| Temporal | Time interval elapsed | Every N steps |
| Event-based | Salient event occurs | Light turned on |
| Prediction Error | Unexpected observation | Object missing |

The `HysteresisBoundaryDetector` uses hysteresis thresholds to prevent rapid oscillation.

## Spreading Activation Retrieval

Memory retrieval uses spreading activation:

1. Cue nodes receive initial activation
2. Activation spreads along graph edges
3. Decay applied at each hop
4. Nodes exceeding threshold are retrieved

```
Activation(n) = Σ(incoming_activation × edge_weight × decay)
```

## Predictions

The prediction module generates expectations:

- Expected entities at current location (based on `typical_in` edges)
- Expected state for known entities
- Prediction error computed when expectations violated

High prediction error can trigger episode boundaries.

## Module Profiles

Profiles configure module combinations:

### `stub` Profile
- All mock/synthetic modules
- No external dependencies
- For testing and development

### `unity_cheat` Profile
- Unity WebSocket sensor
- GUID-based "cheat" perception (perfect recognition)
- For integration testing

### `unity_full` Profile
- Full feature set
- Spreading activation retrieval
- Hysteresis boundary detection
- Prediction module enabled

## See Also

- [Module Interfaces](INTERFACES.md) - Detailed interface documentation
- [Data Contracts](DATA_CONTRACTS.md) - Schema definitions
- [Unity Setup](../unity/SETUP.md) - Unity integration
