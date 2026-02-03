# Episodic Memory Agent

A portable "agent brain" implementing an event-segmented episodic memory system. The agent follows a strict cognitive flow each step:

1. **Where am I?** → Location resolution
2. **What's here?** → Entity recognition  
3. **What changed/happened?** → Event detection

## Quick Start (Phase 1)

### Installation

```bash
# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Run the Agent Loop

```bash
# Run 200 steps at 10 FPS with default settings
python -m episodic_agent.cli run --steps 200 --fps 10

# Run with custom seed and freeze interval
python -m episodic_agent.cli run --steps 200 --fps 10 --seed 42 --freeze-interval 50
```

### Output

- **Console**: One-line summary per step showing location, entity/event counts, episode count
- **Logs**: JSONL file written to `runs/<timestamp>/run.jsonl`

## Architecture

### Module Interfaces

All modules are swappable via constructor injection. Implement the abstract base classes in `episodic_agent.core.interfaces`:

| Interface | Purpose |
|-----------|---------|
| `SensorProvider` | Provides raw sensor frames (Unity, file replay, synthetic) |
| `PerceptionModule` | Converts frames to percepts with embeddings |
| `ACFBuilder` | Maintains the Active Context Frame |
| `LocationResolver` | Resolves current location from percepts |
| `EntityResolver` | Identifies entities in the scene |
| `EventResolver` | Detects state changes and events |
| `BoundaryDetector` | Determines episode boundaries |
| `Retriever` | Queries episodic/graph memory |
| `DialogManager` | Handles user interactions (labels, conflicts) |
| `EpisodeStore` | Persists frozen episodes |
| `GraphStore` | Maintains associative graph memory |

### Data Contracts

All data flows through stable Pydantic v2 models in `episodic_agent.schemas`:

- `SensorFrame` - Raw sensor input
- `Percept` - Processed perception with embeddings
- `ObjectCandidate` - Recognized object with confidence
- `ActiveContextFrame` - Mutable working memory
- `Episode` - Frozen ACF snapshot
- `GraphNode` / `GraphEdge` - Associative memory structure
- `RetrievalResult` - Query results from memory
- `StepResult` - Single step output for logging

All top-level schemas include an `extras: dict` field for forward compatibility.

## Phase Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Python core skeleton, contracts, runnable loop | ✅ Done |
| **2** | Real memory, labels/conflicts, episode freezing | ✅ Done |
| 3 | Unity sensor simulator (rooms, objects, WebSocket) | ✅ Done |
| **4** | Unity integration, cheat perception, GUID mapping | ✅ Current |
| 5 | Change detection, state-change events | Planned |
| 6 | Spreading activation, predictions, test harness | Planned |

## Phase 4: Unity Integration

Phase 4 connects the Python agent to Unity's sensor simulator with "cheat" perception that uses ground-truth GUIDs for perfect location and entity resolution.

### Features

- **WebSocket Sensor Provider**: Connects to Unity with auto-reconnect, frame validation, and buffering
- **Cheat Perception**: Converts Unity GUIDs to deterministic embeddings (same GUID = same embedding)
- **Location Learning**: Learns rooms via GUID, prompts for labels, persists to graph
- **Entity Learning**: Tracks entities by GUID, links to locations with `typical_in` edges
- **Profile System**: Easily switch between stub/unity modes via `--profile`

### Quick Start (Phase 4)

```bash
# Install with websockets
pip install -e ".[dev]"

# List available profiles
python -m episodic_agent.cli profiles

# Run with Unity (infinite loop, Ctrl+C to stop)
python -m episodic_agent.cli run \
    --profile unity_cheat \
    --unity-ws ws://localhost:8765 \
    --fps 10 \
    --steps 0

# Run with auto-labeling (no prompts)
python -m episodic_agent.cli run \
    --profile unity_cheat \
    --auto-label \
    --fps 10 \
    --steps 0
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | `stub` | Module profile (`stub`, `unity_cheat`) |
| `--unity-ws` | `ws://localhost:8765` | Unity WebSocket URL |
| `--fps` | `10` | Target frames per second |
| `--steps` | `100` | Number of steps (0 = infinite) |
| `--auto-label` | `false` | Auto-generate labels without prompting |
| `--verbose` | `false` | Enable debug logging |

### Console Output (Unity Mode)

```
[0001] 🟢 #42 📍 Living Room(95%) 👁 [door:1 furniture:2 item:1] 📚 0
[0002] 🟢 #43 📍 Living Room(95%) 👁 [door:1 furniture:2 item:1] 📚 0
...
[0051] 🟢 #91 📍 Kitchen(95%) 👁 [appliance:3 item:2] 📚 1 📦
```

- 🟢/🟡/🔴 = Connection status
- `#42` = Unity frame ID
- `📍` = Current location and confidence
- `👁` = Visible entities by category
- `📚` = Episode count
- `📦` = Episode frozen this step

### End-to-End Validation (Smoke Test)

1. **Start Unity Simulator**
   ```
   # In Unity Editor, enter Play mode
   # WebSocket server starts on ws://localhost:8765
   ```

2. **Start Python Agent**
   ```bash
   python -m episodic_agent.cli run \
       --profile unity_cheat \
       --unity-ws ws://localhost:8765 \
       --fps 10 \
       --steps 0
   ```

3. **Walk Between Rooms in Unity**
   - Enter a room → Agent prompts: "🆕 New location detected!"
   - Enter label (e.g., "Living Room") → Agent confirms: "✅ Learned location: Living Room"
   - Continue to next room → Repeat labeling

4. **Revisit Rooms**
   - Enter previously labeled room → Agent auto-resolves: "📍 Entered: Living Room"
   - No prompt needed (location learned)

5. **Verify Persistence**
   - Ctrl+C to stop agent
   - Check `runs/<timestamp>/`:
     - `run.jsonl` - Step-by-step logs
     - `episodes.jsonl` - Frozen episodes
     - `nodes.jsonl` - Graph nodes (locations, entities)
     - `edges.jsonl` - Graph edges (typical_in links)

6. **Confirm Memory Works**
   - Restart agent with same run directory
   - Previously learned locations resolve automatically

### Architecture (Phase 4)

```
Unity Simulator                    Python Agent
┌─────────────────┐               ┌────────────────────┐
│  WebSocket      │──────────────>│ UnityWebSocket     │
│  Server         │  JSON frames  │ SensorProvider     │
│  (8765)         │               └─────────┬──────────┘
└─────────────────┘                         │
                                            v
                              ┌─────────────────────────┐
                              │ PerceptionUnityCheat    │
                              │ (GUID → embedding)      │
                              └─────────────┬───────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    v                       v                       v
         ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
         │ LocationResolver │    │ EntityResolver   │    │ BoundaryDetector │
         │ Cheat            │    │ Cheat            │    │                  │
         └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
                  │                       │                       │
                  v                       v                       v
         ┌────────────────────────────────────────────────────────────────┐
         │                     Graph Store (JSONL)                        │
         │  - Location nodes (room GUID → label)                          │
         │  - Entity nodes (entity GUID → label, category)                │
         │  - typical_in edges (entity → location, weighted by visits)    │
         └────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/episodic_agent/
├── cli.py              # CLI entry point with profiles
├── core/
│   ├── interfaces.py   # Abstract base classes
│   └── orchestrator.py # Agent step loop
├── schemas/
│   ├── frames.py       # SensorFrame, Percept, ObjectCandidate
│   ├── context.py      # ActiveContextFrame, Episode
│   ├── graph.py        # GraphNode, GraphEdge
│   └── results.py      # RetrievalResult, StepResult
├── modules/
│   ├── stubs/          # Stub implementations (Phase 1)
│   ├── dialog.py       # CLI dialog manager
│   ├── label_manager.py # Label conflict resolution
│   └── unity/          # Unity integration (Phase 4)
│       ├── sensor_provider.py  # WebSocket sensor
│       ├── perception.py       # Cheat perception
│       └── resolvers.py        # Location/Entity resolvers
├── memory/
│   ├── episode_store.py  # Persistent episode storage
│   └── graph_store.py    # Labeled graph storage
├── metrics/            # Logging utilities
└── utils/
    ├── config.py       # Configuration constants
    ├── confidence.py   # Confidence calculations
    └── profiles.py     # Profile configuration system
