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
| **1** | Python core skeleton, contracts, runnable loop | ✅ Current |
| 2 | Real memory, labels/conflicts, episode freezing | Planned |
| 3 | Unity sensor simulator (rooms, objects, WebSocket) | Planned |
| 4 | Unity integration, cheat perception, GUID mapping | Planned |
| 5 | Change detection, state-change events | Planned |
| 6 | Spreading activation, predictions, test harness | Planned |

## Project Structure

```
src/episodic_agent/
├── cli.py              # CLI entry point
├── core/
│   ├── interfaces.py   # Abstract base classes
│   └── orchestrator.py # Agent step loop
├── schemas/
│   ├── frames.py       # SensorFrame, Percept, ObjectCandidate
│   ├── context.py      # ActiveContextFrame, Episode
│   ├── graph.py        # GraphNode, GraphEdge
│   └── results.py      # RetrievalResult, StepResult
├── modules/stubs/      # Stub implementations
├── memory/stubs/       # In-memory stores
├── metrics/            # Logging utilities
└── utils/              # Config constants
```

## License

MIT
