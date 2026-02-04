# Episodic Memory Agent

A portable **cognitive agent framework** implementing an event-segmented episodic memory system. The agent processes sensor input through a strict cognitive pipeline, building memories that are segmented into episodes at natural boundaries.

## Overview

This system simulates how an intelligent agent might form episodic memories while navigating and interacting with an environment. It can run:

1. **Standalone (CLI)** - Using synthetic/stub sensors for testing and development
2. **Connected to Unity** - Receiving real-time sensor data from a 3D simulation

### Core Cognitive Loop

Each step, the agent answers three fundamental questions:

```
┌─────────────────────────────────────────────────────────────────┐
│                     COGNITIVE STEP                               │
├─────────────────────────────────────────────────────────────────┤
│  1. WHERE AM I?      → Location Resolution                       │
│  2. WHAT'S HERE?     → Entity Recognition                        │
│  3. WHAT CHANGED?    → Event Detection                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Run Modes

#### Standalone Mode (No Unity Required)

```bash
# Run with synthetic data for testing
python -m episodic_agent run --profile stub --steps 200 --fps 10
```

#### Unity Connected Mode

```bash
# First, start Unity (see docs/unity/SETUP.md)
# Then connect the agent:
python -m episodic_agent run --profile unity_full --unity-ws ws://localhost:8765 --fps 10

# Or run automated scenarios:
python -m episodic_agent scenario mixed --profile unity_full
```

#### View Reports

```bash
# Generate a report from a completed run
python -m episodic_agent report runs/<timestamp>

# Generate HTML report with visualizations
python -m episodic_agent report runs/<timestamp> --html
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `run` | Run the agent loop continuously |
| `scenario <name>` | Run a predefined test scenario |
| `report <folder>` | Generate report from run data |
| `profiles` | List available module profiles |

## Architecture

```
                           ┌─────────────────────┐
                           │   Sensor Provider   │
                           │ (Unity/Stub/Replay) │
                           └──────────┬──────────┘
                                      │ SensorFrame
                                      ▼
                           ┌─────────────────────┐
                           │  Perception Module  │
                           │ (Embeddings/Objects)│
                           └──────────┬──────────┘
                                      │ Percept
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Location Resolver│       │ Entity Resolver  │       │ Event Resolver   │
│ "Where am I?"    │       │ "What's here?"   │       │ "What changed?"  │
└────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ Active Context Frame    │
                       │ (Working Memory)        │
                       └───────────┬─────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Boundary Detector│    │    Retriever     │    │   Prediction     │
│ (Episode Segment)│    │ (Memory Query)   │    │  (What to Expect)│
└────────┬─────────┘    └──────────────────┘    └──────────────────┘
         │
         ▼ (on boundary)
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY STORES                             │
├────────────────────────────┬────────────────────────────────────┤
│      Episode Store         │         Graph Store                 │
│   (Frozen ACF Snapshots)   │  (Locations, Entities, Relations)  │
└────────────────────────────┴────────────────────────────────────┘
```

### Module Interfaces

All modules are swappable via dependency injection. See [docs/architecture/INTERFACES.md](docs/architecture/INTERFACES.md) for details.

| Interface | Purpose |
|-----------|---------|
| `SensorProvider` | Raw sensor data source (Unity, file replay, synthetic) |
| `PerceptionModule` | Converts frames to percepts with embeddings |
| `LocationResolver` | Determines current location |
| `EntityResolver` | Identifies entities in the scene |
| `EventResolver` | Detects state changes and events |
| `BoundaryDetector` | Determines episode boundaries |
| `Retriever` | Queries episodic/graph memory |
| `EpisodeStore` | Persists frozen episodes |
| `GraphStore` | Maintains associative graph memory |

### Data Flow

All data flows through Pydantic v2 models defined in `src/episodic_agent/schemas/`:

```
SensorFrame → Percept → ActiveContextFrame → Episode
                              ↓
                    GraphNode / GraphEdge
```

See [docs/architecture/DATA_CONTRACTS.md](docs/architecture/DATA_CONTRACTS.md) for schema documentation.

## Project Structure

```
episodic-memory-agent/
├── src/episodic_agent/
│   ├── core/                 # Core abstractions and orchestrator
│   │   ├── interfaces.py     # Abstract base classes
│   │   └── orchestrator.py   # Cognitive loop implementation
│   ├── schemas/              # Pydantic data models
│   │   ├── frames.py         # SensorFrame, Percept
│   │   ├── context.py        # ActiveContextFrame, Episode
│   │   ├── graph.py          # GraphNode, GraphEdge
│   │   └── events.py         # Delta, EventCandidate
│   ├── modules/              # Module implementations
│   │   ├── stubs/            # Stub implementations for testing
│   │   ├── unity/            # Unity integration modules
│   │   ├── boundary.py       # Episode boundary detection
│   │   ├── retriever.py      # Memory retrieval (spreading activation)
│   │   └── prediction.py     # Prediction and prediction error
│   ├── memory/               # Persistent storage
│   │   ├── episode_store.py  # Episode persistence
│   │   └── graph_store.py    # Graph memory persistence
│   ├── scenarios/            # Test scenario framework
│   ├── metrics/              # Logging and evaluation
│   ├── utils/                # Configuration and helpers
│   └── cli.py                # Command-line interface
├── tests/                    # Test suite
├── docs/                     # Documentation
│   ├── architecture/         # System architecture docs
│   └── unity/                # Unity setup guides
├── runs/                     # Run output (logs, episodes, reports)
└── UnitySensorSim/           # Unity sensor simulator project
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture/README.md) | System design and data flow |
| [Module Interfaces](docs/architecture/INTERFACES.md) | Abstract interfaces for all modules |
| [Data Contracts](docs/architecture/DATA_CONTRACTS.md) | Schema definitions |
| [Unity Setup](docs/unity/SETUP.md) | Complete Unity configuration guide |
| [Scenarios](docs/scenarios/README.md) | Test scenario documentation |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

## Profiles

Profiles configure which module implementations to use:

| Profile | Description | Use Case |
|---------|-------------|----------|
| `stub` | All synthetic/mock modules | Development, unit tests |
| `unity_cheat` | Unity with GUID-based perception | Integration testing |
| `unity_full` | Full features with spreading activation | Production simulation |

```bash
# List all available profiles
python -m episodic_agent profiles
```

## Output Files

Each run creates a timestamped folder in `runs/` containing:

| File | Format | Description |
|------|--------|-------------|
| `run.jsonl` | JSON Lines | Step-by-step execution log |
| `episodes.jsonl` | JSON Lines | Frozen episode records |
| `nodes.jsonl` | JSON Lines | Graph memory nodes |
| `edges.jsonl` | JSON Lines | Graph memory edges |
| `metrics.json` | JSON | Computed run metrics |
| `report.txt` | Text | Human-readable summary |
| `report.html` | HTML | Visual report with charts |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=episodic_agent tests/

# Run specific test file
pytest tests/test_schemas.py -v
```

### Code Quality

```bash
# Lint with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/
```

## License

MIT License - See LICENSE file for details.
