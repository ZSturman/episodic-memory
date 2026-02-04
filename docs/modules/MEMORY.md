# Memory Systems

Documentation for episode and graph memory storage.

## Overview

The agent uses two complementary memory systems:

| System | Purpose | Implementation |
|--------|---------|----------------|
| Episode Store | Sequential experience snapshots | JSONL file |
| Graph Store | Associative knowledge network | JSONL files |

---

## Episode Store

Episodes are frozen snapshots of the Active Context Frame (ACF) at boundary points.

### Interface

```python
class EpisodeStore(ABC):
    def store(self, episode: Episode) -> None
    def get(self, episode_id: str) -> Episode | None
    def list_all(self) -> list[Episode]
    def count(self) -> int
    def list_by_location(self, location: str) -> list[Episode]
    def list_recent(self, n: int) -> list[Episode]
```

### Implementations

#### InMemoryEpisodeStore

- In-memory only (lost on exit)
- Fast, no I/O
- For testing only

#### PersistentEpisodeStore

- JSONL file persistence
- Append-only writes
- Loads existing episodes on init

### File Format

`episodes.jsonl`:
```json
{"episode_id":"ep_001","location_label":"Kitchen","start_time":"2026-02-02T10:00:00","end_time":"2026-02-02T10:01:30","step_count":45,...}
{"episode_id":"ep_002","location_label":"Living Room","start_time":"2026-02-02T10:01:30","end_time":"2026-02-02T10:03:00","step_count":52,...}
```

### Usage

```python
from pathlib import Path
from episodic_agent.memory.episode_store import PersistentEpisodeStore
from episodic_agent.schemas import Episode

# Initialize
store = PersistentEpisodeStore(Path("runs/test/episodes.jsonl"))

# Store episode
episode = Episode(
    episode_id="ep_001",
    location_label="Kitchen",
    start_time=datetime.now(),
    end_time=datetime.now(),
    step_count=45,
    source_acf_id="acf_001",
)
store.store(episode)

# Retrieve
ep = store.get("ep_001")
recent = store.list_recent(5)
kitchen_episodes = store.list_by_location("Kitchen")
```

---

## Graph Store

Associative memory storing entities, locations, and relationships.

### Interface

```python
class GraphStore(ABC):
    # Node operations
    def add_node(self, node: GraphNode) -> None
    def get_node(self, node_id: str) -> GraphNode | None
    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]
    def update_node(self, node_id: str, **properties) -> None
    
    # Edge operations
    def add_edge(self, edge: GraphEdge) -> None
    def get_edge(self, edge_id: str) -> GraphEdge | None
    def get_edges_from(self, node_id: str) -> list[GraphEdge]
    def get_edges_to(self, node_id: str) -> list[GraphEdge]
    def get_edges_between(self, source: str, target: str) -> list[GraphEdge]
    def update_edge_weight(self, edge_id: str, weight: float) -> None
```

### Node Types

| Type | Purpose | Properties |
|------|---------|------------|
| `LOCATION` | Known places | guid, label, visit_count |
| `ENTITY` | Known objects | guid, label, category, typical_state |
| `EVENT` | Learned events | event_type, pattern_count |
| `CONCEPT` | Abstract concepts | (extensible) |

### Edge Types

| Type | Meaning | Example |
|------|---------|---------|
| `TYPICAL_IN` | Entity typically in location | Door → Kitchen |
| `RELATED_TO` | General relationship | - |
| `CAUSED_BY` | Causal relationship | LightOn ← SwitchToggle |
| `PART_OF` | Compositional | Handle → Door |
| `CO_OCCURS` | Temporal co-occurrence | - |

### Implementations

#### InMemoryGraphStore

- Dict-based storage
- No persistence
- For testing

#### LabeledGraphStore

- JSONL file persistence
- Separate files for nodes and edges
- Deduplication on load

### File Format

`nodes.jsonl`:
```json
{"node_id":"loc_kitchen","node_type":"location","label":"Kitchen","properties":{"guid":"room-kitchen-001"},...}
{"node_id":"ent_door_front","node_type":"entity","label":"Front Door","properties":{"guid":"door-001","category":"door"},...}
```

`edges.jsonl`:
```json
{"edge_id":"e_001","edge_type":"typical_in","source_id":"ent_door_front","target_id":"loc_kitchen","weight":5.0,...}
```

### Usage

```python
from pathlib import Path
from episodic_agent.memory.graph_store import LabeledGraphStore
from episodic_agent.schemas import GraphNode, GraphEdge, NodeType, EdgeType

# Initialize
store = LabeledGraphStore(
    nodes_path=Path("runs/test/nodes.jsonl"),
    edges_path=Path("runs/test/edges.jsonl"),
)

# Add location node
location = GraphNode(
    node_id="loc_kitchen",
    node_type=NodeType.LOCATION,
    label="Kitchen",
    properties={"guid": "room-kitchen-001"},
    created_at=datetime.now(),
    updated_at=datetime.now(),
)
store.add_node(location)

# Add entity node
entity = GraphNode(
    node_id="ent_door",
    node_type=NodeType.ENTITY,
    label="Front Door",
    properties={"guid": "door-001", "category": "door"},
    created_at=datetime.now(),
    updated_at=datetime.now(),
)
store.add_node(entity)

# Add edge (entity typically in location)
edge = GraphEdge(
    edge_id="e_door_kitchen",
    edge_type=EdgeType.TYPICAL_IN,
    source_id="ent_door",
    target_id="loc_kitchen",
    weight=1.0,
    created_at=datetime.now(),
    updated_at=datetime.now(),
)
store.add_edge(edge)

# Query
kitchen_entities = store.get_edges_to("loc_kitchen")
door_locations = store.get_edges_from("ent_door")
```

---

## Memory Lifecycle

### Episode Creation

```
ACF (mutable) ──boundary──> Episode (frozen) ──> EpisodeStore
```

1. ACF accumulates step data
2. Boundary detector triggers freeze
3. ACF snapshot becomes Episode
4. Episode stored to JSONL
5. ACF reset for new episode

### Graph Growth

```
Perception ──> New entity? ──> GraphNode
                    │
                    ├──> Known entity? ──> Update visit count
                    │
                    └──> typical_in edge ──> GraphEdge
```

1. New location → Create location node
2. New entity → Create entity node
3. Entity seen at location → Create/update `typical_in` edge
4. Edge weight = visit count

---

## Persistence Details

### Append-Only Writing

Both stores use append-only writes:

```python
def store(self, episode: Episode) -> None:
    line = episode.model_dump_json()
    self._file.write(line + "\n")
    self._file.flush()  # Immediate write
```

Benefits:
- Crash-safe (no data loss)
- Fast writes
- Simple recovery

### Load and Deduplicate

On initialization, stores load and deduplicate:

```python
def _load(self) -> None:
    seen_ids = set()
    for line in self._file:
        obj = json.loads(line)
        if obj["id"] not in seen_ids:
            self._cache[obj["id"]] = obj
            seen_ids.add(obj["id"])
```

---

## Querying Memory

### By Location

```python
# Episodes at location
episodes = episode_store.list_by_location("Kitchen")

# Entities typical at location
edges = graph_store.get_edges_to("loc_kitchen")
entities = [graph_store.get_node(e.source_id) for e in edges]
```

### By Time

```python
# Recent episodes
recent = episode_store.list_recent(10)

# Episodes in time range
all_eps = episode_store.list_all()
filtered = [e for e in all_eps if start <= e.start_time <= end]
```

### By Spreading Activation

```python
from episodic_agent.modules.retriever import SpreadingActivationRetriever

retriever = SpreadingActivationRetriever(
    graph_store=graph_store,
    episode_store=episode_store,
)

# Query from current context
result = retriever.retrieve(acf)
# result.nodes: Activated graph nodes
# result.episodes: Related episodes
# result.scores: Activation scores
```

---

## Performance Considerations

### Memory Usage

| Data | Approximate Size |
|------|------------------|
| Episode | ~500 bytes |
| Graph Node | ~200 bytes |
| Graph Edge | ~150 bytes |

For a typical run:
- 100 episodes ≈ 50 KB
- 50 nodes + 100 edges ≈ 25 KB

### File Growth

Files grow linearly with data:
- `episodes.jsonl`: ~500 bytes per episode
- `nodes.jsonl`: ~200 bytes per node
- `edges.jsonl`: ~150 bytes per edge

### Optimization Tips

1. **Use appropriate freeze interval** - More episodes = more storage
2. **Limit entity tracking** - Filter irrelevant entities
3. **Prune old data** - Remove old runs periodically
