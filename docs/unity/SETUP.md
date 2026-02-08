# Unity Setup & Runtime Guide

> **Design Invariant:** Unity is a stateless sensor interface. All cognition, memory, labeling, and identity reasoning happen in the Python backend. Unity streams raw pose and visual features; the backend decides what they mean.

> **See Also:** [PROTOCOL.md](PROTOCOL.md) for message schemas and field-level documentation.

---

## 1. Architecture Overview

Unity and the Python backend communicate over two WebSocket channels that form a single integrated sensor pipeline:

```
Unity Simulator                          Python Backend
+----------------------------+            +----------------------------------+
|                            |            |                                  |
|  Sensor Camera             |            |  UnityWebSocketSensorProvider    |
|  +----------------------+  |  port 8765 |  +----------------------------+  |
|  | Pose (10 Hz)         |--+----------->|  | SensorFrame                |  |
|  | Room GUID            |  |  bidir.    |  | (pose + room context)      |  |
|  +----------------------+  |<-----------+--| commands, labels, updates  |  |
|                            |            |  +----------+-----------------+  |
|  VisualFeatureExtractor    |            |             |                    |
|  +----------------------+  |  port 8767 |  VisualStreamClient              |
|  | 4x4 Grid (5 Hz)      |--+----------->|  +----------+-----------------+  |
|  | Feature summaries     |  |<-----------+--| VisualSummaryFrame         |  |
|  +----------------------+  |  focus     |  | (scene features)           |  |
|                            |  requests  |  +----------+-----------------+  |
|  StatusBarUI               |            |             |                    |
|  EntityLabelUI             |            |  +----------v-----------------+  |
|  LabelRequestModal         |            |  | Percept (unified)          |  |
|                            |            |  |  motion features (pose d)  |  |
|  DynamicWorldBuilder       |            |  |  visual features (grid)    |  |
|  (visualization only)      |            |  +----------+-----------------+  |
|                            |            |             |                    |
|                            |            |  AgentOrchestrator.step()        |
|                            |            |  +----------v-----------------+  |
|                            |            |  | Location -> Entity -> Event|  |
|                            |            |  | -> Retrieval -> Boundary   |  |
|                            |            |  | -> Memory Write            |  |
|                            |            |  +----------------------------+  |
+----------------------------+            +----------------------------------+
```

### Channels

| Port | Direction | Content | Rate |
|------|-----------|---------|------|
| **8765** | Bidirectional | Sensor frames (Unity->Python), commands + label requests + entity/location updates (Python->Unity), handshake/heartbeat (both) | ~10 Hz frames |
| **8767** | Bidirectional | Visual summary grids (Unity->Python), focus requests (Python->Unity), focus responses (Unity->Python) | ~5 Hz summaries |

Both channels feed a single `Percept` object in the backend. They are not independent subsystems -- they synchronize in `AgentOrchestrator.step()` to produce the unified input for all downstream cognition.

---

## 2. Prerequisites

- **Unity Hub** -- [unity.com/download](https://unity.com/download)
- **Unity 2022.3 LTS** or later
- **Python 3.11+** with the episodic-agent package installed

---

## 3. Setup Pipeline

This is a single sequential process. Complete each step before moving to the next.

### 3.1 Install Unity

1. Download and install [Unity Hub](https://unity.com/download)
2. In Unity Hub -> **Installs** -> **Install Editor** -> select **Unity 2022.3 LTS**
3. Keep default modules. Click **Install**.

### 3.2 Open the Project

1. In Unity Hub -> **Projects** -> **Add** -> **Add project from disk**
2. Navigate to `episodic-memory-agent/UnitySensorSim/` and add it
3. Click **UnitySensorSim** to open (first launch imports assets -- may take several minutes)

### 3.3 Create the Scene

1. **File** -> **New Scene** -> **Basic (Built-in)** -> **Create**
2. **File** -> **Save As** -> `Assets/Scenes/DemoScene.unity`

### 3.4 GameManager

Create the central coordinator that runs all sensor streaming and protocol handling.

1. Hierarchy -> right-click -> **Create Empty** -> rename to `GameManager`
2. With GameManager selected, add these components via **Add Component** in the Inspector:

| Component | Purpose |
|-----------|---------|
| `WebSocketServer` | Hosts the port 8765 bidirectional channel |
| `SensorStreamer` | Captures and transmits sensor frames at target FPS |
| `CommandReceiver` | Handles incoming commands from the Python backend |
| `WorldManager` | Manages room volumes, entities, and world state |
| `ProtocolLogger` | Logs all protocol traffic (console + file) |

3. In the **WebSocket Server** component: set **Port** to `8765`, enable **Auto Start**.

### 3.5 Player and Camera

The Player object carries a single Sensor Camera. This camera is the agent's only perceptual input -- its stream is what reaches the backend.

1. Hierarchy -> **3D Object** -> **Capsule** -> rename to `Player`
2. Set Transform position to `(0, 1, 0)`
3. Add components: `Character Controller`, `FirstPersonController`, `PlayerInteraction`
4. Set Player's **Tag** to `Player`
5. Delete the existing Main Camera
6. Right-click on **Player** -> **Camera** -> rename to `Main Camera`
7. Set camera local position to `(0, 0.5, 0)`, tag to `MainCamera`

#### Camera Invariants

- **Sensor Camera** (the Main Camera on the Player): this is the only camera whose output reaches the Python backend. It drives both the port 8765 pose stream and the port 8767 visual summary grid.
- **First-Person view** (F1): debug-only display of the player's perspective. Does not stream to the backend.
- **Third-Person view** (F3): debug-only external view. Does not stream to the backend.

The Sensor Camera is always active for streaming regardless of which debug view is displayed on screen. Resolution, frame rate, and crop region are controlled by the backend via `stream_control` messages.

```
Player
+-- Main Camera (Sensor Camera)
    |-- Streams pose to port 8765 (~10 Hz)
    |-- Feeds VisualFeatureExtractor -> port 8767 (~5 Hz)
    +-- Backend-controllable: resolution, FPS, crop
```

#### Visual Summary Pipeline

The Sensor Camera also drives the 4x4 visual summary channel:

1. `VisualFeatureExtractor` captures the Sensor Camera's RenderTexture
2. Divides the field of view into a 4x4 grid (16 cells)
3. For each cell: computes color histogram (24 bins), edge directions (8 bins), mean brightness, edge density, motion magnitude
4. Sends the `VisualSummaryFrame` to port 8767 at ~5 Hz
5. The backend's `VisualStreamClient` receives these and extracts scene signatures
6. The backend can send `focus_request` messages back for on-demand high-res crops (region, full frame, or entity tracking)

Raw images are never stored long-term. A fixed-size ring buffer (~50 MB, ~100 frames) holds recent high-res crops; raw pixels are discarded after feature extraction.

### 3.6 Environment

#### Room Volumes

Room volumes define spatial regions the agent can occupy. They provide `current_room_guid` in sensor frames -- a stable spatial anchor the backend uses as a ground-truth signal for location fingerprinting.

**Room 1 -- Living Room:**
1. Hierarchy -> **Create Empty** -> rename to `Room_Living`
2. Add **Box Collider**: enable **Is Trigger**, set Size `(10, 4, 10)`, Center `(0, 2, 0)`
3. Add `RoomVolume` -- leave **Label** empty (backend learns labels from user interaction)

**Room 2 -- Kitchen:**
1. Hierarchy -> **Create Empty** -> rename to `Room_Kitchen`, position at `(15, 0, 0)`
2. Add **Box Collider**: enable **Is Trigger**, set Size `(8, 4, 8)`, Center `(0, 2, 0)`
3. Add `RoomVolume` -- leave **Label** empty

#### Walls, Floors, and Visual Structure

Build the physical environment to make rooms navigable. All spatial objects are pure geometry -- they carry no semantic meaning for the backend.

<details>
<summary><strong>Detailed wall/floor construction (click to expand)</strong></summary>

##### Floor
1. Hierarchy -> **3D Object** -> **Plane** -> rename to `Floor`
2. Position `(7, 0, 0)`, Scale `(3, 1, 2)` -- covers both rooms

##### Living Room Walls (10x10 area at origin)

| Object | Position | Scale |
|--------|----------|-------|
| `Wall_Living_North` (Cube) | `(0, 2, 5)` | `(10, 4, 0.2)` |
| `Wall_Living_South` (Cube) | `(0, 2, -5)` | `(10, 4, 0.2)` |
| `Wall_Living_West` (Cube) | `(-5, 2, 0)` | `(0.2, 4, 10)` |
| `Wall_Living_East_Top` (Cube) | `(5, 3, 0)` | `(0.2, 2, 10)` |
| `Wall_Living_East_Left` (Cube) | `(5, 1, -3)` | `(0.2, 2, 4)` |
| `Wall_Living_East_Right` (Cube) | `(5, 1, 3)` | `(0.2, 2, 4)` |

The east wall leaves a doorway at `(5, 0-2, -1 to 1)`.

##### Kitchen Walls (8x8 area at position 15, 0, 0)

| Object | Position | Scale |
|--------|----------|-------|
| `Wall_Kitchen_North` (Cube) | `(15, 2, 4)` | `(8, 4, 0.2)` |
| `Wall_Kitchen_South` (Cube) | `(15, 2, -4)` | `(8, 4, 0.2)` |
| `Wall_Kitchen_East` (Cube) | `(19, 2, 0)` | `(0.2, 4, 8)` |
| `Wall_Kitchen_West_Top` (Cube) | `(11, 3, 0)` | `(0.2, 2, 8)` |
| `Wall_Kitchen_West_Left` (Cube) | `(11, 1, -2.5)` | `(0.2, 2, 3)` |
| `Wall_Kitchen_West_Right` (Cube) | `(11, 1, 2.5)` | `(0.2, 2, 3)` |

##### Hallway

| Object | Position | Scale |
|--------|----------|-------|
| `Wall_Hallway_North` (Cube) | `(8, 2, 1.5)` | `(6, 4, 0.2)` |
| `Wall_Hallway_South` (Cube) | `(8, 2, -1.5)` | `(6, 4, 0.2)` |

##### Hierarchy Organization

```
Environment
+-- Floors
|   +-- Floor
|   +-- Floor_Hallway
+-- Walls_LivingRoom
|   +-- (all living room walls)
+-- Walls_Kitchen
|   +-- (all kitchen walls)
+-- Walls_Hallway
    +-- (hallway walls)
```

##### Materials for Visual Distinction

Create separate materials for each room to make navigation intuitive:
- `Mat_Wall_Living` -- warm color (beige/cream)
- `Mat_Wall_Kitchen` -- cool color (light blue/white)
- `Mat_Wall_Hallway` -- neutral (gray)

</details>

#### Entities

Objects the agent can detect and track. Each needs an `EntityMarker` component -- this provides a stable GUID for tracking. All semantic labels are discovered by the backend through user interaction.

> **Invariant:** Do NOT set semantic labels in Unity. The `EntityMarker.Label` property defaults to the GameObject name (debug use only). The backend learns all real labels from user interaction.

| Object | Type | Position | Scale | Components |
|--------|------|----------|-------|------------|
| `Door_LivingToKitchen` | Cube | `(8, 1, 0)` | `(0.1, 2, 1)` | `EntityMarker` + `InteractableState` (OpenClosed, closed) |
| `Lamp_Living` | Cylinder | `(-3, 0.5, 3)` | `(0.3, 0.5, 0.3)` | `EntityMarker` + `InteractableState` (OnOff, off) |
| `Drawer_Kitchen` | Cube | `(17, 0.5, 3)` | `(1, 1, 0.5)` | `EntityMarker` + `InteractableState` (OpenClosed, closed) |
| `Table_Kitchen` | Cube | `(15, 0.4, 0)` | `(2, 0.8, 1)` | `EntityMarker` only |
| `Chair_Kitchen_1` | Cube | `(15, 0.25, 1.5)` | `(0.5, 0.5, 0.5)` | `EntityMarker` only |
| `Couch_Living` | Cube | `(0, 0.4, -3)` | `(2.5, 0.8, 1)` | `EntityMarker` only |

Organize under a parent `Entities` object with sub-groups (`Doors`, `Furniture`, `Lights`, `Containers`).

**Tips:**
- GUIDs auto-generate -- don't set manually
- Objects without `EntityMarker` are invisible to the agent
- Objects without `InteractableState` can be seen but not interacted with

### 3.7 HUD (Backend-Driven Display)

The in-game UI displays state pushed from the Python backend. Unity caches nothing -- every display element resets to "Unknown" when the backend disconnects.

#### Components

| Component | What It Shows | Data Source |
|-----------|---------------|-------------|
| `StatusBarUI` | Current location, confidence, entity count, stability | `location_update` messages from backend |
| `EntityLabelUI` | Floating 3D labels above detected entities | `entity_update` messages from backend |
| `LabelRequestModal` | Modal dialog when backend needs user input | `label_request` messages from backend |
| `UIMessageHandler` | Routes all backend messages to UI components | Subscribes to WebSocket message types |

#### Setup

1. Hierarchy -> **UI** -> **Canvas** -- set Canvas Scaler to **Scale With Screen Size**, reference `1920x1080`
2. Create a top panel with `StatusBarUI` -- child Text elements for location, confidence, entity count
3. Create a centered panel (initially disabled) with `LabelRequestModal` -- prompt text, input field, submit/skip buttons, suggestions container
4. Add `UIMessageHandler` to GameManager -- assign all UI component references

#### Invariants

- All labels and display values come exclusively from backend messages
- Default state for everything is "Unknown" / 0% confidence / yellow stability
- Suggestions in `LabelRequestModal` come only from `label_request.alternative_labels` -- Unity never generates suggestions from scene data
- When the backend disconnects and reconnects, all UI resets to defaults because Unity stores nothing

### 3.8 Final Wiring

1. Select **GameManager** -> find **World Manager** -> drag **Player** to the Player field
2. **File** -> **Save** (Ctrl+S / Cmd+S)

### 3.9 Verify the Connection

1. Click **Play** in Unity -- console should show `WebSocket server started on port 8765`
2. In a terminal:

```bash
source .venv/bin/activate
python -m episodic_agent run --profile unity_full --unity-ws ws://localhost:8765 --fps 10
```

3. Expected output:
```
[0001] #1 Living Room(95%) [door:1 furniture:1] 0
[0002] #2 Living Room(95%) [door:1 furniture:1] 0
```

4. Walk between rooms in Unity -- the agent should detect location transitions

#### Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look |
| E | Interact with objects |
| Shift | Run |
| Escape | Toggle cursor lock |

---

## 4. Runtime Pipeline

Once the system is running, all sensor data flows through a single cognitive loop. This section describes what happens at each tick.

### 4.1 Sensor Frame Loop (Port 8765, ~10 Hz)

Unity sends `sensor_frame` messages containing the camera pose and current room context:

```json
{
  "frame_id": 42,
  "camera": {
    "position": { "x": 2.5, "y": 1.6, "z": -3.2 },
    "forward": { "x": 0.65, "y": -0.26, "z": 0.71 },
    "up": { "x": 0.0, "y": 1.0, "z": 0.0 },
    "yaw": 45.0,
    "pitch": 15.0
  },
  "current_room_guid": "a1b2c3d4-..."
}
```

The backend's `UnityWebSocketSensorProvider` wraps this into a `SensorFrame`, which is passed to `SensorGateway` -> `UnitySensorHandler` -> `SensorMessage` with `LocationContext`.

### 4.2 Visual Summary Loop (Port 8767, ~5 Hz)

Unity's `VisualFeatureExtractor` divides the Sensor Camera's field of view into a 4x4 grid and transmits per-cell features:

```json
{
  "type": "summary",
  "frame_id": 42,
  "cells": [
    {
      "row": 0, "col": 0,
      "color_histogram": [0.1, 0.05, "...24 values..."],
      "edge_directions": [0.2, 0.1, "...8 values..."],
      "mean_brightness": 0.65,
      "edge_density": 0.3,
      "motion_magnitude": 0.05
    }
  ],
  "global_brightness": 0.6,
  "global_motion": 0.1
}
```

The backend's `VisualStreamClient` receives these and computes scene signatures via `VisualFeatureExtractor`.

### 4.3 Unified Ingestion

These two channels are **not independent subsystems**. They converge in the backend's `PerceptionModule.process()` to produce a single `Percept`:

```
Port 8765 (pose stream)  --+
                            +--> PerceptionModule.process() --> Percept
Port 8767 (visual grid)  --+
                                   |
                                   +-- motion_features (from pose deltas):
                                   |     linear_speed, angular_speed,
                                   |     position_delta, rotation_delta
                                   |
                                   +-- visual_features (from grid summaries):
                                         scene_signature, cell features,
                                         global brightness/motion
```

This `Percept` is the sole input to all downstream cognition. There is no path where pose data or visual data is processed independently for cognitive purposes.

### 4.4 The Cognitive Loop

`AgentOrchestrator.step()` runs a 9-step loop on each tick:

```
Percept
  |
  +-- 1. Sensor Read        --  receive SensorFrame + VisualSummary
  +-- 2. Perception          --  fuse into Percept (pose + visual + motion)
  +-- 3. Location Resolution --  "where am I?" (fingerprint matching)
  +-- 4. Entity Resolution   --  "what's around me?" (embedding matching)
  +-- 5. Event Detection     --  "what changed?" (state deltas)
  +-- 6. ACF Update          --  update ActiveContextFrame
  +-- 7. Retrieval           --  check memories for context
  +-- 8. Boundary Check      --  should the current episode end?
  +-- 9. Episode Freeze      --  if boundary: freeze ACF -> Episode -> store
```

---

## 5. Location-First Cognition

The primary cognitive pipeline -- and the current implementation priority -- is robust location identity: answering **"where am I?"** before attempting to answer "what is this?" or "what happened?"

This separation is deliberate. Location identity can stabilize from pose and scene features alone, even when entity recognition is uncertain. Entity and event reasoning are layered on top of a stable location foundation.

### 5.1 End-to-End Location Loop

The complete "where am I?" pipeline:

```
User moves through environment
         |
         v
Unity sends pose + visual summaries
         |
         v
PerceptionModule fuses into Percept
         |
    +----+---------------------------------------------+
    |                                                  |
    v                                                  v
LocationResolverReal                    ACFStabilityGuard
  |                                      |
  +- compute scene embedding             +- STABLE: identity holding
  +- cosine distance vs current          +- MONITORING: minor variation
  |  location centroid                   +- UNCERTAIN: confidence dropping
  +- if distance < 0.40: same place      +- TRANSITIONING: location change
  +- if distance > 0.40 for 5 frames:
  |  transition detected                MotionPerceptionArbitrator
  |                                      |
  +- match candidate embedding           +- perception is authoritative
  |  against known fingerprints          +- motion is advisory
  |  (threshold 0.35)                    +- if agent moved but scene
  |                                         looks identical -> trust
  +- MATCH -> revisit (update centroid,     perception
  |  merge with prior memory)
  |
  +- NO MATCH -> new location
     (create LOCATION node,
      request label from user)
```

### 5.2 Location Fingerprints

Each known location maintains a `LocationFingerprint`:

| Field | Description |
|-------|-------------|
| `centroid` | Running average of scene embeddings observed at this location |
| `observation_count` | How many frames contributed to the centroid |
| `entity_co_occurrence` | Which entities are typically seen here |
| `transition_positions` | Where the agent was when entering/leaving |

When a known location is revisited:
- The centroid updates via running average (new observations refine identity, not replace it)
- `TYPICAL_IN` edge weights increment (strengthening entity-location associations)
- The `ConsolidationModule` merges observations rather than creating duplicate nodes

This prevents **identity fragmentation** -- the same physical location doesn't become multiple memory nodes.

### 5.3 Stability and Arbitration

Two subsystems protect location identity against noise:

**ACFStabilityGuard** -- tracks identity persistence across perceptual variation (lighting changes, shadows, clutter). States: STABLE -> MONITORING -> UNCERTAIN -> TRANSITIONING. Prevents spurious location changes from transient visual differences.

**MotionPerceptionArbitrator** -- resolves conflicts between physical movement (displacement, velocity, rotation) and perceptual similarity. Perception is authoritative; motion is advisory. If the agent moved significantly but the scene looks identical, perception wins. If the agent is stationary but visual features changed dramatically (e.g., a light switched on), the system enters MONITORING rather than immediately declaring a transition.

### 5.4 Memory Formation

When the `HysteresisBoundaryDetector` triggers (multi-factor scoring exceeds high threshold 0.8):

| Factor | Weight | Description |
|--------|--------|-------------|
| Location change | 1.0 | Primary trigger -- entering a new location |
| Salient events | 0.5 | Interesting state changes |
| Prediction error | 0.3 | Scene doesn't match expectations |
| Time elapsed | 0.2 | Natural episode segmentation |

The current `ActiveContextFrame` is frozen into an immutable `Episode` and stored via `PersistentEpisodeStore` (JSONL). Location nodes and `TYPICAL_IN` edges are written to `LabeledGraphStore`.

Hysteresis prevents oscillation: boundary score must cross 0.8 to trigger, then drop below 0.3 before a new boundary can fire.

### 5.5 Exploratory Recognition Under Uncertainty

When location confidence is low, the system enters an exploratory recognition mode rather than committing to an identification prematurely.

This is conceptually inspired by the [Thousand Brains Project's Monty system](https://github.com/thousandbrainsproject/tbp.monty), which uses continuous evidence accumulation for sensorimotor recognition. Key principles adapted from that approach:

- **Evidence is continuous, not binary.** A poor observation degrades a location hypothesis without eliminating it. Confidence accumulates across multiple frames of consistent pose + visual data.
- **Movement generates evidence.** As the agent moves and the scene changes consistently with a known location's fingerprint, confidence rises. Inconsistency reduces it.
- **No premature commitment.** The system maintains the most-likely-hypothesis at every step but does not commit to an identification until evidence exceeds a threshold. This allows graceful handling of ambiguous transitions (e.g., hallways that look similar to each other).

**Key distinction from Monty:** Monty resolves location-on-object and object-identity jointly through the same evidence mechanism. This system maintains a strict separation:
- **Location discovery** ("where am I?") is resolved from pose + scene features via `LocationResolverReal`
- **Entity interpretation** ("what is this?") is resolved from entity embeddings via `EntityResolverReal`

Location identity stabilizes first. Entity reasoning is layered on afterward. This means the system can confidently know *where* it is even when entity recognition is uncertain.

---

## 6. Entity and Event Reasoning (Layered on Location)

Entity and event cognition depend on stable location identity. They are not the current implementation priority but are architecturally complete.

### Entities

`EntityResolverReal` compares entity embeddings against known `GraphNode(ENTITY)` nodes via cosine similarity:
- >= 0.80: re-identification (update running average centroid)
- < 0.80: new entity (create node, request label)

All entity nodes are connected to location nodes via `TYPICAL_IN` edges -- location is the spatial anchor for entity reasoning.

### Events

`EventResolverStateChange` detects state deltas (door opened, light toggled) within episodes. Events create `GraphNode(EVENT)` nodes linked to the entities and locations involved.

### Label Learning

`LabelLearner` manages emergent knowledge from user feedback:
- Observations above 0.7 similarity to a learned label: auto-recognize
- 0.5-0.7: confirm with user
- Below 0.5: request new label

---

## 7. Backend-Owned Labeling

Labels enter the system through exactly one path: the user.

```
Backend detects unknown entity/location
        |
        v
LabelRequest sent to Unity (port 8765)
  +- target_type: entity | location | event
  +- confidence: low | medium | high
  +- alternative_labels: [...] (backend suggestions only)
  +- description: "What is this?"
        |
        v
Unity displays LabelRequestModal
  (or CLIDialogManager prompts in terminal)
        |
        v
User provides / confirms / skips
        |
        v
LabelResponse sent to backend
  +- response_type: provided | confirmed | rejected | timeout | skipped
  +- label: "kitchen" (or null)
        |
        v
Backend stores label in GraphNode.label
Backend creates LabelAssignment
Backend sends entity_update / location_update to Unity for display
```

Unity never stores, caches, or interprets labels. It is a pass-through display. Disconnecting and reconnecting resets all displayed labels to "Unknown."

---

## 8. Debugging and Diagnostics

### Keyboard Shortcuts

| Key | Action | Scope |
|-----|--------|-------|
| **F1** | First-Person debug camera view | Display only -- does not affect backend stream |
| **F2** | Cycle `ProtocolLogger` verbosity: Off -> Headers -> Full JSON | Unity console |
| **F3** | Third-Person debug camera view | Display only |
| **F4** | Toggle `ProtocolLogHUD` overlay | In-game traffic inspector |

### Protocol Logger (Console + File)

The `ProtocolLogger` component logs every message crossing the WebSocket.

**Console verbosity levels:**

| Level | Output |
|-------|--------|
| Off | Silent |
| Headers | `[Protocol >>>] SENSOR_FRAME (1842 bytes)` |
| Full | Headers + pretty-printed JSON payload |

**File logging:** Toggle in Inspector via **Log To File**. Format:

```json
{"ts":"2026-02-06T01:00:00.123Z","dir":"OUT","cat":"SENSOR_FRAME","msg":{...}}
```

Log file location:
- **macOS:** `~/Library/Application Support/DefaultCompany/UnitySensorSim/protocol_traffic.jsonl`
- **Windows:** `%USERPROFILE%\AppData\LocalLow\DefaultCompany\UnitySensorSim\protocol_traffic.jsonl`

```bash
# Tail in real time
tail -f ~/Library/Application\ Support/DefaultCompany/UnitySensorSim/protocol_traffic.jsonl
```

### Protocol Log HUD (In-Game Overlay)

Toggle with **F4**. Shows a scrollable, filterable list of recent messages (last 40). Green = outgoing, orange = incoming. Click the arrow next to any entry to expand full JSON. Filter by Frames / Cmds / Resp.

### Python Test Client

```bash
# Raw JSON dump to stdout
python UnitySensorSim/python_client/sensor_client.py --dump-json

# Log to file for comparison with Unity-side logs
python UnitySensorSim/python_client/sensor_client.py --dump-json --log-file /tmp/client_traffic.jsonl
```

Compare Unity-side and client-side logs:

```bash
diff <(jq -c '.msg' ~/Library/Application\ Support/DefaultCompany/UnitySensorSim/protocol_traffic.jsonl) \
     <(jq -c '.msg' /tmp/client_traffic.jsonl)
```

---

## 9. Automated Scenarios

Once the basic pipeline is working, run automated test scenarios:

```bash
# Run the mixed scenario (walk rooms + toggle objects + spawn ball)
python -m episodic_agent scenario mixed --profile unity_full

# Individual scenarios
# - walk_rooms: Visit multiple rooms
# - toggle_drawer_light: Toggle object states
# - spawn_move_ball: Spawn and manipulate ball
```

---

## 10. Project Structure

```
UnitySensorSim/
+-- Assets/
|   +-- Scenes/
|   |   +-- DemoScene.unity
|   +-- Scripts/
|       +-- Core/
|       |   +-- WebSocketServer.cs       # Port 8765 server
|       |   +-- SensorStreamer.cs         # Sensor frame transmission
|       |   +-- CommandReceiver.cs        # Incoming command handler
|       |   +-- ProtocolMessages.cs       # Message serialization
|       |   +-- ProtocolLogger.cs         # Traffic logger (F2/F4)
|       +-- World/
|       |   +-- RoomVolume.cs             # Room trigger volumes
|       |   +-- EntityMarker.cs           # Entity GUID provider
|       |   +-- InteractableState.cs      # Entity state machine
|       |   +-- WorldManager.cs           # World state coordination
|       |   +-- DynamicWorldBuilder.cs    # Runtime visualization overlays
|       +-- Player/
|       |   +-- FirstPersonController.cs  # WASD + mouse look
|       |   +-- PlayerInteraction.cs      # E to interact
|       +-- Visual/
|       |   +-- VisualSummaryServer.cs    # Port 8767 server
|       |   +-- VisualFeatureExtractor.cs # RenderTexture -> 4x4 grid
|       +-- UI/
|           +-- StatusBarUI.cs            # Location/confidence display
|           +-- EntityLabelUI.cs          # Floating 3D entity labels
|           +-- LabelRequestModal.cs      # User label input dialog
|           +-- UIMessageHandler.cs       # Routes backend -> UI
|           +-- ConnectionHUD.cs          # Connection status
|           +-- ProtocolLogHUD.cs         # Traffic overlay (F4)
+-- protocol/
|   +-- sensor_frame_schema.json
|   +-- command_schema.json
+-- python_client/
    +-- sensor_client.py                  # Standalone test client
```

---

## Troubleshooting

### No scripts found / missing components
1. Check Unity Console for compilation errors
2. Right-click `Assets` -> **Reimport All** -> wait for recompile

### Connection refused in Python
1. Ensure Unity is in **Play Mode**
2. Verify console shows "WebSocket server started"
3. Check port availability: `lsof -i :8765`

### Player falls through floor
1. Ensure Player has **Character Controller** component
2. Position at `(0, 1, 0)` not `(0, 0, 0)`
3. Add a floor plane at `(0, 0, 0)`

### No frames received in Python
1. Move around in Unity to trigger frame updates
2. Check `SensorStreamer` is enabled on GameManager
3. Verify target frame rate >= 10

### Room not detected
1. Ensure room has `RoomVolume` component
2. Verify Box Collider **Is Trigger** is enabled
3. Walk fully into the trigger volume

### Visual channel not connecting
1. Verify `VisualSummaryServer` is running on port 8767
2. Check `enable_visual_channel=True` in the backend profile
3. Verify port availability: `lsof -i :8767`
