# Unity Communication Protocol

This document describes the WebSocket protocol between the Unity sensor simulator and the Python agent.

> **See Also:** [INVARIANTS.md](../INVARIANTS.md) for non-negotiable architectural constraints that this protocol enforces.

## Architectural Invariants

This protocol enforces several architectural invariants:

| Invariant | How Protocol Enforces It |
|-----------|-------------------------|
| **No Pre-Wired Semantics** | Labels come from user input, not sensor inference |
| **Protocol is Sensor-Agnostic** | No Unity-specific features; works with any sensor |
| **Labels Come From Users** | `label_request`/`label_response` flow is the only label path |
| **Relative Position Over Categories** | Positions are relative to agent, not world origin |

## Overview

Communication happens over WebSocket on port 8765 (configurable):

```
Unity Simulator                    Python Agent
┌─────────────────┐               ┌────────────────────┐
│  WebSocket      │──────────────>│ Sensor Frames      │
│  Server         │  (JSON)       │ (10-30 Hz)         │
│                 │<──────────────│                    │
│                 │  Commands     │                    │
└─────────────────┘               └────────────────────┘
```

## Message Type Enumeration

All messages include a `message_type` field identifying their purpose:

### Sensor → Backend Messages
| Type | Description |
|------|-------------|
| `sensor_frame` | Raw sensor data (primary data stream) |
| `capabilities_report` | Sensor capabilities announcement |
| `visual_summary` | 4×4 grid visual summary |
| `visual_focus` | High-res crop response |
| `label_response` | User's response to label request |
| `error` | Sensor error report |

### Backend → Sensor Messages
| Type | Description |
|------|-------------|
| `frame_ack` | Acknowledgment of received frame |
| `stream_control` | Control commands (resolution, FPS, crop) |
| `focus_request` | Request high-res region |
| `label_request` | Request label from user |
| `entity_update` | Backend entity state |
| `location_update` | Backend location state |

### Bidirectional Messages
| Type | Description |
|------|-------------|
| `heartbeat` | Keep-alive |
| `handshake` | Connection setup |

---

## Universal Message Envelope

All messages are wrapped in a standard envelope:

```json
{
  "message_id": "uuid-v4",
  "message_type": "sensor_frame",
  "timestamp": "2024-01-01T12:00:00.123Z",
  "correlation_id": "optional-correlation-id",
  "in_reply_to": "optional-request-id",
  "source": "sensor-001",
  "payload": { ... }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message_id` | string | Yes | Unique message identifier (UUID v4) |
| `message_type` | string | Yes | One of the message types above |
| `timestamp` | ISO 8601 | Yes | When message was created |
| `correlation_id` | string | No | Links related messages |
| `in_reply_to` | string | No | ID of message this replies to |
| `source` | string | Yes | Source identifier |
| `payload` | object | Yes | Type-specific payload |

---

## Message Types

### 1. Sensor Frames (Unity → Python)

Streamed at configurable rate (default 10 Hz).

```json
{
  "protocol_version": "1.0.0",
  "frame_id": 42,
  "timestamp": 1704067200.123,
  "camera_pose": {
    "position": { "x": 2.5, "y": 1.6, "z": -3.2 },
    "rotation": { "x": 15.0, "y": 45.0, "z": 0.0 },
    "forward": { "x": 0.65, "y": -0.26, "z": 0.71 }
  },
  "current_room": "room-living-001",
  "current_room_label": "Living Room",
  "entities": [
    {
      "guid": "door-front-001",
      "label": "Front Door",
      "category": "door",
      "position": { "x": 0.0, "y": 1.0, "z": 5.0 },
      "state": "Closed",
      "visible": true,
      "distance": 4.2,
      "interactable": true
    },
    {
      "guid": "lamp-table-001",
      "label": "Table Lamp",
      "category": "furniture",
      "position": { "x": 3.0, "y": 0.8, "z": 2.0 },
      "state": "Off",
      "visible": true,
      "distance": 2.1,
      "interactable": true
    }
  ],
  "state_changes": [
    {
      "entity_guid": "door-front-001",
      "entity_label": "Front Door",
      "old_state": "Closed",
      "new_state": "Open",
      "timestamp": 1704067200.100
    }
  ]
}
```

#### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `protocol_version` | string | Protocol version for compatibility |
| `frame_id` | int | Monotonically increasing frame counter |
| `timestamp` | float | Unix timestamp (seconds) |
| `camera_pose` | object | Player camera position and orientation |
| `current_room` | string | GUID of room player is in |
| `current_room_label` | string | Human label of room (if known) |
| `entities` | array | Visible entities in scene |
| `state_changes` | array | State changes since last frame |

#### Entity Object

| Field | Type | Description |
|-------|------|-------------|
| `guid` | string | Unique identifier (stable across frames) |
| `label` | string | Human-readable label |
| `category` | string | Entity category (door, furniture, etc.) |
| `position` | object | World position {x, y, z} |
| `state` | string | Current state (Open/Closed, On/Off) |
| `visible` | bool | Whether currently visible to camera |
| `distance` | float | Distance from player (meters) |
| `interactable` | bool | Whether entity can be interacted with |

#### State Change Object

| Field | Type | Description |
|-------|------|-------------|
| `entity_guid` | string | Entity that changed |
| `entity_label` | string | Entity label |
| `old_state` | string | Previous state |
| `new_state` | string | New state |
| `timestamp` | float | When change occurred |

### 2. Commands (Python → Unity)

Commands control the simulation for testing.

```json
{
  "command": "teleport_player",
  "command_id": "cmd-001",
  "parameters": {
    "room_guid": "room-kitchen-001"
  }
}
```

#### Available Commands

##### teleport_player

Move player to a specific room.

```json
{
  "command": "teleport_player",
  "command_id": "cmd-001",
  "parameters": {
    "room_guid": "room-kitchen-001"
  }
}
```

##### toggle_interactable

Change entity state.

```json
{
  "command": "toggle_interactable",
  "command_id": "cmd-002",
  "parameters": {
    "entity_guid": "door-front-001",
    "target_state": "Open"
  }
}
```

##### spawn_ball

Spawn a ball at position.

```json
{
  "command": "spawn_ball",
  "command_id": "cmd-003",
  "parameters": {
    "position": { "x": 0, "y": 1, "z": 0 }
  }
}
```

##### move_ball

Move existing ball.

```json
{
  "command": "move_ball",
  "command_id": "cmd-004",
  "parameters": {
    "position": { "x": 5, "y": 1, "z": 3 }
  }
}
```

##### despawn_ball

Remove ball from scene.

```json
{
  "command": "despawn_ball",
  "command_id": "cmd-005",
  "parameters": {}
}
```

##### reset_world

Reset to initial state.

```json
{
  "command": "reset_world",
  "command_id": "cmd-006",
  "parameters": {}
}
```

##### get_world_state

Request full world state.

```json
{
  "command": "get_world_state",
  "command_id": "cmd-007",
  "parameters": {}
}
```

### 3. Command Responses (Unity → Python)

Sent after each command.

```json
{
  "command_id": "cmd-001",
  "success": true,
  "error_message": null,
  "data": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `command_id` | string | ID from original command |
| `success` | bool | Whether command succeeded |
| `error_message` | string | Error description if failed |
| `data` | object | Optional response data |

---

## Capabilities Report (Sensor → Backend)

Sent by sensor on connection to inform backend what it can do.

**ARCHITECTURAL INVARIANT:** Backend must not assume capabilities. All feature requests must be validated against this report.

```json
{
  "message_type": "capabilities_report",
  "payload": {
    "sensor_id": "unity-sim-001",
    "sensor_type": "camera",
    "sensor_version": "1.0.0",
    "capabilities": [
      "rgb_camera",
      "depth_camera",
      "bounding_boxes",
      "tracking"
    ],
    "max_resolution": [1920, 1080],
    "min_resolution": [320, 240],
    "max_fps": 30.0,
    "min_fps": 1.0,
    "compute_available": true,
    "compute_tflops": 2.5,
    "buffer_size_mb": 50,
    "supports_visual_channel": true,
    "supports_focus_requests": true,
    "extras": {}
  }
}
```

### Sensor Capabilities Enum

| Capability | Description |
|------------|-------------|
| `rgb_camera` | Can provide RGB images |
| `depth_camera` | Can provide depth data |
| `stereo_camera` | Has stereo vision |
| `zoom` | Can zoom in/out |
| `focus` | Can focus on regions |
| `pan_tilt` | Can pan/tilt |
| `odometry` | Reports movement |
| `imu` | Has inertial measurement |
| `gps` | Has global positioning |
| `lidar` | Has LIDAR |
| `bounding_boxes` | Can detect bounding boxes |
| `segmentation` | Can do segmentation |
| `tracking` | Can track objects over time |
| `microphone` | Has audio input |
| `speech_to_text` | Can transcribe speech |
| `edge_compute` | Has local compute |
| `feature_extraction` | Can extract features locally |

---

## Stream Control (Backend → Sensor)

Backend commands to control sensor stream.

**ARCHITECTURAL INVARIANT:** Backend controls the stream, not sensor. Sensor must obey or report incapability.

```json
{
  "message_type": "stream_control",
  "payload": {
    "command": "set_resolution",
    "resolution": [640, 480],
    "fps": null,
    "crop_region": null,
    "duration_seconds": null,
    "request_id": "ctrl-001"
  }
}
```

### Stream Control Commands

| Command | Parameters | Description |
|---------|------------|-------------|
| `start` | — | Start streaming |
| `stop` | — | Stop streaming |
| `pause` | — | Pause streaming |
| `resume` | — | Resume streaming |
| `set_resolution` | `resolution: [w, h]` | Change resolution |
| `set_fps` | `fps: float` | Change frame rate |
| `set_crop` | `crop_region: [x, y, w, h]` | Set crop region |
| `clear_crop` | — | Clear crop region |
| `enable_summary` | — | Enable 4×4 summary mode |
| `disable_summary` | — | Disable summary mode |
| `request_keyframe` | — | Request full keyframe |

---

## Label Request/Response Flow

**ARCHITECTURAL INVARIANT:** Labels come from users, not assumptions. This is the ONLY way labels enter the system.

### Label Request (Backend → Sensor)

Backend asks user to provide a label:

```json
{
  "message_type": "label_request",
  "payload": {
    "request_id": "label-001",
    "target_type": "entity",
    "target_id": "entity-abc123",
    "confidence": "low",
    "current_label": null,
    "alternative_labels": [],
    "thumbnail_base64": "<base64 image>",
    "bounding_box": [100, 50, 200, 150],
    "description": "What is this object?",
    "timeout_seconds": 30.0
  }
}
```

### Target Types

| Type | Description |
|------|-------------|
| `entity` | An entity/object |
| `location` | A location/room |
| `event` | An event type |
| `relation` | A relationship |

### Confidence Levels

| Level | Meaning | Behavior |
|-------|---------|----------|
| `low` | No idea, need user input | Request fresh label |
| `medium` | Have a guess, need confirmation | Show suggestion for approval |
| `high` | Confident, informing user | Auto-accept, notify user |

### Label Response (Sensor → Backend)

User's response to a label request:

```json
{
  "message_type": "label_response",
  "payload": {
    "request_id": "label-001",
    "response_type": "provided",
    "label": "coffee mug",
    "confidence": 1.0,
    "notes": null,
    "response_time_ms": 2500.0
  }
}
```

### Response Types

| Type | Description |
|------|-------------|
| `provided` | User provided a label |
| `confirmed` | User confirmed suggestion |
| `rejected` | User rejected all suggestions |
| `timeout` | No response in time |
| `skipped` | User explicitly skipped |

---

## Entity/Location Updates (Backend → Sensor)

Backend informs sensor of current entity/location understanding for UI display.

### Entity Update

```json
{
  "message_type": "entity_update",
  "payload": {
    "entity_id": "entity-abc123",
    "label": "coffee mug",
    "labels": ["coffee mug", "mug", "cup"],
    "confidence": 0.85,
    "visible": true,
    "state": "full",
    "relative_position": [1.2, -0.3, 0.5]
  }
}
```

### Location Update

```json
{
  "message_type": "location_update",
  "payload": {
    "location_id": "loc-xyz789",
    "label": "kitchen",
    "labels": ["kitchen", "cooking area"],
    "confidence": 0.92,
    "is_stable": true,
    "uncertainty_reason": null
  }
}
```

---

## Handshake

Connection handshake to establish protocol version:

```json
{
  "message_type": "handshake",
  "payload": {
    "protocol_version": "1.0.0",
    "role": "sensor",
    "identity": "unity-sim-001",
    "sent_at": "2024-01-01T12:00:00.000Z",
    "session_id": null
  }
}
```

---

## Error Reporting

Structured error reports:

```json
{
  "message_type": "error",
  "payload": {
    "severity": "error",
    "code": "CAPABILITY_NOT_SUPPORTED",
    "message": "Zoom capability not available",
    "related_message_id": "ctrl-001",
    "details": {"requested": "zoom", "available": ["rgb_camera"]},
    "recoverable": true,
    "suggested_action": "Use crop instead of zoom"
  }
}
```

### Error Severities

| Severity | Description |
|----------|-------------|
| `warning` | Non-fatal issue |
| `error` | Recoverable error |
| `critical` | Fatal error |

---

## Connection Lifecycle

### Connection

1. Python client connects to `ws://localhost:8765`
2. Unity accepts connection
3. Unity begins streaming frames

### Frame Streaming

- Frames sent at target rate (default 10 Hz)
- Each frame contains complete world snapshot
- Frame ID increases monotonically

### Command/Response

1. Python sends command JSON
2. Unity executes command
3. Unity sends response with matching `command_id`
4. Next sensor frame reflects changes

### Disconnection

- Either side can close connection
- Unity continues running (can reconnect)
- Python agent handles reconnection automatically

## Protocol Schemas

JSON schemas are in `UnitySensorSim/protocol/`:

- `sensor_frame_schema.json` - Frame validation
- `command_schema.json` - Command validation

## Python Client Example

```python
import asyncio
import json
import websockets

async def sensor_client():
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as ws:
        # Send a command
        command = {
            "command": "teleport_player",
            "command_id": "cmd-001",
            "parameters": {
                "room_guid": "room-kitchen-001"
            }
        }
        await ws.send(json.dumps(command))
        
        # Receive frames
        while True:
            message = await ws.recv()
            data = json.loads(message)
            
            if "frame_id" in data:
                # Sensor frame
                print(f"Frame {data['frame_id']}: {data['current_room_label']}")
                for entity in data.get('entities', []):
                    print(f"  - {entity['label']}: {entity['state']}")
                    
            elif "command_id" in data:
                # Command response
                print(f"Command {data['command_id']}: {data['success']}")

asyncio.run(sensor_client())
```

## Frame Rate Considerations

| FPS | Use Case |
|-----|----------|
| 5 | Low-power testing |
| 10 | Default, balanced |
| 30 | High-fidelity capture |

Higher frame rates provide more temporal resolution but increase processing load.

## Error Handling

### Invalid Command

```json
{
  "command_id": "cmd-001",
  "success": false,
  "error_message": "Unknown command: invalid_command"
}
```

### Entity Not Found

```json
{
  "command_id": "cmd-002",
  "success": false,
  "error_message": "Entity not found: invalid-guid"
}
```

### Connection Lost

The Python agent handles reconnection:
- Automatic retry with exponential backoff
- Frames buffered during reconnection
- State preserved across reconnects

---

## Visual Summary Channel (Port 8767)

**ARCHITECTURAL INVARIANT: Bandwidth-Efficient Sensing**

The visual summary channel provides bandwidth-efficient visual context without raw image streaming.
This design enables deployment on bandwidth-constrained real-world systems.

```
Unity Simulator                    Python Agent
┌─────────────────┐               ┌────────────────────┐
│  Visual Channel │──────────────>│ 4×4 Grid Summaries │
│  Server (8767)  │  (JSON, ~5Hz) │ (features only)    │
│                 │<──────────────│                    │
│                 │  Focus Reqs   │                    │
│                 │──────────────>│                    │
│                 │  High-res Crop│ Ring Buffer        │
└─────────────────┘               └────────────────────┘
```

### Visual Summary Messages (Unity → Python)

Streamed at ~5 Hz with extracted features (no raw pixels).

```json
{
  "type": "summary",
  "frame_id": 42,
  "timestamp": 1704067200.123,
  "cells": [
    {
      "row": 0,
      "col": 0,
      "color_histogram": [0.1, 0.05, ...],  // 24 values (8 bins × 3 channels)
      "edge_directions": [0.2, 0.1, ...],   // 8 values (directional bins)
      "mean_brightness": 0.65,
      "edge_density": 0.3,
      "motion_magnitude": 0.05
    },
    // ... 15 more cells for 4×4 grid
  ],
  "global_brightness": 0.6,
  "global_edge_density": 0.35,
  "global_motion": 0.1,
  "dominant_colors": [[128, 64, 32], [200, 180, 160], [50, 50, 80]],
  "high_res_available": true
}
```

#### Cell Fields

| Field | Type | Description |
|-------|------|-------------|
| `row` | int | Row index (0-3) |
| `col` | int | Column index (0-3) |
| `color_histogram` | float[] | RGB histogram (24 bins: 8×3 channels) |
| `edge_directions` | float[] | Edge direction histogram (8 bins) |
| `mean_brightness` | float | Average brightness (0-1) |
| `edge_density` | float | Edge pixels ratio (0-1) |
| `motion_magnitude` | float | Motion since last frame |

### Focus Requests (Python → Unity)

Request high-resolution data on demand.

#### Region Focus

Focus on specific grid cell:

```json
{
  "type": "focus_request",
  "request_id": "focus_abc123",
  "request_type": "region",
  "target_row": 2,
  "target_col": 3,
  "duration_seconds": 1.0,
  "priority": 5
}
```

#### Full Frame

Request entire high-resolution frame:

```json
{
  "type": "focus_request",
  "request_id": "full_def456",
  "request_type": "full_frame",
  "frame_id": 42,
  "duration_seconds": 2.0,
  "priority": 8
}
```

#### Entity Tracking

Auto-focus on entity:

```json
{
  "type": "focus_request",
  "request_id": "track_ghi789",
  "request_type": "track",
  "entity_guid": "lamp-table-001",
  "duration_seconds": 5.0,
  "priority": 7
}
```

### Focus Response (Unity → Python)

Response with high-resolution image data:

```json
{
  "type": "focus_response",
  "request_id": "focus_abc123",
  "success": true,
  "timestamp": 1704067201.456,
  "image_data": "<base64 encoded JPEG>",
  "image_format": "jpeg",
  "image_width": 320,
  "image_height": 240,
  "source_frame_id": 42,
  "crop_x": 160,
  "crop_y": 120,
  "crop_width": 160,
  "crop_height": 120
}
```

### Ring Buffer

Python maintains a fixed-size ring buffer (~50MB) for recent high-res frames:
- Auto-evicts oldest frames when size limit exceeded
- Enables retrospective analysis of recent visual context
- Raw images discarded after feature extraction

### Design Rationale

1. **Bandwidth Efficiency**: 4×4 grid summaries (~2KB/frame) vs raw images (~500KB/frame)
2. **On-Demand Detail**: High-res only when recognition requires it
3. **Memory Budget**: Fixed ring buffer prevents unbounded memory growth
4. **Feature Extraction**: Python extracts and stores features, discards raw pixels
5. **Attention-Driven**: Saliency-based attention guides focus requests
