# Unity Communication Protocol

This document describes the WebSocket protocol between the Unity sensor simulator and the Python agent.

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
