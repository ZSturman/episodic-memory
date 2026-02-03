# UnitySensorSim - Episodic Memory Agent Sensor Simulator

A Unity-based sensor simulator that streams world state to the Python episodic memory agent via WebSocket.

## Overview

This Unity project provides:
- **Two-room playable environment** for testing episodic memory
- **First-person controller** with WASD + mouse look
- **Interactable objects** (doors, lamps, switches)
- **WebSocket sensor streaming** at configurable rate (10-30 Hz)
- **Optional command channel** for Python → Unity testing commands
- **HUD** showing connection status and sensor info

## Requirements

- Unity 2022.3 LTS or later
- No external packages required (uses built-in networking)

## Quick Start

### 1. Open in Unity

1. Open Unity Hub
2. Add project from: `UnitySensorSim/`
3. Open with Unity 2022.3+
4. Open scene: `Assets/Scenes/DemoScene`

### 2. Configure WebSocket Server

The `WebSocketServer` component (on the GameManager object) has:
- **Port**: Default 8765 (configurable)
- **Auto Start**: Enable to start server on play

### 3. Play and Connect

1. Press Play in Unity
2. Run your Python agent with WebSocket client on port 8765
3. Walk around, interact with objects, and observe sensor data

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look around |
| Shift | Run |
| Space | Jump |
| E | Interact |
| Escape | Toggle cursor lock |
| F1 | Toggle HUD |

## Architecture

```
UnitySensorSim/
├── Assets/
│   ├── Scripts/
│   │   ├── Core/
│   │   │   ├── ProtocolMessages.cs    # JSON message schemas
│   │   │   ├── WebSocketServer.cs     # Lightweight WS server
│   │   │   ├── SensorStreamer.cs      # Sensor frame building
│   │   │   └── CommandReceiver.cs     # Python command handling
│   │   ├── World/
│   │   │   ├── RoomVolume.cs          # Room trigger zones
│   │   │   ├── EntityMarker.cs        # Entity tracking
│   │   │   ├── InteractableState.cs   # Toggle state (On/Off, Open/Closed)
│   │   │   └── WorldManager.cs        # Central world state
│   │   ├── Player/
│   │   │   ├── FirstPersonController.cs
│   │   │   └── PlayerInteraction.cs
│   │   └── UI/
│   │       └── ConnectionHUD.cs
│   ├── Scenes/
│   │   └── DemoScene.unity
│   └── Prefabs/
├── protocol/
│   ├── sensor_frame_schema.json
│   ├── command_schema.json
│   └── examples/
└── README.md
```

## Protocol

### Sensor Frame (Unity → Python)

Streamed at configurable rate (default 10 Hz):

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
      "distance": 4.2
    }
  ],
  "state_changes": []
}
```

### Command Messages (Python → Unity)

Supported commands:
- `teleport_player` - Move player to a room
- `toggle_interactable` - Change entity state
- `spawn_ball` / `despawn_ball` / `move_ball` - Ball manipulation
- `reset_world` - Reset to initial state
- `get_world_state` - Request full state

Example:
```json
{
  "command": "toggle_interactable",
  "command_id": "cmd-001",
  "parameters": {
    "entity_guid": "door-front-001",
    "target_state": "Open"
  }
}
```

### Command Response

```json
{
  "command_id": "cmd-001",
  "success": true,
  "error_message": null
}
```

## Setting Up the Demo Scene

### 1. Create GameManager

1. Create empty GameObject named "GameManager"
2. Add components:
   - `WebSocketServer`
   - `SensorStreamer`
   - `CommandReceiver`
   - `WorldManager`
   - `ConnectionHUD`

### 2. Create Player

1. Create Capsule named "Player"
2. Add tag "Player"
3. Add components:
   - `CharacterController`
   - `FirstPersonController`
   - `PlayerInteraction`
4. Add child Camera

### 3. Create Rooms

For each room:
1. Create empty GameObject with BoxCollider (trigger)
2. Add `RoomVolume` component
3. Set Label (e.g., "Living Room")
4. GUID auto-generates

### 4. Create Interactables

For each interactable object:
1. Add `EntityMarker` component (set Label, Category)
2. Add `InteractableState` component (choose On/Off or Open/Closed)
3. Optionally add Animator for state visuals

### 5. Create Ball Prefab

1. Create Sphere with Rigidbody
2. Add `EntityMarker` component
3. Save as prefab
4. Assign to WorldManager's Ball Prefab slot

## Python Client Example

```python
import asyncio
import websockets
import json

async def sensor_receiver():
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as ws:
        # Send a command
        command = {
            "command": "toggle_interactable",
            "command_id": "test-001",
            "parameters": {
                "entity_guid": "door-front-001"
            }
        }
        await ws.send(json.dumps(command))
        
        # Receive sensor frames
        while True:
            message = await ws.recv()
            data = json.loads(message)
            
            if "frame_id" in data:
                print(f"Frame {data['frame_id']}: Room={data.get('current_room_label')}")
                print(f"  Entities: {len(data.get('entities', []))}")
                for change in data.get('state_changes', []):
                    print(f"  Change: {change}")
            elif "command_id" in data:
                print(f"Command response: {data}")

asyncio.run(sensor_receiver())
```

## Configuration

### SensorStreamer Settings

| Property | Default | Description |
|----------|---------|-------------|
| Target Frame Rate | 10 | Sensor frames per second |
| Include All Entities | true | Send all entities or only visible |
| Include State Changes | true | Include state_changes array |

### WebSocketServer Settings

| Property | Default | Description |
|----------|---------|-------------|
| Port | 8765 | WebSocket server port |
| Auto Start | true | Start server on Awake |

## Troubleshooting

### Connection Issues
- Ensure Unity is in Play mode
- Check port isn't blocked by firewall
- Verify HUD shows "Connected"

### No Sensor Data
- Check SensorStreamer is enabled
- Verify WorldManager has player reference
- Check Python client is receiving on correct port

### Entity Not Found
- Ensure entity has `EntityMarker` component
- Verify GUID matches (check Unity Inspector)
- Call `WorldManager.DiscoverWorldObjects()` after scene changes

## Integration with Episodic Agent

The sensor frames map directly to the agent's perception:

| Sensor Data | Agent Concept |
|-------------|---------------|
| `current_room` | Location context |
| `entities` | What's here |
| `state_changes` | What changed |
| `camera_pose` | Agent viewpoint |

The agent uses these to:
1. Detect location boundaries (room changes)
2. Build entity graphs per location
3. Track state changes for episode segmentation
4. Apply hysteresis for stable boundaries

## License

Part of the Episodic Memory Agent project.
