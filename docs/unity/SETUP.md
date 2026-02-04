# Unity Setup Guide
u
This guide provides complete instructions for setting up the Unity sensor simulator to work with the Episodic Memory Agent. Even if you've never used Unity before, these steps will get you running.

## Prerequisites

- **Unity Hub** - Download from [unity.com/download](https://unity.com/download)
- **Unity 2022.3 LTS** or later
- **Python 3.11+** with the episodic-agent package installed

## Step 1: Install Unity

### 1.1 Download Unity Hub

1. Go to [unity.com/download](https://unity.com/download)
2. Download Unity Hub for your platform
3. Install and launch Unity Hub

### 1.2 Install Unity Editor

1. In Unity Hub, click **Installs** in the left sidebar
2. Click **Install Editor**
3. Select **Unity 2022.3 LTS** (or latest LTS version)
4. Keep default modules selected
5. Click **Install**

> **Note**: Installation may take 10-30 minutes depending on your internet speed.

## Step 2: Open the Project

### 2.1 Add Project to Unity Hub

1. In Unity Hub, click **Projects** in the left sidebar
2. Click **Add** → **Add project from disk**
3. Navigate to `episodic-memory-agent/UnitySensorSim/`
4. Select the folder and click **Add Project**

### 2.2 Open the Project

1. Click on **UnitySensorSim** in your project list
2. Unity will open and import the project (first time may take several minutes)
3. If prompted about safe mode, click **Enter Safe Mode** then **Exit Safe Mode**

## Step 3: Create the Demo Scene

The project contains scripts but needs a scene to run. Let's create one:

### 3.1 Create a New Scene

1. In Unity, go to **File** → **New Scene**
2. Select **Basic (Built-in)** and click **Create**
3. Go to **File** → **Save As**
4. Navigate to `Assets/Scenes/` (create the folder if needed)
5. Name it `DemoScene` and save

### 3.2 Set Up the GameManager

The GameManager coordinates all sensor streaming.

1. In the **Hierarchy** window (left panel), right-click and select **Create Empty**
2. Rename it to `GameManager`
3. With GameManager selected, in the **Inspector** (right panel):
   - Click **Add Component**
   - Search for and add: `WebSocketServer`
   - Click **Add Component** again
   - Search for and add: `SensorStreamer`
   - Click **Add Component** again
   - Search for and add: `CommandReceiver`
   - Click **Add Component** again
   - Search for and add: `WorldManager`

### 3.3 Configure WebSocket Server

1. Select the **GameManager** object
2. Find the **WebSocket Server** component in the Inspector
3. Set **Port** to `8765`
4. Enable **Auto Start**

### 3.4 Create the Player

1. In Hierarchy, right-click and select **3D Object** → **Capsule**
2. Rename it to `Player`
3. Position it at `(0, 1, 0)` in the Transform component
4. Select the Player, then:
   - Click **Add Component** → `Character Controller`
   - Click **Add Component** → `FirstPersonController`
   - Click **Add Component** → `PlayerInteraction`
5. Set the Player's **Tag** to `Player`:
   - In Inspector, click the **Tag** dropdown
   - Select **Player**

### 3.5 Create the Main Camera

1. Delete the existing Main Camera (select it and press Delete)
2. Right-click on the **Player** object and select **Camera**
3. Rename it to `Main Camera`
4. Position it at `(0, 0.5, 0)` relative to Player (local position)
5. Set its **Tag** to `MainCamera`

### 3.6 Create Rooms

Create at least two rooms for testing:

#### Room 1: Living Room

1. In Hierarchy, right-click → **Create Empty**
2. Rename to `Room_Living`
3. Add component: **Box Collider**
   - Enable **Is Trigger**
   - Set **Size** to `(10, 4, 10)` (10x4x10 meters)
   - Set **Center** to `(0, 2, 0)`
4. Add component: `RoomVolume`
   - Set **Label** to `Living Room`
   - GUID will auto-generate

#### Room 2: Kitchen

1. In Hierarchy, right-click → **Create Empty**
2. Rename to `Room_Kitchen`
3. Position at `(15, 0, 0)` (adjacent to living room)
4. Add component: **Box Collider**
   - Enable **Is Trigger**
   - Set **Size** to `(8, 4, 8)`
   - Set **Center** to `(0, 2, 0)`
5. Add component: `RoomVolume`
   - Set **Label** to `Kitchen`

### 3.7 Create Visual Environment (Recommended)

Adding visual walls and floors makes navigation intuitive and helps you see room boundaries. This section provides detailed instructions.

#### Understanding Unity's Coordinate System

- **Y-axis** is up (height)
- **X-axis** is left/right  
- **Z-axis** is forward/back
- Units are in meters (1 unit = 1 meter)

#### 3.7.1 Create the Floor

1. In Hierarchy, right-click → **3D Object** → **Plane**
2. Rename to `Floor`
3. In the Inspector, set Transform:
   - **Position**: `(7, 0, 0)` - centers between both rooms
   - **Scale**: `(3, 1, 2)` - covers ~30m x 20m area
4. (Optional) Create a material for the floor:
   - In Project window, right-click → **Create** → **Material**
   - Name it `FloorMaterial`
   - Set **Albedo** color to a floor-like color
   - Drag it onto the Floor object

#### 3.7.2 Create Walls for Living Room

Create 4 walls to enclose the Living Room (10x10 area centered at origin):

**Wall: Living Room - North Wall**
1. Hierarchy → **3D Object** → **Cube**
2. Rename to `Wall_Living_North`
3. Set Transform:
   - **Position**: `(0, 2, 5)`
   - **Scale**: `(10, 4, 0.2)`
4. Ensure it does NOT have "Is Trigger" enabled (walls should be solid)

**Wall: Living Room - South Wall**
1. Create another Cube, rename to `Wall_Living_South`
2. Set Transform:
   - **Position**: `(0, 2, -5)`
   - **Scale**: `(10, 4, 0.2)`

**Wall: Living Room - West Wall**
1. Create another Cube, rename to `Wall_Living_West`
2. Set Transform:
   - **Position**: `(-5, 2, 0)`
   - **Scale**: `(0.2, 4, 10)`

**Wall: Living Room - East Wall (with doorway)**
For realism, create this wall with a gap for a doorway:
1. Create Cube, rename to `Wall_Living_East_Top`
2. Set Transform:
   - **Position**: `(5, 3, 0)`
   - **Scale**: `(0.2, 2, 10)`

3. Create another Cube, rename to `Wall_Living_East_Left`
4. Set Transform:
   - **Position**: `(5, 1, -3)`
   - **Scale**: `(0.2, 2, 4)`

5. Create Cube, rename to `Wall_Living_East_Right`
6. Set Transform:
   - **Position**: `(5, 1, 3)`
   - **Scale**: `(0.2, 2, 4)`

This creates a doorway at `(5, 0-2, -1 to 1)` to walk through.

#### 3.7.3 Create Walls for Kitchen

The Kitchen is at position (15, 0, 0), size 8x8:

**Wall: Kitchen - North Wall**
1. Create Cube, rename to `Wall_Kitchen_North`
2. Set Transform:
   - **Position**: `(15, 2, 4)`
   - **Scale**: `(8, 4, 0.2)`

**Wall: Kitchen - South Wall**
1. Create Cube, rename to `Wall_Kitchen_South`
2. Set Transform:
   - **Position**: `(15, 2, -4)`
   - **Scale**: `(8, 4, 0.2)`

**Wall: Kitchen - East Wall**
1. Create Cube, rename to `Wall_Kitchen_East`
2. Set Transform:
   - **Position**: `(19, 2, 0)`
   - **Scale**: `(0.2, 4, 8)`

**Wall: Kitchen - West Wall (connects to Living Room)**
This should have a doorway aligned with the Living Room's east wall doorway:
1. Create Cube, rename to `Wall_Kitchen_West_Top`
2. Set Transform:
   - **Position**: `(11, 3, 0)`
   - **Scale**: `(0.2, 2, 8)`

3. Create Cube, rename to `Wall_Kitchen_West_Left`
4. Set Transform:
   - **Position**: `(11, 1, -2.5)`
   - **Scale**: `(0.2, 2, 3)`

5. Create Cube, rename to `Wall_Kitchen_West_Right`
6. Set Transform:
   - **Position**: `(11, 1, 2.5)`
   - **Scale**: `(0.2, 2, 3)`

#### 3.7.4 Create the Hallway/Corridor

Connect the two rooms with a hallway:

1. Create Cube, rename to `Wall_Hallway_North`
   - **Position**: `(8, 2, 1.5)`
   - **Scale**: `(6, 4, 0.2)`

2. Create Cube, rename to `Wall_Hallway_South`
   - **Position**: `(8, 2, -1.5)`
   - **Scale**: `(6, 4, 0.2)`

3. Add a floor section for the hallway if needed:
   - Create Plane, rename to `Floor_Hallway`
   - **Position**: `(8, 0.01, 0)` (slightly above main floor to avoid z-fighting)
   - **Scale**: `(0.6, 1, 0.3)`

#### 3.7.5 Organize Your Hierarchy

Keep things tidy by organizing walls:

1. In Hierarchy, right-click → **Create Empty**
2. Rename to `Environment`
3. Drag all walls and floors into this parent object
4. Create child empties for organization:
   ```
   Environment
   ├── Floors
   │   ├── Floor
   │   └── Floor_Hallway
   ├── Walls_LivingRoom
   │   ├── Wall_Living_North
   │   ├── Wall_Living_South
   │   └── ...
   ├── Walls_Kitchen
   │   └── ...
   └── Walls_Hallway
       └── ...
   ```

#### 3.7.6 Add Materials for Visual Distinction

Make rooms visually distinct:

1. Create materials for each room:
   - `Mat_Wall_Living` - warm color (beige/cream)
   - `Mat_Wall_Kitchen` - cool color (light blue/white)
   - `Mat_Wall_Hallway` - neutral (gray)

2. Apply materials by:
   - Select wall objects
   - Drag material onto them in Scene view, OR
   - In Inspector, find Mesh Renderer → Materials → Element 0
   - Assign the material

### 3.8 Create Interactable Objects

Objects the agent can detect, track, and potentially interact with. Each object needs an `EntityMarker` component for the agent to recognize it.

#### Understanding EntityMarker Categories

Common categories used by the agent:
- `door` - Doors, gates, entryways
- `furniture` - Tables, chairs, lamps, couches
- `appliance` - Stoves, refrigerators, washing machines
- `container` - Drawers, cabinets, boxes
- `light` - Light sources
- `decoration` - Pictures, plants, ornaments

#### 3.8.1 Create a Door

1. Hierarchy → **3D Object** → **Cube**
2. Rename to `Door_LivingToKitchen`
3. Set Transform (in the doorway between rooms):
   - **Position**: `(8, 1, 0)` - centered in hallway
   - **Scale**: `(0.1, 2, 1)` - thin, door-sized
4. Add component: `EntityMarker`
   - Click **Add Component** → search "EntityMarker"
   - Set **Label** to `Hallway Door`
   - Set **Category** to `door`
   - **GUID** will auto-generate (leave empty)
5. Add component: `InteractableState`
   - Click **Add Component** → search "InteractableState"
   - Set **State Type** to `OpenClosed`
   - Set **Initial State** to `closed`

**Note**: The door starts closed. When the agent or player interacts, it toggles to open/closed.

#### 3.8.2 Create a Lamp

1. Hierarchy → **3D Object** → **Cylinder**
2. Rename to `Lamp_Living`
3. Set Transform:
   - **Position**: `(-3, 0.5, 3)` - corner of living room
   - **Scale**: `(0.3, 0.5, 0.3)` - small table lamp size
4. Add component: `EntityMarker`
   - Set **Label** to `Table Lamp`
   - Set **Category** to `light`
5. Add component: `InteractableState`
   - Set **State Type** to `OnOff`
   - Set **Initial State** to `off`
6. (Optional) Add a Point Light as child:
   - Right-click Lamp_Living → **Light** → **Point Light**
   - Position at `(0, 0.5, 0)` (local, above lamp)
   - The light can be toggled with the InteractableState

#### 3.8.3 Create a Drawer/Cabinet

1. Hierarchy → **3D Object** → **Cube**
2. Rename to `Drawer_Kitchen`
3. Set Transform:
   - **Position**: `(17, 0.5, 3)` - against kitchen wall
   - **Scale**: `(1, 1, 0.5)` - cabinet size
4. Add component: `EntityMarker`
   - Set **Label** to `Kitchen Drawer`
   - Set **Category** to `container`
5. Add component: `InteractableState`
   - Set **State Type** to `OpenClosed`
   - Set **Initial State** to `closed`

#### 3.8.4 Create a Table

1. Hierarchy → **3D Object** → **Cube**
2. Rename to `Table_Kitchen`
3. Set Transform:
   - **Position**: `(15, 0.4, 0)` - center of kitchen
   - **Scale**: `(2, 0.8, 1)` - table proportions
4. Add component: `EntityMarker`
   - Set **Label** to `Kitchen Table`
   - Set **Category** to `furniture`
5. No InteractableState needed (tables don't change state)

#### 3.8.5 Create a Chair

1. Hierarchy → **3D Object** → **Cube** (for simplicity)
2. Rename to `Chair_Kitchen_1`
3. Set Transform:
   - **Position**: `(15, 0.25, 1.5)` - beside table
   - **Scale**: `(0.5, 0.5, 0.5)` - chair size
4. Add component: `EntityMarker`
   - Set **Label** to `Kitchen Chair`
   - Set **Category** to `furniture`

#### 3.8.6 Create a Couch

1. Hierarchy → **3D Object** → **Cube**
2. Rename to `Couch_Living`
3. Set Transform:
   - **Position**: `(0, 0.4, -3)` - living room
   - **Scale**: `(2.5, 0.8, 1)` - couch proportions
4. Add component: `EntityMarker`
   - Set **Label** to `Living Room Couch`
   - Set **Category** to `furniture`

#### 3.8.7 Organize Objects in Hierarchy

Keep objects organized:

```
Entities
├── Doors
│   └── Door_LivingToKitchen
├── Furniture
│   ├── Table_Kitchen
│   ├── Chair_Kitchen_1
│   └── Couch_Living
├── Lights
│   └── Lamp_Living
└── Containers
    └── Drawer_Kitchen
```

#### 3.8.8 Tips for Creating Custom Objects

- **Label** should be human-readable (e.g., "Kitchen Table" not "table_01")
- **Category** should match one of the standard categories
- **GUID** is auto-generated - don't set it manually
- Objects without `EntityMarker` won't be detected by the agent
- Objects without `InteractableState` can be seen but not interacted with

### 3.9 Link References in WorldManager

1. Select **GameManager**
2. Find the **World Manager** component
3. Drag the **Player** object to the **Player** field
4. If you created a ball prefab, assign it to **Ball Prefab**

### 3.10 Save the Scene

1. Press **Ctrl+S** (Windows) or **Cmd+S** (Mac)
2. Ensure the scene is saved in `Assets/Scenes/DemoScene.unity`

## Step 4: Run the Simulation

### 4.1 Enter Play Mode

1. In Unity, click the **Play** button (▶) at the top center
2. The WebSocket server will start automatically
3. You'll see in the Console: `WebSocket server started on port 8765`

### 4.2 Test Controls

While in Play mode:
- **WASD** - Move
- **Mouse** - Look around
- **E** - Interact with objects
- **Shift** - Run
- **Escape** - Toggle cursor lock

### 4.3 Connect the Python Agent

In a terminal:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the agent
python -m episodic_agent run --profile unity_full --unity-ws ws://localhost:8765 --fps 10
```

You should see:
```
[0001] 🟢 #1 📍 Living Room(95%) 👁 [door:1 furniture:1] 📚 0
[0002] 🟢 #2 📍 Living Room(95%) 👁 [door:1 furniture:1] 📚 0
...
```

### 4.4 Test Location Detection

1. In Unity, walk from one room to another
2. The agent should detect the location change
3. If auto-labeling is off, you'll be prompted to label new locations

## Step 5: Run Automated Scenarios

Once basic operation works, try automated scenarios:

```bash
# Run the mixed scenario
python -m episodic_agent scenario mixed --profile unity_full

# Available scenarios:
# - walk_rooms: Visit multiple rooms
# - toggle_drawer_light: Toggle object states
# - spawn_move_ball: Spawn and manipulate ball
# - mixed: All of the above
```

## Project Structure

After setup, your Unity project should look like:

```
UnitySensorSim/
├── Assets/
│   ├── Scenes/
│   │   └── DemoScene.unity      # Your created scene
│   ├── Scripts/
│   │   ├── Core/
│   │   │   ├── WebSocketServer.cs
│   │   │   ├── SensorStreamer.cs
│   │   │   ├── CommandReceiver.cs
│   │   │   └── ProtocolMessages.cs
│   │   ├── World/
│   │   │   ├── RoomVolume.cs
│   │   │   ├── EntityMarker.cs
│   │   │   ├── InteractableState.cs
│   │   │   └── WorldManager.cs
│   │   ├── Player/
│   │   │   ├── FirstPersonController.cs
│   │   │   └── PlayerInteraction.cs
│   │   └── UI/
│   │       └── ConnectionHUD.cs
│   └── Prefabs/               # Optional prefabs
├── protocol/
│   ├── sensor_frame_schema.json
│   └── command_schema.json
└── python_client/
    └── sensor_client.py        # Example Python client
```

## Troubleshooting

### "No scripts found" or Missing Components

If components like `WebSocketServer` don't appear:
1. Check for compilation errors in Unity Console
2. Right-click on `Assets` folder → **Reimport All**
3. Wait for Unity to recompile

### "Connection refused" in Python

1. Ensure Unity is in **Play Mode**
2. Check the Console shows "WebSocket server started"
3. Verify port 8765 isn't used by another application:
   ```bash
   lsof -i :8765
   ```

### Player falls through floor

1. Ensure Player has **Character Controller** component
2. Position Player at `(0, 1, 0)` not `(0, 0, 0)`
3. Add a floor plane at `(0, 0, 0)`

### No frames received in Python

1. Move around in Unity to trigger frame updates
2. Check `SensorStreamer` is enabled on GameManager
3. Set **Target Frame Rate** to at least 10

### Room not detected

1. Ensure room has `RoomVolume` component
2. Verify Box Collider **Is Trigger** is enabled
3. Check Player walks into the trigger volume

## Next Steps

- See [Protocol Documentation](PROTOCOL.md) for message format details
- Read [Troubleshooting Guide](../TROUBLESHOOTING.md) for more solutions
- Try [Custom Scenarios](../scenarios/CUSTOM.md) for advanced testing

## Video Walkthrough

For a visual guide, see the companion video tutorial (if available) or follow Unity's official [Getting Started tutorials](https://learn.unity.com/).
