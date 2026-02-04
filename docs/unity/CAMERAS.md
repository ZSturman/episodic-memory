# Unity Camera System

This document describes the three-camera architecture for the Unity sensor simulator.

> **ARCHITECTURAL INVARIANT:** Only the Sensor Camera stream goes to the Python backend. Other cameras are for debugging and visualization only.

---

## Camera Overview

| Camera | Purpose | Streams to Backend? |
|--------|---------|---------------------|
| **Sensor Camera** | Agent's actual perception | ✅ Yes |
| **First-Person Camera** | Player view for debugging | ❌ No |
| **Third-Person Camera** | External view for debugging | ❌ No |

```
┌─────────────────────────────────────────────────────────────┐
│                    Unity Scene                               │
│                                                              │
│   ┌──────────┐                                              │
│   │ 3rd      │  Third-Person: Debug overview                │
│   │ Person   │  - Shows full scene                          │
│   │ Camera   │  - Toggle: F3                                │
│   └──────────┘                                              │
│                                                              │
│              ┌─────────────────┐                            │
│              │                 │                            │
│              │   ┌───────┐     │  Player                    │
│              │   │1st    │     │  - First-Person: Player's  │
│              │   │Person │     │    eyes (debug)            │
│              │   └───────┘     │  - Toggle: F1              │
│              │                 │                            │
│              │   ┌───────┐     │                            │
│              │   │Sensor │──────────> Python Backend        │
│              │   │Camera │     │  - Sensor: Actual stream   │
│              │   └───────┘     │  - Toggle: F2              │
│              └─────────────────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Sensor Camera (Primary)

The Sensor Camera is the **only camera that streams to the Python backend**.

### Characteristics

- **Resolution:** Controlled by backend via `stream_control.set_resolution`
- **Frame Rate:** Controlled by backend (default 10 Hz)
- **Crop Region:** Can be dynamically adjusted via `stream_control.set_crop`
- **Position:** Attached to player, represents agent's actual perception

### Configuration

```csharp
[Header("Sensor Camera")]
[SerializeField] private Camera sensorCamera;
[SerializeField] private int defaultWidth = 640;
[SerializeField] private int defaultHeight = 480;
```

### Backend Control

The backend can control the sensor camera via `stream_control` messages:

```json
{
  "message_type": "stream_control",
  "payload": {
    "command": "set_resolution",
    "resolution": [1280, 720]
  }
}
```

---

## First-Person Camera (Debug)

Shows what the player sees. Useful for understanding the player's perspective.

### Characteristics

- **Toggle:** Press F1
- **Purpose:** Debug and development only
- **Does NOT stream to backend**

### Configuration

```csharp
[Header("First-Person Camera")]
[SerializeField] private Camera firstPersonCamera;
[SerializeField] private bool firstPersonActive = true;
```

---

## Third-Person Camera (Debug)

External view of the scene. Useful for understanding spatial relationships.

### Characteristics

- **Toggle:** Press F3
- **Purpose:** Debug and development only
- **Does NOT stream to backend**
- **Position:** Follows player from above/behind

### Configuration

```csharp
[Header("Third-Person Camera")]
[SerializeField] private Camera thirdPersonCamera;
[SerializeField] private Vector3 thirdPersonOffset = new Vector3(0, 5, -8);
[SerializeField] private float thirdPersonSmoothTime = 0.2f;
```

---

## Camera Manager Implementation

### CameraManager.cs

```csharp
using UnityEngine;

namespace EpisodicAgent.Player
{
    /// <summary>
    /// Manages the three-camera system.
    /// Only sensorCamera streams to the Python backend.
    /// </summary>
    public class CameraManager : MonoBehaviour
    {
        [Header("Cameras")]
        [SerializeField] private Camera sensorCamera;
        [SerializeField] private Camera firstPersonCamera;
        [SerializeField] private Camera thirdPersonCamera;

        [Header("Third-Person Settings")]
        [SerializeField] private Transform followTarget;
        [SerializeField] private Vector3 thirdPersonOffset = new Vector3(0, 5, -8);
        [SerializeField] private float smoothTime = 0.2f;

        private Vector3 _velocity;
        private CameraMode _currentMode = CameraMode.FirstPerson;

        public enum CameraMode
        {
            FirstPerson,
            Sensor,
            ThirdPerson
        }

        public Camera ActiveCamera => _currentMode switch
        {
            CameraMode.FirstPerson => firstPersonCamera,
            CameraMode.Sensor => sensorCamera,
            CameraMode.ThirdPerson => thirdPersonCamera,
            _ => firstPersonCamera
        };

        public Camera SensorCamera => sensorCamera;

        private void Update()
        {
            HandleInput();
            UpdateThirdPersonPosition();
        }

        private void HandleInput()
        {
            if (Input.GetKeyDown(KeyCode.F1))
                SetCameraMode(CameraMode.FirstPerson);
            else if (Input.GetKeyDown(KeyCode.F2))
                SetCameraMode(CameraMode.Sensor);
            else if (Input.GetKeyDown(KeyCode.F3))
                SetCameraMode(CameraMode.ThirdPerson);
        }

        public void SetCameraMode(CameraMode mode)
        {
            _currentMode = mode;

            // Update which camera renders to screen
            firstPersonCamera.enabled = (mode == CameraMode.FirstPerson);
            sensorCamera.enabled = true;  // Always enabled for streaming
            thirdPersonCamera.enabled = (mode == CameraMode.ThirdPerson);

            // Update audio listeners
            SetActiveAudioListener(ActiveCamera);
        }

        private void SetActiveAudioListener(Camera activeCamera)
        {
            // Disable all audio listeners first
            if (firstPersonCamera.TryGetComponent<AudioListener>(out var fp))
                fp.enabled = false;
            if (sensorCamera.TryGetComponent<AudioListener>(out var s))
                s.enabled = false;
            if (thirdPersonCamera.TryGetComponent<AudioListener>(out var tp))
                tp.enabled = false;

            // Enable on active camera
            if (activeCamera.TryGetComponent<AudioListener>(out var active))
                active.enabled = true;
        }

        private void UpdateThirdPersonPosition()
        {
            if (thirdPersonCamera == null || followTarget == null)
                return;

            Vector3 targetPos = followTarget.position + 
                               followTarget.rotation * thirdPersonOffset;

            thirdPersonCamera.transform.position = Vector3.SmoothDamp(
                thirdPersonCamera.transform.position,
                targetPos,
                ref _velocity,
                smoothTime
            );

            thirdPersonCamera.transform.LookAt(followTarget);
        }

        /// <summary>
        /// Set sensor camera resolution (called by backend).
        /// </summary>
        public void SetSensorResolution(int width, int height)
        {
            // Implementation depends on render texture setup
            Debug.Log($"Sensor resolution set to {width}x{height}");
        }
    }
}
```

---

## Setup Instructions

### 1. Create Camera Hierarchy

```
Player
├── FirstPersonCamera (Main Camera, Tag: MainCamera)
├── SensorCamera (streams to backend)
└── ThirdPersonCameraRig
    └── ThirdPersonCamera
```

### 2. Configure Each Camera

**First-Person Camera:**
- Position: (0, 0.5, 0) relative to Player
- Clear Flags: Skybox
- Culling Mask: Everything
- Tag: MainCamera

**Sensor Camera:**
- Position: Same as First-Person or slightly different FOV
- Clear Flags: Solid Color (for bandwidth)
- Culling Mask: Everything
- No Tag (not main camera)

**Third-Person Camera:**
- Position: Controlled by script
- Clear Flags: Skybox
- Culling Mask: Everything

### 3. Add CameraManager

1. Add `CameraManager` component to Player
2. Assign all three cameras
3. Set `followTarget` to Player transform

### 4. Verify Streaming

Only `sensorCamera` should be referenced by `SensorStreamer`:

```csharp
// In SensorStreamer.cs
[SerializeField] private CameraManager cameraManager;

// Use sensor camera for visibility checks
private Camera GetSensorCamera() => cameraManager.SensorCamera;
```

---

## Visual Summary Channel

The Sensor Camera also supports the 4×4 visual summary channel (port 8767).

### Summary Mode

When `stream_control.enable_summary` is received:
1. Sensor camera captures full frame
2. Frame is divided into 4×4 grid
3. Each cell computes: color histogram, edge directions, motion
4. Summary sent instead of raw pixels

### Focus Mode

When `focus_request` is received:
1. Backend specifies row/col (0-3)
2. Sensor camera captures that region at high resolution
3. High-res crop sent back

See [PROTOCOL.md](PROTOCOL.md) for message formats.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| F1 | Switch to First-Person Camera |
| F2 | Switch to Sensor Camera view |
| F3 | Switch to Third-Person Camera |

---

## Troubleshooting

### "No camera found for streaming"

Ensure `SensorStreamer` references the Sensor Camera, not the First-Person Camera.

### "Backend receives wrong resolution"

Check that `SetSensorResolution()` is properly hooked to `stream_control` handler.

### "Third-person view is jittery"

Increase `smoothTime` or check that `followTarget` is assigned.
