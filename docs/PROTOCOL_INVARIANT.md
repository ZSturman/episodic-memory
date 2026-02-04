# Protocol Invariants

This document specifies the **non-negotiable protocol constraints** that must be enforced at the Unity↔Python wire boundary. These invariants protect the architectural principles of the episodic memory system.

> **See Also:** [INVARIANTS.md](../INVARIANTS.md) for the complete set of architectural invariants.

---

## Critical Wire-Level Invariants

### 1. No Absolute World Coordinates

**INVARIANT:** Absolute world coordinates MUST NEVER cross the wire.

#### What This Means

- Unity sends positions **relative to the agent** (player camera)
- Distances are computed as `target.position - player.position`
- Rotations/directions are relative to agent's forward vector
- No `Transform.position` values are serialized directly

#### Why This Matters

- Allows protocol to work with any sensor (real robot, phone, VR headset)
- Prevents backend from depending on Unity's coordinate system
- Enables sensor replacement without backend changes

#### Implementation

```csharp
// WRONG: Sending world coordinates
position = entity.transform.position;

// CORRECT: Sending agent-relative coordinates
Vector3 agentPos = playerCamera.position;
Vector3 relativePos = entity.transform.position - agentPos;
position = new Vector3Data(relativePos);
```

---

### 2. No Semantic Labels From Unity

**INVARIANT:** Unity MUST NOT send semantic labels (room names, object categories, event types).

#### What This Means

- No `"Living Room"` in `current_room_label`
- No `"door"`, `"table"`, `"chair"` categories
- No `"DOOR_OPENED"` event types
- Only structural data: GUIDs, positions, states, timestamps

#### Why This Matters

- Labels emerge from user interaction, not scene configuration
- Backend owns all knowledge about what things are called
- Prevents predefined semantics from contaminating the system

#### Implementation

```csharp
// WRONG: Including semantic label
current_room_label = room.Label;  // "Kitchen"

// CORRECT: Only structural identifier
current_room_guid = room.Guid;  // "room-abc123"
// Backend will ask user "What room is this?" when needed
```

---

### 3. Unity is Stateless

**INVARIANT:** Unity MUST NOT maintain cognitive state.

#### What This Means

- Unity does not remember entity labels between frames
- Unity does not track "known" vs "unknown" entities
- Unity does not cache backend responses
- Each frame is a fresh observation

#### Why This Matters

- All cognition happens in Python backend
- Sensor replacement doesn't lose knowledge
- Backend has single source of truth

#### Implementation

```csharp
// WRONG: Caching labels from backend
private Dictionary<string, string> _entityLabels;

void OnBackendLabelUpdate(string guid, string label) {
    _entityLabels[guid] = label;  // Don't do this!
}

// CORRECT: Just display what backend sends, no caching
void OnEntityUpdate(EntityUpdate update) {
    // UI displays update.label directly
    // No local storage
}
```

---

### 4. Backend Controls the Stream

**INVARIANT:** Backend sends commands; Unity executes without question.

#### What This Means

- Backend can change resolution, FPS, crop region
- Backend can request focus on specific regions
- Backend can pause/resume streaming
- Unity complies or reports incapability (never ignores)

#### Protocol Commands

| Command | Unity Action |
|---------|--------------|
| `stream_control.start` | Begin streaming frames |
| `stream_control.stop` | Stop streaming |
| `stream_control.set_resolution` | Change camera resolution |
| `stream_control.set_fps` | Change frame rate |
| `stream_control.set_crop` | Focus on region |
| `stream_control.enable_summary` | Switch to 4×4 grid mode |

---

### 5. Label Requests Are UI-Only

**INVARIANT:** `label_request` triggers UI display only; Unity does NOT interpret requests.

#### What This Means

- Unity shows modal/overlay with backend's question
- Unity captures user input verbatim
- Unity sends `label_response` with exact user text
- Unity does NOT validate, correct, or suggest labels

#### Implementation

```csharp
// WRONG: Interpreting the request
void OnLabelRequest(LabelRequest req) {
    if (req.target_type == "entity") {
        // Don't try to help - just show the prompt
        SuggestLabelFromScene(req.target_id);  // NO!
    }
}

// CORRECT: Pure UI relay
void OnLabelRequest(LabelRequest req) {
    labelModal.Show(
        prompt: req.description,
        suggestions: req.alternative_labels,  // From backend only
        onSubmit: (userLabel) => {
            SendLabelResponse(req.request_id, userLabel);
        }
    );
}
```

---

## Message Flow Diagrams

### Connection Handshake

```
Unity                          Backend
  │                               │
  │───── handshake (sensor) ─────>│
  │<──── handshake (backend) ─────│
  │                               │
  │─── capabilities_report ──────>│
  │                               │
  │<──── stream_control.start ────│
  │                               │
  │═════ sensor_frame (loop) ════>│
```

### Label Acquisition

```
Unity                          Backend
  │                               │
  │════════ sensor_frame ════════>│
  │                               │ (detects unknown entity)
  │<───── label_request ──────────│
  │                               │
  │     [User sees modal]         │
  │     [User types "coffee mug"] │
  │                               │
  │────── label_response ────────>│
  │                               │
  │<───── entity_update ──────────│ (now has label)
```

### Visual Focus

```
Unity                          Backend
  │                               │
  │════ visual_summary (4×4) ════>│
  │                               │ (needs detail in cell 2,3)
  │<───── focus_request ──────────│
  │                               │
  │────── visual_focus ──────────>│ (high-res crop)
```

---

## Verification Checklist

Before any Unity build:

- [ ] Capture WebSocket traffic - verify no world coordinates
- [ ] Search codebase for hardcoded labels - zero matches
- [ ] `label_request` UI shows only backend-provided suggestions
- [ ] Entity positions are all relative to player
- [ ] No `current_room_label` field in frames
- [ ] Stream control commands affect sensor output

---

## Error Handling

When Unity cannot comply with a command:

```json
{
  "message_type": "error",
  "payload": {
    "severity": "error",
    "code": "CAPABILITY_NOT_SUPPORTED",
    "message": "Zoom capability not available",
    "related_message_id": "cmd-001",
    "recoverable": true,
    "suggested_action": "Use set_crop instead of zoom"
  }
}
```

Unity MUST:
1. Report the error immediately
2. Not silently ignore commands
3. Suggest alternatives when possible
4. Continue streaming unless error is critical
