# Data Contracts

All data flows through Pydantic v2 models defined in `src/episodic_agent/schemas/`. These contracts ensure type safety and enable validation at module boundaries.

## Design Principles

1. **Immutability** - Use frozen models where appropriate
2. **Forward Compatibility** - All models include `extras: dict` field
3. **Validation** - Pydantic handles all validation
4. **Serialization** - JSON-serializable for logging and persistence
5. **Relative Coordinates** - All positions are agent-relative (see below)

---

## Coordinate System Invariants

> **ARCHITECTURAL INVARIANT:** Absolute world coordinates MUST NEVER cross the wire or enter storage.

### All Positions Are Agent-Relative

Every position in the system is expressed relative to the agent (player/camera):

```
Agent Position = (0, 0, 0)  // Agent is always at origin
Entity at (2.5, 0, 3.0)     // 2.5m right, 3.0m forward of agent
```

### Why Relative Coordinates?

1. **Sensor Portability** - Protocol works with any sensor (robot, phone, VR)
2. **No Coordinate Assumptions** - Backend doesn't depend on Unity's world origin
3. **Natural Navigation** - Relative positions match human spatial reasoning

### Coordinate Fields

| Field | Frame | Description |
|-------|-------|-------------|
| `position.x` | Agent | Meters right (+) or left (-) of agent |
| `position.y` | Agent | Meters above (+) or below (-) agent |
| `position.z` | Agent | Meters forward (+) or behind (-) agent |
| `distance` | Agent | Euclidean distance from agent (always positive) |

### Landmark-Relative Storage

For persistent storage, positions are stored relative to recognized landmarks:

```python
# Stored edge: "mug is 0.3m left of table"
edge = GraphEdge(
    source_id="mug-001",
    target_id="table-001",
    edge_type=EDGE_TYPE_NEAR,
    properties={
        "relative_position": (-0.3, 0.0, 0.0),  # Landmark-relative
        "reference_frame": "table-001",
    }
)
```

---

## Core Schemas

### SensorFrame

Raw sensor data from any source.

```python
class SensorFrame(BaseModel):
    """Raw sensor input from any source."""
    
    frame_id: int                    # Unique frame identifier
    timestamp: datetime              # Frame timestamp
    raw_data: dict[str, Any]         # Raw sensor payload
    sensor_type: str                 # Source type (e.g., "unity_websocket")
    extras: dict[str, Any] = {}      # Forward compatibility

# Unity-specific extras:
# - current_room_guid: str (room GUID only - no label!)
# - entities: list[dict] (visible entities with RELATIVE positions)
# - camera_pose: dict (orientation only - position is always origin)
# - state_changes: list[dict] (state changes since last frame)
#
# NOTE: current_room_label is NOT included - backend learns labels from users
```

**Location**: `schemas/frames.py`

---

### Percept

Processed perception with embeddings and candidates.

```python
class Percept(BaseModel):
    """Processed perception from a sensor frame."""
    
    percept_id: str                           # Unique identifier
    timestamp: datetime                        # Processing timestamp
    scene_embedding: list[float]              # Scene-level embedding vector
    objects: list[ObjectCandidate] = []       # Recognized objects
    raw_frame: SensorFrame | None = None      # Optional: original frame
    extras: dict[str, Any] = {}               # Forward compatibility
```

**Location**: `schemas/frames.py`

---

### ObjectCandidate

A recognized object/entity.

```python
class ObjectCandidate(BaseModel):
    """A recognized object candidate."""
    
    candidate_id: str                  # Unique identifier
    label: str = "unknown"             # Best-guess label
    labels: list[str] = []             # Alternative label candidates
    confidence: float = 0.0            # Recognition confidence [0, 1]
    embedding: list[float] | None = None   # Vector embedding for similarity
    position: tuple[float, float, float] | None = None  # 3D position
    bounding_box: dict[str, float] | None = None  # Bounding box coords
    conflict_id: str | None = None     # Label conflict reference
    extras: dict[str, Any] = {}        # Forward compatibility

# Common extras (sensor-specific, NOT semantic):
# - interactable_state: str (Open, Closed, On, Off)
# - is_visible: bool
# - distance: float
# NOTE: labels come from users via LabelRequest, not from sensors.
```

**Location**: `schemas/frames.py`

---

### ActiveContextFrame (ACF)

Mutable working memory accumulating episode information.

```python
class ActiveContextFrame(BaseModel):
    """Active context frame - mutable working memory."""
    
    acf_id: str                                    # Unique identifier
    step_count: int = 0                            # Steps in this episode
    location_label: str = "unknown"                # Current location
    location_confidence: float = 0.0               # Location confidence
    entities: list[ObjectCandidate] = []           # Current entities
    events: list[dict] = []                        # Accumulated events
    deltas: list[Delta] = []                       # Recent changes
    start_time: datetime                           # Episode start
    last_update: datetime                          # Last update time
    extras: dict[str, Any] = {}                    # Forward compatibility

    def touch(self) -> None:
        """Update last_update timestamp."""
        self.last_update = datetime.now()
```

**Location**: `schemas/context.py`

---

### Episode

Frozen snapshot of an ACF at episode boundary.

```python
class Episode(BaseModel):
    """Frozen episode snapshot."""
    
    episode_id: str                    # Unique identifier
    source_acf_id: str                 # Original ACF ID
    location_label: str                # Primary location
    start_time: datetime               # Episode start
    end_time: datetime                 # Episode end
    step_count: int                    # Total steps
    entities: list[str] = []           # Entity IDs present
    events: list[dict] = []            # Events that occurred
    summary: str = ""                  # Optional text summary
    boundary_reason: str = ""          # Why episode ended
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/context.py`

---

### Delta

A detected change between steps.

```python
class DeltaType(str, Enum):
    """Types of deltas."""
    
    NEW = "new"                 # Entity appeared
    MISSING = "missing"         # Entity disappeared
    MOVED = "moved"             # Entity position changed
    STATE_CHANGED = "state_changed"  # Entity state changed


class Delta(BaseModel):
    """A detected change."""
    
    delta_type: DeltaType           # Type of change
    entity_id: str                  # Affected entity
    entity_label: str = ""          # Entity label
    old_value: Any = None           # Previous value
    new_value: Any = None           # New value
    extras: dict[str, Any] = {}     # Forward compatibility
```

**Location**: `schemas/events.py`

---

### EventCandidate

A detected event.

```python
class EventType(str, Enum):
    """Types of events."""
    
    STATE_CHANGE = "state_change"    # Entity state changed
    APPEARANCE = "appearance"         # Entity appeared
    DISAPPEARANCE = "disappearance"  # Entity disappeared
    MOVEMENT = "movement"            # Entity moved
    INTERACTION = "interaction"      # User interaction
    LOCATION_CHANGE = "location_change"  # Location changed


class EventCandidate(BaseModel):
    """A detected event."""
    
    event_type: EventType              # Type of event
    entity_id: str                     # Primary entity involved
    label: str = ""                    # Event label
    confidence: float = 1.0            # Detection confidence
    timestamp: datetime                # When event occurred
    details: dict[str, Any] = {}       # Event-specific details
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/events.py`

---

### GraphNode

A node in the associative memory graph.

```python
# Node types are string constants defined in schemas/graph.py:
# LOCATION, ENTITY, EVENT, EPISODE, CONCEPT


class GraphNode(BaseModel):
    """A node in the associative graph memory.
    
    ARCHITECTURAL INVARIANT: node_type is structural, not semantic.
    Semantic meaning is stored in label/labels fields, learned from user.
    """
    
    node_id: str                       # Unique identifier
    node_type: str                     # Structural type (location/entity/event/concept)
    label: str = "unknown"             # Primary label (learned from user)
    labels: list[str] = []             # Alternative/hierarchical labels
    embedding: list[float] | None = None  # Vector embedding for similarity
    activation: float = 0.0            # Current activation level
    base_activation: float = 0.0       # Base-level activation (recency/frequency)
    created_at: datetime               # Creation time
    last_accessed: datetime            # Last activation time
    access_count: int = 0              # Times this node has been accessed
    source_id: str | None = None       # Source object reference
    confidence: float = 0.0            # Confidence in node identity
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/graph.py`

---

### GraphEdge

An edge connecting nodes in the graph.

```python
# Edge types are string constants, not an enum.
# Defined in schemas/graph.py as: TYPICAL_IN, RELATED_TO, CAUSED_BY,
# PART_OF, CO_OCCURS, TRIGGERED_BY, OCCURRED_IN, ALIAS_OF, MERGED_INTO,
# CONTAINS, OCCURRED_AT, SIMILAR_TO, TEMPORAL, CAUSAL


class GraphEdge(BaseModel):
    """An edge in the associative graph.
    
    ARCHITECTURAL INVARIANT: edge_type is structural, not semantic.
    Salience weights are learned from experience, not predefined.
    """
    
    edge_id: str                       # Unique identifier
    edge_type: str                     # Structural type of relationship
    source_node_id: str                # Source node ID
    target_node_id: str                # Target node ID
    weight: float = 1.0                # Edge weight for activation spreading
    confidence: float = 0.0            # Confidence in this relationship
    salience: dict[str, float] = {}    # Per-cue-type salience weights
    created_at: datetime               # Creation time
    last_accessed: datetime            # Last traversal time
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/graph.py`

---

### RetrievalResult

Result from memory retrieval.

```python
class RetrievalResult(BaseModel):
    """Result from memory retrieval."""
    
    query_id: str                      # Query identifier
    episodes: list[Episode] = []       # Retrieved episodes
    nodes: list[GraphNode] = []        # Retrieved graph nodes
    edges: list[GraphEdge] = []        # Retrieved graph edges
    scores: dict[str, float] = {}      # Relevance scores by ID
    extras: dict[str, Any] = {}        # Forward compatibility
```

**Location**: `schemas/results.py`

---

### StepResult

Complete result of one cognitive step (for logging).

```python
class StepResult(BaseModel):
    """Result of a single cognitive step."""
    
    step_number: int                   # Step counter
    timestamp: datetime                # Step timestamp
    location_label: str                # Resolved location
    location_confidence: float         # Location confidence
    entity_count: int                  # Number of entities
    event_count: int                   # Number of events
    episode_count: int                 # Total episodes so far
    boundary_triggered: bool           # Whether episode froze
    boundary_reason: str | None = None # Freeze reason if any
    frame_id: int | None = None        # Source frame ID
    extras: dict[str, Any] = {}        # Forward compatibility

    def to_log_dict(self) -> dict:
        """Convert to dictionary for JSONL logging."""
        return self.model_dump(mode="json")
```

**Location**: `schemas/results.py`

---

## Validation

Pydantic handles validation automatically:

```python
# This will raise ValidationError
frame = SensorFrame(
    frame_id="not_an_int",  # Error: expected int
    timestamp=datetime.now(),
    raw_data={},
    sensor_type="test",
)

# Correct usage
frame = SensorFrame(
    frame_id=1,
    timestamp=datetime.now(),
    raw_data={"example": "data"},
    sensor_type="test",
)
```

## Serialization

All models serialize to JSON:

```python
# To JSON string
json_str = frame.model_dump_json()

# To dict
data = frame.model_dump()

# From JSON
frame = SensorFrame.model_validate_json(json_str)

# From dict
frame = SensorFrame.model_validate(data)
```

## Forward Compatibility

The `extras` field allows adding data without schema changes:

```python
# Adding custom data
frame = SensorFrame(
    frame_id=1,
    timestamp=datetime.now(),
    raw_data={},
    sensor_type="test",
    extras={
        "custom_field": "custom_value",
        "experiment_id": "exp_001",
    },
)

# Accessing extras
if "custom_field" in frame.extras:
    value = frame.extras["custom_field"]
```

---

## Protocol Schemas

Wire protocol message schemas for sensor-backend communication.

**Location**: `schemas/protocol.py`

> **ARCHITECTURAL INVARIANT:** Protocol is sensor-agnostic and portable. No Unity-specific shortcuts. All messages work with physical sensors.

### MessageType

All valid protocol message types:

```python
class MessageType(str, Enum):
    """All valid protocol message types."""
    
    # Sensor → Backend
    SENSOR_FRAME = "sensor_frame"
    CAPABILITIES_REPORT = "capabilities_report"
    VISUAL_SUMMARY = "visual_summary"
    VISUAL_FOCUS = "visual_focus"
    LABEL_RESPONSE = "label_response"
    ERROR = "error"
    
    # Backend → Sensor
    FRAME_ACK = "frame_ack"
    STREAM_CONTROL = "stream_control"
    FOCUS_REQUEST = "focus_request"
    LABEL_REQUEST = "label_request"
    ENTITY_UPDATE = "entity_update"
    LOCATION_UPDATE = "location_update"
    
    # Bidirectional
    HEARTBEAT = "heartbeat"
    HANDSHAKE = "handshake"
```

---

### CapabilitiesReport

Sensor capabilities announcement sent on connection.

```python
class SensorCapability(str, Enum):
    """Capabilities a sensor may report."""
    
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    STEREO_CAMERA = "stereo_camera"
    ZOOM = "zoom"
    FOCUS = "focus"
    PAN_TILT = "pan_tilt"
    ODOMETRY = "odometry"
    IMU = "imu"
    GPS = "gps"
    LIDAR = "lidar"
    BOUNDING_BOXES = "bounding_boxes"
    SEGMENTATION = "segmentation"
    TRACKING = "tracking"
    MICROPHONE = "microphone"
    SPEECH_TO_TEXT = "speech_to_text"
    EDGE_COMPUTE = "edge_compute"
    FEATURE_EXTRACTION = "feature_extraction"


class CapabilitiesReport(BaseModel):
    """Sensor capabilities announcement."""
    
    sensor_id: str                              # Unique sensor identifier
    sensor_type: str                            # Type of sensor
    sensor_version: str = "1.0.0"               # Protocol version
    capabilities: list[SensorCapability] = []   # Supported capabilities
    max_resolution: tuple[int, int] | None = None  # Maximum resolution
    min_resolution: tuple[int, int] | None = None  # Minimum resolution
    max_fps: float = 30.0                       # Maximum frame rate
    min_fps: float = 1.0                        # Minimum frame rate
    compute_available: bool = False             # Edge compute available
    compute_tflops: float | None = None         # Compute power
    buffer_size_mb: int = 50                    # Buffer memory
    supports_visual_channel: bool = False       # Visual summary support
    supports_focus_requests: bool = False       # Focus request support
    extras: dict[str, Any] = {}                 # Additional metadata
```

---

### StreamControl

Backend command to control sensor stream.

```python
class StreamControlCommand(str, Enum):
    """Commands backend can send to control sensor stream."""
    
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    SET_RESOLUTION = "set_resolution"
    SET_FPS = "set_fps"
    SET_CROP = "set_crop"
    CLEAR_CROP = "clear_crop"
    ENABLE_SUMMARY = "enable_summary"
    DISABLE_SUMMARY = "disable_summary"
    REQUEST_KEYFRAME = "request_keyframe"


class StreamControl(BaseModel):
    """Backend command to control sensor stream."""
    
    command: StreamControlCommand               # Control command
    resolution: tuple[int, int] | None = None   # For SET_RESOLUTION
    fps: float | None = None                    # For SET_FPS
    crop_region: tuple[int, int, int, int] | None = None  # For SET_CROP
    duration_seconds: float | None = None       # Temporary duration
    request_id: str | None = None               # Response correlation
```

---

### LabelRequest

Backend request for user to provide a label.

> **ARCHITECTURAL INVARIANT:** Labels come from users, not assumptions. This is the ONLY way labels enter the system.

```python
class LabelTargetType(str, Enum):
    """What kind of thing needs a label."""
    
    ENTITY = "entity"
    LOCATION = "location"
    EVENT = "event"
    RELATION = "relation"


class LabelConfidence(str, Enum):
    """Confidence level triggering the request."""
    
    LOW = "low"        # Need user input
    MEDIUM = "medium"  # Need confirmation
    HIGH = "high"      # Informing user


class LabelRequest(BaseModel):
    """Backend request for user to provide a label."""
    
    request_id: str                             # Unique request ID
    target_type: LabelTargetType                # Type of thing to label
    target_id: str                              # ID of the target
    confidence: LabelConfidence                 # Why we're asking
    current_label: str | None = None            # Current best guess
    alternative_labels: list[str] = []          # Other possibilities
    thumbnail_base64: str | None = None         # Visual context
    bounding_box: tuple[int, int, int, int] | None = None  # Bounding box
    description: str = ""                       # Human-readable prompt
    timeout_seconds: float = 30.0               # Response timeout
```

---

### LabelResponse

User's response to a label request.

```python
class LabelResponseType(str, Enum):
    """How user responded to label request."""
    
    PROVIDED = "provided"      # User provided a label
    CONFIRMED = "confirmed"    # User confirmed suggestion
    REJECTED = "rejected"      # User rejected all suggestions
    TIMEOUT = "timeout"        # No response in time
    SKIPPED = "skipped"        # User explicitly skipped


class LabelResponse(BaseModel):
    """User's response to a label request."""
    
    request_id: str                     # Original request ID
    response_type: LabelResponseType    # How user responded
    label: str | None = None            # The label provided/confirmed
    confidence: float = 1.0             # User's confidence [0, 1]
    notes: str | None = None            # Optional user notes
    response_time_ms: float | None = None  # Response time
```

---

### ProtocolMessage

Universal protocol message envelope.

```python
class ProtocolMessage(BaseModel):
    """Universal protocol message envelope."""
    
    message_id: str                     # Unique message identifier
    message_type: MessageType           # Type of message
    timestamp: datetime                 # Creation timestamp
    correlation_id: str | None = None   # Related message correlation
    in_reply_to: str | None = None      # Reply-to message ID
    payload: dict[str, Any] = {}        # Message payload
    source: str = "unknown"             # Message source
    
    @classmethod
    def create(
        cls,
        message_type: MessageType,
        payload: BaseModel | dict[str, Any],
        source: str = "backend",
        correlation_id: str | None = None,
        in_reply_to: str | None = None,
    ) -> "ProtocolMessage":
        """Create a new protocol message with auto-generated ID."""
```

**Usage:**

```python
# Create a label request message
request = LabelRequest(
    request_id="req-001",
    target_type=LabelTargetType.ENTITY,
    target_id="entity-abc",
    confidence=LabelConfidence.LOW,
    description="What is this object?"
)

message = ProtocolMessage.create(
    message_type=MessageType.LABEL_REQUEST,
    payload=request,
    source="backend"
)

# Serialize for transmission
json_str = message.model_dump_json()
```

---

### EntityUpdate / LocationUpdate

Backend updates about entities and locations for UI display.

```python
class EntityUpdate(BaseModel):
    """Backend update about an entity's state."""
    
    entity_id: str                              # Unique entity ID
    label: str = "unknown"                      # Current best label
    labels: list[str] = []                      # All known labels/aliases
    confidence: float = 0.0                     # Recognition confidence
    visible: bool = True                        # Currently visible
    state: str | None = None                    # Current state
    relative_position: tuple[float, float, float] | None = None


class LocationUpdate(BaseModel):
    """Backend update about current location."""
    
    location_id: str                            # Unique location ID
    label: str = "unknown"                      # Current best label
    labels: list[str] = []                      # All known labels/aliases
    confidence: float = 0.0                     # Recognition confidence
    is_stable: bool = True                      # Location identity stable
    uncertainty_reason: str | None = None       # Why uncertain
```

---

### Handshake / ProtocolError

Connection setup and error reporting.

```python
class Handshake(BaseModel):
    """Connection handshake message."""
    
    protocol_version: str = "1.0.0"             # Semantic version
    role: Literal["sensor", "backend"]          # Which end
    identity: str = "unknown"                   # Human-readable ID
    sent_at: datetime                           # For latency calc
    session_id: str | None = None               # For session resume


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProtocolError(BaseModel):
    """Error report message."""
    
    severity: ErrorSeverity                     # Error severity
    code: str                                   # Error code
    message: str                                # Human-readable message
    related_message_id: str | None = None       # Related message
    details: dict[str, Any] = {}                # Additional details
    recoverable: bool = True                    # Can recover
    suggested_action: str | None = None         # Recovery suggestion
```
