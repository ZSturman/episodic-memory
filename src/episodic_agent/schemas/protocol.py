"""Wire protocol message schemas for sensor-backend communication.

ARCHITECTURAL INVARIANT: Protocol is sensor-agnostic and portable.
No Unity-specific shortcuts. All messages work with physical sensors.

This module defines:
- MessageType: All valid protocol message types
- CapabilitiesReport: Sensor capabilities announcement
- StreamControl: Backend commands to sensor
- LabelRequest/LabelResponse: User labeling flow
- FrameAck: Backend acknowledgment of frames
- ProtocolMessage: Universal message envelope
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class MessageType(str, Enum):
    """All valid protocol message types.
    
    ARCHITECTURAL INVARIANT: Messages are structural, not semantic.
    No message type implies domain-specific knowledge.
    """
    
    # Sensor → Backend
    SENSOR_FRAME = "sensor_frame"            # Raw sensor data
    CAPABILITIES_REPORT = "capabilities_report"  # What sensor can do
    VISUAL_SUMMARY = "visual_summary"        # 4×4 grid summary
    VISUAL_FOCUS = "visual_focus"            # High-res crop response
    LABEL_RESPONSE = "label_response"        # User's label input
    ERROR = "error"                          # Sensor error report
    
    # Backend → Sensor
    FRAME_ACK = "frame_ack"                  # Acknowledgment
    STREAM_CONTROL = "stream_control"        # Control commands
    FOCUS_REQUEST = "focus_request"          # Request high-res region
    LABEL_REQUEST = "label_request"          # Request label from user
    ENTITY_UPDATE = "entity_update"          # Backend entity state
    LOCATION_UPDATE = "location_update"      # Backend location state
    
    # Bidirectional
    HEARTBEAT = "heartbeat"                  # Keep-alive
    HANDSHAKE = "handshake"                  # Connection setup


# =============================================================================
# SENSOR CAPABILITIES
# =============================================================================

class SensorCapability(str, Enum):
    """Capabilities a sensor may report."""
    
    # Visual capabilities
    RGB_CAMERA = "rgb_camera"                # Can provide RGB images
    DEPTH_CAMERA = "depth_camera"            # Can provide depth data
    STEREO_CAMERA = "stereo_camera"          # Has stereo vision
    ZOOM = "zoom"                            # Can zoom in/out
    FOCUS = "focus"                          # Can focus on regions
    PAN_TILT = "pan_tilt"                    # Can pan/tilt
    
    # Spatial capabilities
    ODOMETRY = "odometry"                    # Reports movement
    IMU = "imu"                              # Has inertial measurement
    GPS = "gps"                              # Has global positioning
    LIDAR = "lidar"                          # Has LIDAR
    
    # Object detection
    BOUNDING_BOXES = "bounding_boxes"        # Can detect bounding boxes
    SEGMENTATION = "segmentation"            # Can do segmentation
    TRACKING = "tracking"                    # Can track objects over time
    
    # Audio
    MICROPHONE = "microphone"                # Has audio input
    SPEECH_TO_TEXT = "speech_to_text"        # Can transcribe speech
    
    # Compute
    EDGE_COMPUTE = "edge_compute"            # Has local compute
    FEATURE_EXTRACTION = "feature_extraction"  # Can extract features locally


class CapabilitiesReport(BaseModel):
    """Sensor capabilities announcement.
    
    Sent by sensor on connection to inform backend what it can do.
    Backend uses this to adapt its requests appropriately.
    
    ARCHITECTURAL INVARIANT: Backend must not assume capabilities.
    All feature requests must be validated against this report.
    """
    
    # Sensor identification
    sensor_id: str = Field(..., description="Unique sensor identifier")
    sensor_type: str = Field(..., description="Type of sensor (camera, robot, etc.)")
    sensor_version: str = Field(default="1.0.0", description="Sensor protocol version")
    
    # Capabilities
    capabilities: list[SensorCapability] = Field(
        default_factory=list,
        description="List of supported capabilities",
    )
    
    # Resolution constraints
    max_resolution: tuple[int, int] | None = Field(
        default=None,
        description="Maximum supported resolution (width, height)",
    )
    min_resolution: tuple[int, int] | None = Field(
        default=None,
        description="Minimum supported resolution (width, height)",
    )
    
    # Frame rate constraints
    max_fps: float = Field(default=30.0, description="Maximum frames per second")
    min_fps: float = Field(default=1.0, description="Minimum frames per second")
    
    # Compute availability
    compute_available: bool = Field(
        default=False,
        description="Whether edge compute is available",
    )
    compute_tflops: float | None = Field(
        default=None,
        description="Available compute in TFLOPS if known",
    )
    
    # Memory constraints
    buffer_size_mb: int = Field(
        default=50,
        description="Available buffer memory in MB",
    )
    
    # Protocol features
    supports_visual_channel: bool = Field(
        default=False,
        description="Whether visual summary channel is supported",
    )
    supports_focus_requests: bool = Field(
        default=False,
        description="Whether focus/crop requests are supported",
    )
    
    # Additional metadata
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Sensor-specific additional info",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# STREAM CONTROL
# =============================================================================

class StreamControlCommand(str, Enum):
    """Commands backend can send to control sensor stream."""
    
    START = "start"                          # Start streaming
    STOP = "stop"                            # Stop streaming
    PAUSE = "pause"                          # Pause streaming
    RESUME = "resume"                        # Resume streaming
    SET_RESOLUTION = "set_resolution"        # Change resolution
    SET_FPS = "set_fps"                      # Change frame rate
    SET_CROP = "set_crop"                    # Set crop region
    CLEAR_CROP = "clear_crop"                # Clear crop region
    ENABLE_SUMMARY = "enable_summary"        # Enable 4×4 summary mode
    DISABLE_SUMMARY = "disable_summary"      # Disable summary mode
    REQUEST_KEYFRAME = "request_keyframe"    # Request full keyframe


class StreamControl(BaseModel):
    """Backend command to control sensor stream.
    
    ARCHITECTURAL INVARIANT: Backend controls the stream, not sensor.
    Sensor must obey or report incapability.
    """
    
    command: StreamControlCommand = Field(..., description="Control command")
    
    # Parameters (used by some commands)
    resolution: tuple[int, int] | None = Field(
        default=None,
        description="Target resolution for SET_RESOLUTION",
    )
    fps: float | None = Field(
        default=None,
        description="Target FPS for SET_FPS",
    )
    crop_region: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Crop region (x, y, width, height) for SET_CROP",
    )
    
    # Duration for temporary commands
    duration_seconds: float | None = Field(
        default=None,
        description="Duration for temporary changes",
    )
    
    # Request ID for response correlation
    request_id: str | None = Field(
        default=None,
        description="ID to correlate with response",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# LABEL REQUEST/RESPONSE
# =============================================================================

class LabelTargetType(str, Enum):
    """What kind of thing needs a label."""
    
    ENTITY = "entity"                        # An entity/object
    LOCATION = "location"                    # A location/room
    EVENT = "event"                          # An event type
    RELATION = "relation"                    # A relationship


class LabelConfidence(str, Enum):
    """Confidence level triggering the request."""
    
    LOW = "low"                              # No idea, need user input
    MEDIUM = "medium"                        # Have a guess, need confirmation
    HIGH = "high"                            # Confident, informing user


class LabelRequest(BaseModel):
    """Backend request for user to provide a label.
    
    ARCHITECTURAL INVARIANT: Labels come from users, not assumptions.
    This is the ONLY way labels enter the system.
    """
    
    # Request identification
    request_id: str = Field(..., description="Unique request identifier")
    
    # What needs labeling
    target_type: LabelTargetType = Field(..., description="Type of thing to label")
    target_id: str = Field(..., description="ID of the target (entity, location, etc.)")
    
    # Context for the user
    confidence: LabelConfidence = Field(..., description="Why we're asking")
    current_label: str | None = Field(
        default=None,
        description="Current best guess (for confirmation)",
    )
    alternative_labels: list[str] = Field(
        default_factory=list,
        description="Other possible labels",
    )
    
    # Visual context (if available)
    thumbnail_base64: str | None = Field(
        default=None,
        description="Base64-encoded thumbnail image",
    )
    bounding_box: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Bounding box in visual context",
    )
    
    # Description for user
    description: str = Field(
        default="",
        description="Human-readable description of what needs labeling",
    )
    
    # Timeout
    timeout_seconds: float = Field(
        default=30.0,
        description="How long to wait for response",
    )
    
    model_config = {"frozen": False}


class LabelResponseType(str, Enum):
    """How user responded to label request."""
    
    PROVIDED = "provided"                    # User provided a label
    CONFIRMED = "confirmed"                  # User confirmed suggestion
    REJECTED = "rejected"                    # User rejected all suggestions
    TIMEOUT = "timeout"                      # No response in time
    SKIPPED = "skipped"                      # User explicitly skipped


class LabelResponse(BaseModel):
    """User's response to a label request.
    
    ARCHITECTURAL INVARIANT: This is user input, not sensor inference.
    The sensor (Unity) does NOT interpret or modify this.
    """
    
    # Response identification
    request_id: str = Field(..., description="ID of the original request")
    
    # Response
    response_type: LabelResponseType = Field(..., description="How user responded")
    label: str | None = Field(
        default=None,
        description="The label provided or confirmed",
    )
    
    # Additional user input
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="User's confidence in their label",
    )
    notes: str | None = Field(
        default=None,
        description="Optional user notes",
    )
    
    # Timing
    response_time_ms: float | None = Field(
        default=None,
        description="How long user took to respond",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# FRAME ACKNOWLEDGMENT
# =============================================================================

class FrameAck(BaseModel):
    """Backend acknowledgment of received frame.
    
    Allows sensor to track what backend has processed.
    """
    
    frame_id: int = Field(..., description="ID of acknowledged frame")
    received_at: datetime = Field(
        default_factory=datetime.now,
        description="When frame was received",
    )
    
    # Processing status
    processed: bool = Field(
        default=True,
        description="Whether frame was fully processed",
    )
    processing_time_ms: float | None = Field(
        default=None,
        description="Processing time in milliseconds",
    )
    
    # Backend state updates (optional)
    entities_detected: int = Field(
        default=0,
        description="Number of entities detected",
    )
    events_detected: int = Field(
        default=0,
        description="Number of events detected",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# ENTITY/LOCATION UPDATES
# =============================================================================

class EntityUpdate(BaseModel):
    """Backend update about an entity's state.
    
    Sent to sensor/UI to display current entity information.
    This is what the backend has learned, not sensor inference.
    """
    
    entity_id: str = Field(..., description="Unique entity identifier")
    
    # Labels (from user learning)
    label: str = Field(
        default="unknown",
        description="Current best label",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="All known labels/aliases",
    )
    
    # Confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Recognition confidence",
    )
    
    # State
    visible: bool = Field(default=True, description="Currently visible")
    state: str | None = Field(default=None, description="Current state if any")
    
    # Position (relative, not absolute)
    relative_position: tuple[float, float, float] | None = Field(
        default=None,
        description="Position relative to agent",
    )
    
    model_config = {"frozen": False}


class LocationUpdate(BaseModel):
    """Backend update about current location.
    
    Sent to sensor/UI to display where agent believes it is.
    """
    
    location_id: str = Field(..., description="Unique location identifier")
    
    # Labels
    label: str = Field(
        default="unknown",
        description="Current best label",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="All known labels/aliases",
    )
    
    # Confidence
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Location recognition confidence",
    )
    
    # Stability state
    is_stable: bool = Field(
        default=True,
        description="Whether location identity is stable",
    )
    uncertainty_reason: str | None = Field(
        default=None,
        description="Why location is uncertain if not stable",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# HANDSHAKE
# =============================================================================

class Handshake(BaseModel):
    """Connection handshake message.
    
    Exchanged on connection to establish protocol version
    and validate compatibility.
    """
    
    # Protocol version
    protocol_version: str = Field(
        default="1.0.0",
        description="Semantic version of protocol",
    )
    
    # Role
    role: Literal["sensor", "backend"] = Field(
        ...,
        description="Which end is sending this",
    )
    
    # Identity
    identity: str = Field(
        default="unknown",
        description="Human-readable identity",
    )
    
    # Timestamp for latency calculation
    sent_at: datetime = Field(
        default_factory=datetime.now,
        description="When handshake was sent",
    )
    
    # Session
    session_id: str | None = Field(
        default=None,
        description="Session ID if resuming",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# ERROR REPORTING
# =============================================================================

class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    WARNING = "warning"                      # Non-fatal issue
    ERROR = "error"                          # Recoverable error
    CRITICAL = "critical"                    # Fatal error


class ProtocolError(BaseModel):
    """Error report message."""
    
    severity: ErrorSeverity = Field(..., description="Error severity")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    
    # Context
    related_message_id: str | None = Field(
        default=None,
        description="ID of message that caused error",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error details",
    )
    
    # Recovery
    recoverable: bool = Field(
        default=True,
        description="Whether error is recoverable",
    )
    suggested_action: str | None = Field(
        default=None,
        description="Suggested recovery action",
    )
    
    model_config = {"frozen": False}


# =============================================================================
# UNIVERSAL MESSAGE ENVELOPE
# =============================================================================

class ProtocolMessage(BaseModel):
    """Universal protocol message envelope.
    
    All messages are wrapped in this envelope for consistent
    handling, routing, and logging.
    
    ARCHITECTURAL INVARIANT: Every message has a type and timestamp.
    All communication is traceable.
    """
    
    # Message identification
    message_id: str = Field(..., description="Unique message identifier")
    message_type: MessageType = Field(..., description="Type of message")
    
    # Timing
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When message was created",
    )
    
    # Correlation
    correlation_id: str | None = Field(
        default=None,
        description="ID linking to related messages",
    )
    in_reply_to: str | None = Field(
        default=None,
        description="ID of message this replies to",
    )
    
    # Payload (one of the message types above)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload",
    )
    
    # Metadata
    source: str = Field(
        default="unknown",
        description="Source of message (sensor ID, 'backend', etc.)",
    )
    
    model_config = {"frozen": False}
    
    @classmethod
    def create(
        cls,
        message_type: MessageType,
        payload: BaseModel | dict[str, Any],
        source: str = "backend",
        correlation_id: str | None = None,
        in_reply_to: str | None = None,
    ) -> "ProtocolMessage":
        """Create a new protocol message.
        
        Args:
            message_type: Type of message
            payload: Message payload (will be converted to dict)
            source: Message source
            correlation_id: Optional correlation ID
            in_reply_to: Optional reply-to ID
            
        Returns:
            New ProtocolMessage instance
        """
        import uuid
        
        payload_dict = (
            payload.model_dump() if isinstance(payload, BaseModel)
            else payload
        )
        
        return cls(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            payload=payload_dict,
            source=source,
            correlation_id=correlation_id,
            in_reply_to=in_reply_to,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Message types
    "MessageType",
    # Capabilities
    "SensorCapability",
    "CapabilitiesReport",
    # Stream control
    "StreamControlCommand",
    "StreamControl",
    # Label flow
    "LabelTargetType",
    "LabelConfidence",
    "LabelRequest",
    "LabelResponseType",
    "LabelResponse",
    # Acknowledgment
    "FrameAck",
    # Updates
    "EntityUpdate",
    "LocationUpdate",
    # Handshake
    "Handshake",
    # Errors
    "ErrorSeverity",
    "ProtocolError",
    # Envelope
    "ProtocolMessage",
]
