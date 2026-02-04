"""Type definitions for the Sensor Gateway.

Defines the universal messaging types and enums for multi-sensor support.
These types are designed to be sensor-agnostic while capturing the essential
information needed to answer the three fundamental questions:
1. "Where am I?" - Spatial/location context
2. "What's around?" - Entity/object awareness
3. "What's happening/can happen?" - Events and predictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class SensorType(str, Enum):
    """Known sensor types that the gateway can handle.
    
    Extensible - add new sensor types as needed.
    """
    
    # Visual/spatial sensors
    UNITY_WEBSOCKET = "unity_websocket"  # Unity simulation
    CAMERA_RGB = "camera_rgb"            # RGB camera
    CAMERA_DEPTH = "camera_depth"        # Depth camera
    CAMERA_STEREO = "camera_stereo"      # Stereo vision
    LIDAR = "lidar"                       # LiDAR point cloud
    RADAR = "radar"                       # Radar
    
    # Audio sensors
    MICROPHONE = "microphone"            # Audio input
    ULTRASOUND = "ultrasound"            # Ultrasonic sensor
    
    # Position/motion sensors
    GPS = "gps"                          # GPS location
    IMU = "imu"                          # Inertial measurement
    ODOMETRY = "odometry"                # Wheel odometry
    
    # Environmental sensors
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    
    # Generic/unknown
    UNKNOWN = "unknown"
    CUSTOM = "custom"


class SensorCapability(str, Enum):
    """Capabilities that a sensor can provide.
    
    Used to determine which fundamental questions a sensor can help answer.
    """
    
    # "Where am I?" capabilities
    PROVIDES_LOCATION = "provides_location"       # Can give absolute position
    PROVIDES_POSE = "provides_pose"               # Position + orientation
    PROVIDES_ODOMETRY = "provides_odometry"       # Relative motion
    PROVIDES_ROOM_ID = "provides_room_id"         # Room/zone identification
    
    # "What's around?" capabilities
    PROVIDES_ENTITIES = "provides_entities"       # Object detection
    PROVIDES_DEPTH = "provides_depth"             # Distance measurement
    PROVIDES_POINTCLOUD = "provides_pointcloud"   # 3D point cloud
    PROVIDES_AUDIO = "provides_audio"             # Audio signals
    
    # "What's happening?" capabilities
    PROVIDES_EVENTS = "provides_events"           # Discrete events
    PROVIDES_STATE_CHANGES = "provides_state_changes"  # State transitions
    PROVIDES_MOTION = "provides_motion"           # Motion detection
    
    # Meta capabilities
    PROVIDES_CONFIDENCE = "provides_confidence"   # Has reliability metrics
    PROVIDES_TIMESTAMPS = "provides_timestamps"   # Has timing info


class SensorStatus(str, Enum):
    """Status of a sensor connection/stream."""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    TIMEOUT = "timeout"
    CALIBRATING = "calibrating"


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""
    
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Potential issue, but can proceed
    ERROR = "error"         # Invalid data, needs correction
    CRITICAL = "critical"   # Cannot process, requires intervention


@dataclass
class ValidationError:
    """A single validation error or warning.
    
    Contains details about what went wrong and how to fix it.
    """
    
    code: str                                      # Error code (e.g., "MISSING_FIELD")
    message: str                                   # Human-readable description
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field_path: str | None = None                  # Path to problematic field
    expected: Any = None                           # What was expected
    actual: Any = None                             # What was received
    suggestion: str | None = None                  # How to fix it
    
    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.code}: {self.message}"]
        if self.field_path:
            parts.append(f"  Field: {self.field_path}")
        if self.expected is not None:
            parts.append(f"  Expected: {self.expected}")
        if self.actual is not None:
            parts.append(f"  Got: {self.actual}")
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        return "\n".join(parts)


@dataclass
class ValidationResult:
    """Result of validating sensor data.
    
    Contains validation status, any errors/warnings, and corrected data.
    """
    
    is_valid: bool                                 # Whether data passed validation
    errors: list[ValidationError] = field(default_factory=list)
    corrected_data: dict[str, Any] | None = None   # Data after corrections
    raw_data: dict[str, Any] | None = None         # Original data
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(e.severity == ValidationSeverity.WARNING for e in self.errors)
    
    @property
    def has_critical(self) -> bool:
        """Check if there are critical errors."""
        return any(e.severity == ValidationSeverity.CRITICAL for e in self.errors)
    
    @property
    def error_summary(self) -> str:
        """Get a brief summary of errors."""
        if not self.errors:
            return "No errors"
        
        by_severity = {}
        for e in self.errors:
            by_severity.setdefault(e.severity, []).append(e)
        
        parts = []
        for sev in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                    ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            if sev in by_severity:
                parts.append(f"{len(by_severity[sev])} {sev.value}")
        
        return ", ".join(parts)


# =============================================================================
# Universal Sensor Message
# =============================================================================

@dataclass
class LocationContext:
    """Spatial context for answering "Where am I?"
    
    Aggregated from all sensors that provide location information.
    """
    
    # Position (if known)
    position: tuple[float, float, float] | None = None  # (x, y, z)
    position_confidence: float = 0.0
    
    # Orientation (if known)
    rotation: tuple[float, float, float] | None = None  # (roll, pitch, yaw)
    forward_vector: tuple[float, float, float] | None = None
    
    # Room/zone (if known or inferred)
    room_id: str | None = None
    room_label: str | None = None
    room_confidence: float = 0.0
    
    # Source information
    source_sensors: list[str] = field(default_factory=list)
    
    # Relative position support (Phase 2)
    # Raw position is for internal calculations only
    relative_position: Any | None = None  # RelativePosition from schemas.spatial
    qualitative_position: str | None = None  # e.g., "near the kitchen table"


@dataclass
class EntityObservation:
    """A single entity observation for answering "What's around?"
    
    Represents an object detected by any sensor.
    """
    
    # Identity (may be uncertain)
    entity_id: str                                 # Unique ID if known
    label: str = "unknown"                         # Best guess label
    category: str = "unknown"                      # Category (furniture, door, etc.)
    confidence: float = 0.0                        # Recognition confidence
    
    # Spatial information
    position: tuple[float, float, float] | None = None  # Raw position (internal use)
    distance: float | None = None
    visible: bool = True
    
    # Relative position support (Phase 2)
    relative_position: Any | None = None  # RelativePosition from schemas.spatial
    spatial_relation: str | None = None  # Qualitative relation (e.g., "near_left")
    reference_landmark: str | None = None  # ID of reference landmark
    
    # State information
    state: str | None = None
    previous_state: str | None = None
    
    # Source
    source_sensor: str = "unknown"
    
    # Extra attributes
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventObservation:
    """An event observation for answering "What's happening/can happen?"
    
    Represents a discrete event or state change.
    """
    
    event_type: str                                # Type of event
    entity_id: str | None = None                   # Related entity if any
    description: str = ""                          # Human-readable description
    
    # State change details
    old_state: str | None = None
    new_state: str | None = None
    
    # Timing
    timestamp: datetime | None = None
    
    # Confidence and source
    confidence: float = 1.0
    source_sensor: str = "unknown"
    
    # Additional context
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorMessage:
    """Universal sensor message format.
    
    This is the common format that all sensor data is converted to,
    regardless of the original source. It provides structured answers
    to the three fundamental questions.
    """
    
    # Message metadata
    message_id: str                                # Unique message ID
    timestamp: datetime                            # When the data was captured
    sensor_type: SensorType                        # Source sensor type
    sensor_id: str = "default"                     # Specific sensor instance
    
    # Validation status
    validation: ValidationResult | None = None    # How validation went
    
    # The three fundamental questions
    
    # 1. "Where am I?" - Location context
    location: LocationContext | None = None
    
    # 2. "What's around?" - Entity observations
    entities: list[EntityObservation] = field(default_factory=list)
    
    # 3. "What's happening?" - Event observations
    events: list[EventObservation] = field(default_factory=list)
    
    # Raw data preservation (for debugging/replay)
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    processing_time_ms: float = 0.0
    
    def summary(self) -> str:
        """Get a human-readable summary of this message."""
        parts = [
            f"SensorMessage[{self.sensor_type.value}]",
            f"  ID: {self.message_id}",
            f"  Time: {self.timestamp.strftime('%H:%M:%S.%f')[:-3]}",
        ]
        
        if self.location:
            loc = self.location
            if loc.room_label:
                parts.append(f"  📍 Location: {loc.room_label} ({loc.room_confidence:.0%})")
            if loc.position:
                parts.append(f"     Position: ({loc.position[0]:.2f}, {loc.position[1]:.2f}, {loc.position[2]:.2f})")
        
        if self.entities:
            parts.append(f"  👁 Entities: {len(self.entities)}")
            for ent in self.entities[:5]:  # Show first 5
                vis = "👁" if ent.visible else "🔇"
                state = f" [{ent.state}]" if ent.state else ""
                parts.append(f"     {vis} {ent.label} ({ent.category}){state}")
            if len(self.entities) > 5:
                parts.append(f"     ... and {len(self.entities) - 5} more")
        
        if self.events:
            parts.append(f"  ⚡ Events: {len(self.events)}")
            for evt in self.events[:3]:
                parts.append(f"     {evt.event_type}: {evt.description}")
        
        if self.validation and not self.validation.is_valid:
            parts.append(f"  ⚠️ Validation: {self.validation.error_summary}")
        
        return "\n".join(parts)


@dataclass
class SensorRegistration:
    """Registration info for a sensor source.
    
    Used by the gateway to track active sensors and their capabilities.
    """
    
    sensor_id: str
    sensor_type: SensorType
    capabilities: set[SensorCapability]
    status: SensorStatus = SensorStatus.DISCONNECTED
    
    # Connection details
    connection_info: dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    frames_received: int = 0
    errors_count: int = 0
    last_frame_time: datetime | None = None
    
    # Reliability tracking
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
