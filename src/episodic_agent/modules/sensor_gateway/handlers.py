"""Sensor data handlers for converting raw data to SensorMessage format.

Handlers are responsible for transforming validated sensor data into the
universal SensorMessage format that answers the three fundamental questions.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from episodic_agent.modules.sensor_gateway.types import (
    EntityObservation,
    EventObservation,
    LocationContext,
    SensorCapability,
    SensorMessage,
    SensorType,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class SensorHandler(ABC):
    """Base class for sensor data handlers.
    
    Handlers convert validated sensor data into the universal SensorMessage
    format, extracting location, entity, and event information.
    """
    
    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """The sensor type this handler processes."""
        ...
    
    @property
    @abstractmethod
    def capabilities(self) -> set[SensorCapability]:
        """What this sensor can provide."""
        ...
    
    @abstractmethod
    def process(
        self,
        data: dict[str, Any],
        validation: ValidationResult | None = None,
        sensor_id: str = "default",
    ) -> SensorMessage:
        """Process validated sensor data into a SensorMessage.
        
        Args:
            data: Validated (and possibly corrected) sensor data.
            validation: The validation result for this data.
            sensor_id: Identifier for this specific sensor instance.
            
        Returns:
            Universal SensorMessage format.
        """
        ...
    
    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        return f"{self.sensor_type.value}_{uuid.uuid4().hex[:12]}"


class UnitySensorHandler(SensorHandler):
    """Handler for Unity WebSocket sensor data.
    
    Extracts location, entity, and event information from Unity frames.
    """
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.UNITY_WEBSOCKET
    
    @property
    def capabilities(self) -> set[SensorCapability]:
        return {
            # Location capabilities
            SensorCapability.PROVIDES_LOCATION,
            SensorCapability.PROVIDES_POSE,
            SensorCapability.PROVIDES_ROOM_ID,
            # Entity capabilities
            SensorCapability.PROVIDES_ENTITIES,
            SensorCapability.PROVIDES_DEPTH,  # Via distance field
            # Event capabilities
            SensorCapability.PROVIDES_STATE_CHANGES,
            SensorCapability.PROVIDES_EVENTS,
            # Meta
            SensorCapability.PROVIDES_TIMESTAMPS,
        }
    
    def process(
        self,
        data: dict[str, Any],
        validation: ValidationResult | None = None,
        sensor_id: str = "unity",
    ) -> SensorMessage:
        """Process Unity sensor frame into SensorMessage.
        
        Args:
            data: Validated Unity frame data.
            validation: Validation result.
            sensor_id: Sensor identifier.
            
        Returns:
            SensorMessage with Unity data mapped to universal format.
        """
        start_time = time.perf_counter()
        
        # Generate message ID
        frame_id = data.get("frame_id", 0)
        message_id = f"unity_{frame_id}_{uuid.uuid4().hex[:8]}"
        
        # Parse timestamp
        timestamp_val = data.get("timestamp")
        if isinstance(timestamp_val, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp_val)
        elif isinstance(timestamp_val, datetime):
            timestamp = timestamp_val
        else:
            timestamp = datetime.now()
        
        # Extract location context ("Where am I?")
        location = self._extract_location(data, sensor_id)
        
        # Extract entity observations ("What's around?")
        entities = self._extract_entities(data, sensor_id)
        
        # Extract events ("What's happening?")
        events = self._extract_events(data, sensor_id, timestamp)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Wire-level debug logging for diagnosing sensor pipeline issues
        if logger.isEnabledFor(logging.DEBUG):
            pos = location.position if location else None
            fwd = location.forward_vector if location else None
            logger.debug(
                f"[{sensor_id}] frame_id={frame_id} "
                f"pos={pos} fwd={fwd} "
                f"room_id={location.room_id if location else None} "
                f"entities={len(entities)} events={len(events)}"
            )
        
        return SensorMessage(
            message_id=message_id,
            timestamp=timestamp,
            sensor_type=self.sensor_type,
            sensor_id=sensor_id,
            validation=validation,
            location=location,
            entities=entities,
            events=events,
            raw_data=data,
            processing_time_ms=processing_time,
        )
    
    def _extract_location(
        self,
        data: dict[str, Any],
        sensor_id: str,
    ) -> LocationContext | None:
        """Extract location context from Unity frame.
        
        Answers: "Where am I?"
        """
        # Get camera pose for position/orientation
        camera_pose = data.get("camera_pose", data.get("camera", {}))
        
        position = None
        rotation = None
        forward = None
        
        if camera_pose:
            pos = camera_pose.get("position", {})
            if pos:
                position = (
                    pos.get("x", 0.0),
                    pos.get("y", 0.0),
                    pos.get("z", 0.0),
                )
            
            rot = camera_pose.get("rotation", {})
            if rot:
                rotation = (
                    rot.get("x", 0.0),
                    rot.get("y", 0.0),
                    rot.get("z", 0.0),
                )
            
            fwd = camera_pose.get("forward", {})
            if fwd:
                forward = (
                    fwd.get("x", 0.0),
                    fwd.get("y", 0.0),
                    fwd.get("z", 0.0),
                )
        
        # Get room information (optional - may not be available from all sensors)
        # ARCHITECTURAL INVARIANT: Unity sends GUID only, backend learns labels from user
        room_id = data.get("current_room_guid") or data.get("current_room")
        # REMOVED: room_label - backend owns all semantic labels
        
        # Room confidence is high if we have explicit room GUID from Unity
        room_confidence = 0.95 if room_id else 0.0
        
        # Position confidence: high if we have a non-zero position (indicating
        # the sensor is actually providing camera data), lower if all zeros
        # (which may indicate a null/uninitialized camera reference)
        position_is_valid = position and any(abs(v) > 1e-6 for v in position)
        position_confidence = 0.99 if position_is_valid else 0.1
        
        return LocationContext(
            position=position,
            position_confidence=position_confidence,
            rotation=rotation,
            forward_vector=forward,
            room_id=room_id,
            room_label=None,  # Label learned from user, not from Unity
            room_confidence=room_confidence,
            source_sensors=[sensor_id],
        )
    
    def _extract_entities(
        self,
        data: dict[str, Any],
        sensor_id: str,
    ) -> list[EntityObservation]:
        """Extract entity observations from Unity frame.
        
        Answers: "What's around?"
        """
        entities_data = data.get("entities", [])
        observations = []
        
        for entity in entities_data:
            if not isinstance(entity, dict):
                continue
            
            # Extract position
            pos = entity.get("position", {})
            position = None
            if pos and isinstance(pos, dict):
                position = (
                    pos.get("x", 0.0),
                    pos.get("y", 0.0),
                    pos.get("z", 0.0),
                )
            
            # ARCHITECTURAL INVARIANT: Unity sends GUID only, no label/category
            # Backend learns labels from user interaction
            observation = EntityObservation(
                entity_id=entity.get("guid", "unknown"),
                label=None,  # Backend learns labels from user
                category=None,  # Backend learns categories from user
                confidence=1.0 if entity.get("guid") else 0.5,  # Unity GUIDs are reliable
                position=position,
                distance=entity.get("distance"),
                visible=entity.get("is_visible", False),
                state=entity.get("interactable_state"),
                source_sensor=sensor_id,
                attributes={
                    "room_guid": entity.get("room_guid"),
                    "interactable_type": entity.get("interactable_type"),
                    "size": entity.get("size"),
                },
            )
            observations.append(observation)
        
        return observations
    
    def _extract_events(
        self,
        data: dict[str, Any],
        sensor_id: str,
        timestamp: datetime,
    ) -> list[EventObservation]:
        """Extract event observations from Unity frame.
        
        Answers: "What's happening?"
        """
        state_changes = data.get("state_changes", [])
        events = []
        
        for change in state_changes:
            if not isinstance(change, dict):
                continue
            
            event = EventObservation(
                event_type="state_change",
                entity_id=change.get("guid") or change.get("entity_guid"),
                description=f"{change.get('entity_label', 'Entity')} changed from {change.get('old_state', '?')} to {change.get('new_state', '?')}",
                old_state=change.get("old_state"),
                new_state=change.get("new_state"),
                timestamp=timestamp,
                confidence=1.0,  # Unity state changes are reliable
                source_sensor=sensor_id,
                context={
                    "entity_label": change.get("entity_label"),
                    "category": change.get("category"),
                },
            )
            events.append(event)
        
        return events


class GenericSensorHandler(SensorHandler):
    """Generic handler for unknown or custom sensor types.
    
    Attempts to extract whatever information is available.
    """
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.UNKNOWN
    
    @property
    def capabilities(self) -> set[SensorCapability]:
        return {
            SensorCapability.PROVIDES_TIMESTAMPS,
        }
    
    def process(
        self,
        data: dict[str, Any],
        validation: ValidationResult | None = None,
        sensor_id: str = "generic",
    ) -> SensorMessage:
        """Process generic sensor data into SensorMessage.
        
        Extracts whatever standard fields are available.
        """
        start_time = time.perf_counter()
        
        # Try to get timestamp
        timestamp_val = data.get("timestamp", data.get("time"))
        if isinstance(timestamp_val, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp_val)
        elif isinstance(timestamp_val, datetime):
            timestamp = timestamp_val
        else:
            timestamp = datetime.now()
        
        message_id = self._generate_message_id()
        
        # Try to extract location if available
        location = None
        if "position" in data or "location" in data or "pose" in data:
            location = self._try_extract_location(data, sensor_id)
        
        # Try to extract entities if available
        entities = []
        if "entities" in data or "objects" in data or "detections" in data:
            entities = self._try_extract_entities(data, sensor_id)
        
        # Try to extract events if available
        events = []
        if "events" in data or "changes" in data:
            events = self._try_extract_events(data, sensor_id, timestamp)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return SensorMessage(
            message_id=message_id,
            timestamp=timestamp,
            sensor_type=SensorType.UNKNOWN,
            sensor_id=sensor_id,
            validation=validation,
            location=location,
            entities=entities,
            events=events,
            raw_data=data,
            processing_time_ms=processing_time,
        )
    
    def _try_extract_location(
        self,
        data: dict[str, Any],
        sensor_id: str,
    ) -> LocationContext | None:
        """Try to extract location from various field names."""
        position = None
        
        for key in ["position", "location", "pose", "coords"]:
            if key in data:
                pos = data[key]
                if isinstance(pos, dict):
                    position = (
                        pos.get("x", pos.get("lng", 0.0)),
                        pos.get("y", pos.get("alt", 0.0)),
                        pos.get("z", pos.get("lat", 0.0)),
                    )
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    position = (float(pos[0]), float(pos[1]), float(pos[2]))
                break
        
        if position:
            return LocationContext(
                position=position,
                position_confidence=0.5,  # Unknown reliability
                source_sensors=[sensor_id],
            )
        
        return None
    
    def _try_extract_entities(
        self,
        data: dict[str, Any],
        sensor_id: str,
    ) -> list[EntityObservation]:
        """Try to extract entities from various field names."""
        entities = []
        
        for key in ["entities", "objects", "detections", "items"]:
            if key in data and isinstance(data[key], list):
                for i, item in enumerate(data[key]):
                    if isinstance(item, dict):
                        entities.append(EntityObservation(
                            entity_id=item.get("id", item.get("guid", f"entity_{i}")),
                            label=item.get("label", item.get("name", "unknown")),
                            category=item.get("category", item.get("type", "unknown")),
                            confidence=item.get("confidence", 0.5),
                            source_sensor=sensor_id,
                        ))
                break
        
        return entities
    
    def _try_extract_events(
        self,
        data: dict[str, Any],
        sensor_id: str,
        timestamp: datetime,
    ) -> list[EventObservation]:
        """Try to extract events from various field names."""
        events = []
        
        for key in ["events", "changes", "updates"]:
            if key in data and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        events.append(EventObservation(
                            event_type=item.get("type", item.get("event", "unknown")),
                            description=item.get("description", item.get("message", "")),
                            timestamp=timestamp,
                            confidence=0.5,
                            source_sensor=sensor_id,
                        ))
                break
        
        return events


class LidarSensorHandler(SensorHandler):
    """Handler for LiDAR sensor data."""
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.LIDAR
    
    @property
    def capabilities(self) -> set[SensorCapability]:
        return {
            SensorCapability.PROVIDES_POINTCLOUD,
            SensorCapability.PROVIDES_DEPTH,
            SensorCapability.PROVIDES_TIMESTAMPS,
        }
    
    def process(
        self,
        data: dict[str, Any],
        validation: ValidationResult | None = None,
        sensor_id: str = "lidar",
    ) -> SensorMessage:
        """Process LiDAR point cloud data."""
        start_time = time.perf_counter()
        
        timestamp_val = data.get("timestamp")
        if isinstance(timestamp_val, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp_val)
        else:
            timestamp = datetime.now()
        
        message_id = self._generate_message_id()
        
        # Extract point cloud as raw data (too large to convert to entities)
        points_key = "points" if "points" in data else "point_cloud"
        points = data.get(points_key, [])
        
        # Could detect obstacles/objects from point cloud clustering
        # For now, just preserve the point cloud in raw_data
        entities = []
        
        # Extract location from sensor pose if available
        location = None
        if "pose" in data or "sensor_pose" in data:
            pose = data.get("pose", data.get("sensor_pose", {}))
            if isinstance(pose, dict):
                pos = pose.get("position", {})
                location = LocationContext(
                    position=(pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)),
                    position_confidence=0.8,
                    source_sensors=[sensor_id],
                )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return SensorMessage(
            message_id=message_id,
            timestamp=timestamp,
            sensor_type=self.sensor_type,
            sensor_id=sensor_id,
            validation=validation,
            location=location,
            entities=entities,
            events=[],
            raw_data={"points": points, "point_count": len(points)},
            processing_time_ms=processing_time,
        )


class AudioSensorHandler(SensorHandler):
    """Handler for audio/microphone sensor data."""
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.MICROPHONE
    
    @property
    def capabilities(self) -> set[SensorCapability]:
        return {
            SensorCapability.PROVIDES_AUDIO,
            SensorCapability.PROVIDES_EVENTS,  # Sound events
            SensorCapability.PROVIDES_TIMESTAMPS,
        }
    
    def process(
        self,
        data: dict[str, Any],
        validation: ValidationResult | None = None,
        sensor_id: str = "microphone",
    ) -> SensorMessage:
        """Process audio sensor data."""
        start_time = time.perf_counter()
        
        timestamp_val = data.get("timestamp")
        if isinstance(timestamp_val, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp_val)
        else:
            timestamp = datetime.now()
        
        message_id = self._generate_message_id()
        
        # Extract events from audio (e.g., speech, sound detection)
        events = []
        
        # If there's a transcript, that's a speech event
        if "transcript" in data and data["transcript"]:
            events.append(EventObservation(
                event_type="speech",
                description=data["transcript"],
                timestamp=timestamp,
                confidence=data.get("confidence", 0.8),
                source_sensor=sensor_id,
                context={
                    "speaker": data.get("speaker"),
                    "language": data.get("language"),
                },
            ))
        
        # Sound detection events
        if "sounds" in data:
            for sound in data["sounds"]:
                if isinstance(sound, dict):
                    events.append(EventObservation(
                        event_type="sound_detected",
                        description=sound.get("label", "unknown sound"),
                        timestamp=timestamp,
                        confidence=sound.get("confidence", 0.5),
                        source_sensor=sensor_id,
                        context=sound,
                    ))
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Don't include raw audio samples in raw_data (too large)
        # Just metadata
        raw_data = {
            "sample_rate": data.get("sample_rate"),
            "duration_ms": data.get("duration_ms", data.get("duration")),
            "channels": data.get("channels"),
        }
        
        return SensorMessage(
            message_id=message_id,
            timestamp=timestamp,
            sensor_type=self.sensor_type,
            sensor_id=sensor_id,
            validation=validation,
            location=None,  # Audio doesn't provide location
            entities=[],
            events=events,
            raw_data=raw_data,
            processing_time_ms=processing_time,
        )


# Registry of handlers by sensor type
HANDLERS: dict[SensorType, type[SensorHandler]] = {
    SensorType.UNITY_WEBSOCKET: UnitySensorHandler,
    SensorType.LIDAR: LidarSensorHandler,
    SensorType.MICROPHONE: AudioSensorHandler,
    SensorType.ULTRASOUND: AudioSensorHandler,  # Similar handling
    SensorType.UNKNOWN: GenericSensorHandler,
    SensorType.CUSTOM: GenericSensorHandler,
}


def get_handler(sensor_type: SensorType) -> SensorHandler:
    """Get the appropriate handler for a sensor type.
    
    Args:
        sensor_type: The type of sensor to handle.
        
    Returns:
        A handler instance for that sensor type.
    """
    handler_class = HANDLERS.get(sensor_type, GenericSensorHandler)
    return handler_class()
