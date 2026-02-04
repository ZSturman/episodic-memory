"""Sensor Gateway - Universal sensor data processing and validation.

The SensorGateway is the central hub for processing sensor data from multiple
sources. It provides:

1. Automatic sensor type detection
2. Data validation and error handling  
3. Conversion to universal SensorMessage format
4. User notification for critical issues
5. Logging for debugging

This module is designed to be modular and portable - new sensor types can be
added by implementing SensorValidator and SensorHandler for that type.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

from episodic_agent.modules.sensor_gateway.handlers import (
    SensorHandler,
    get_handler,
    HANDLERS,
)
from episodic_agent.modules.sensor_gateway.types import (
    SensorCapability,
    SensorMessage,
    SensorRegistration,
    SensorStatus,
    SensorType,
    ValidationError,
    ValidationResult,
    ValidationSeverity,
)
from episodic_agent.modules.sensor_gateway.validators import (
    SensorValidator,
    get_validator,
    VALIDATORS,
)

logger = logging.getLogger(__name__)


class SensorGateway:
    """Central gateway for processing sensor data from multiple sources.
    
    The gateway handles:
    - Sensor registration and tracking
    - Automatic sensor type detection
    - Data validation with error recovery
    - Conversion to universal message format
    - User notifications for issues requiring attention
    - Comprehensive logging for debugging
    
    Usage:
        gateway = SensorGateway()
        
        # Register a sensor
        gateway.register_sensor("unity_main", SensorType.UNITY_WEBSOCKET)
        
        # Process incoming data
        message = gateway.process(raw_data, sensor_id="unity_main")
        
        # Check for issues
        if message.validation and not message.validation.is_valid:
            print(f"Validation issues: {message.validation.error_summary}")
    """
    
    def __init__(
        self,
        on_user_notification: Callable[[str, str], None] | None = None,
        log_raw_data: bool = False,
        log_level: str = "INFO",
        max_error_history: int = 100,
    ) -> None:
        """Initialize the sensor gateway.
        
        Args:
            on_user_notification: Callback for user notifications (severity, message).
            log_raw_data: Whether to log full raw data (verbose).
            log_level: Logging level for gateway messages.
            max_error_history: Max errors to keep in history per sensor.
        """
        self._on_user_notification = on_user_notification
        self._log_raw_data = log_raw_data
        self._max_error_history = max_error_history
        
        # Sensor registrations
        self._sensors: dict[str, SensorRegistration] = {}
        
        # Validators and handlers (can be customized)
        self._validators: dict[SensorType, SensorValidator] = {}
        self._handlers: dict[SensorType, SensorHandler] = {}
        
        # Error tracking
        self._error_history: dict[str, list[ValidationError]] = defaultdict(list)
        self._total_messages = 0
        self._total_errors = 0
        
        # Stats
        self._start_time = datetime.now()
        self._last_message_time: datetime | None = None
        
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    # =========================================================================
    # Sensor Registration
    # =========================================================================
    
    def register_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        capabilities: set[SensorCapability] | None = None,
        connection_info: dict[str, Any] | None = None,
    ) -> None:
        """Register a sensor with the gateway.
        
        Args:
            sensor_id: Unique identifier for this sensor instance.
            sensor_type: Type of sensor (determines validator/handler).
            capabilities: What this sensor provides (auto-detected if not given).
            connection_info: Connection details (URL, port, etc.).
        """
        # Get capabilities from handler if not specified
        if capabilities is None:
            handler = self._get_handler(sensor_type)
            capabilities = handler.capabilities
        
        registration = SensorRegistration(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            capabilities=capabilities,
            status=SensorStatus.DISCONNECTED,
            connection_info=connection_info or {},
        )
        
        self._sensors[sensor_id] = registration
        logger.info(f"Registered sensor: {sensor_id} ({sensor_type.value})")
        logger.debug(f"  Capabilities: {[c.value for c in capabilities]}")
    
    def update_sensor_status(
        self,
        sensor_id: str,
        status: SensorStatus,
    ) -> None:
        """Update the status of a registered sensor.
        
        Args:
            sensor_id: Sensor to update.
            status: New status.
        """
        if sensor_id in self._sensors:
            old_status = self._sensors[sensor_id].status
            self._sensors[sensor_id].status = status
            
            if old_status != status:
                logger.info(f"Sensor {sensor_id} status: {old_status.value} -> {status.value}")
    
    def get_sensor(self, sensor_id: str) -> SensorRegistration | None:
        """Get registration info for a sensor.
        
        Args:
            sensor_id: Sensor to look up.
            
        Returns:
            SensorRegistration or None if not found.
        """
        return self._sensors.get(sensor_id)
    
    def list_sensors(self) -> list[SensorRegistration]:
        """List all registered sensors.
        
        Returns:
            List of sensor registrations.
        """
        return list(self._sensors.values())
    
    # =========================================================================
    # Data Processing
    # =========================================================================
    
    def process(
        self,
        raw_data: dict[str, Any] | str,
        sensor_id: str = "default",
        sensor_type: SensorType | None = None,
    ) -> SensorMessage:
        """Process raw sensor data into a universal SensorMessage.
        
        This is the main entry point for sensor data. It:
        1. Parses JSON if needed
        2. Detects sensor type if not specified
        3. Validates the data
        4. Converts to universal format
        5. Logs and notifies on issues
        
        Args:
            raw_data: Raw sensor data (dict or JSON string).
            sensor_id: Identifier for the sensor source.
            sensor_type: Type of sensor (auto-detected if not given).
            
        Returns:
            SensorMessage in universal format.
        """
        self._total_messages += 1
        start_time = time.perf_counter()
        
        # Parse JSON if needed
        if isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError as e:
                return self._handle_parse_error(raw_data, sensor_id, e)
        else:
            data = raw_data
        
        # Log raw data if enabled
        if self._log_raw_data:
            self._log_raw("RECEIVED", sensor_id, data)
        
        # Auto-detect sensor type if not provided
        if sensor_type is None:
            if sensor_id in self._sensors:
                sensor_type = self._sensors[sensor_id].sensor_type
            else:
                sensor_type = self._detect_sensor_type(data)
                logger.debug(f"Auto-detected sensor type: {sensor_type.value}")
        
        # Validate the data
        validator = self._get_validator(sensor_type)
        validation = validator.validate(data)
        
        # Log validation issues
        if validation.errors:
            self._log_validation_errors(sensor_id, validation)
        
        # Update sensor stats
        if sensor_id in self._sensors:
            self._sensors[sensor_id].frames_received += 1
            self._sensors[sensor_id].last_frame_time = datetime.now()
            if not validation.is_valid:
                self._sensors[sensor_id].errors_count += 1
        
        # Use corrected data if available
        processed_data = validation.corrected_data or data
        
        # Convert to universal format
        handler = self._get_handler(sensor_type)
        message = handler.process(
            processed_data,
            validation=validation,
            sensor_id=sensor_id,
        )
        
        # Calculate total processing time
        total_time = (time.perf_counter() - start_time) * 1000
        message.processing_time_ms = total_time
        
        # Log processed message summary
        self._log_processed(sensor_id, message, validation)
        
        # Notify user of critical issues
        if validation.has_critical:
            self._notify_user(
                "CRITICAL",
                f"Sensor {sensor_id} has critical validation errors",
            )
        
        self._last_message_time = datetime.now()
        
        return message
    
    def process_batch(
        self,
        items: list[tuple[dict[str, Any], str]],
    ) -> list[SensorMessage]:
        """Process multiple sensor data items.
        
        Args:
            items: List of (data, sensor_id) tuples.
            
        Returns:
            List of SensorMessages.
        """
        return [self.process(data, sensor_id) for data, sensor_id in items]
    
    # =========================================================================
    # Type Detection
    # =========================================================================
    
    def _detect_sensor_type(self, data: dict[str, Any]) -> SensorType:
        """Auto-detect sensor type from data fields.
        
        Args:
            data: Raw sensor data.
            
        Returns:
            Detected SensorType.
        """
        # Unity detection
        if "protocol_version" in data and "entities" in data:
            return SensorType.UNITY_WEBSOCKET
        
        # LiDAR detection
        if "points" in data or "point_cloud" in data:
            return SensorType.LIDAR
        
        # Audio detection
        if any(k in data for k in ["samples", "audio", "waveform", "transcript"]):
            return SensorType.MICROPHONE
        
        # GPS detection
        if "latitude" in data and "longitude" in data:
            return SensorType.GPS
        
        # IMU detection
        if "accelerometer" in data or "gyroscope" in data:
            return SensorType.IMU
        
        # Camera detection
        if "image" in data or "rgb" in data:
            return SensorType.CAMERA_RGB
        
        if "depth" in data or "depth_image" in data:
            return SensorType.CAMERA_DEPTH
        
        return SensorType.UNKNOWN
    
    # =========================================================================
    # Validators and Handlers
    # =========================================================================
    
    def _get_validator(self, sensor_type: SensorType) -> SensorValidator:
        """Get or create a validator for a sensor type."""
        if sensor_type not in self._validators:
            self._validators[sensor_type] = get_validator(sensor_type)
        return self._validators[sensor_type]
    
    def _get_handler(self, sensor_type: SensorType) -> SensorHandler:
        """Get or create a handler for a sensor type."""
        if sensor_type not in self._handlers:
            self._handlers[sensor_type] = get_handler(sensor_type)
        return self._handlers[sensor_type]
    
    def set_validator(
        self,
        sensor_type: SensorType,
        validator: SensorValidator,
    ) -> None:
        """Set a custom validator for a sensor type.
        
        Args:
            sensor_type: Type to set validator for.
            validator: Custom validator instance.
        """
        self._validators[sensor_type] = validator
    
    def set_handler(
        self,
        sensor_type: SensorType,
        handler: SensorHandler,
    ) -> None:
        """Set a custom handler for a sensor type.
        
        Args:
            sensor_type: Type to set handler for.
            handler: Custom handler instance.
        """
        self._handlers[sensor_type] = handler
    
    # =========================================================================
    # Error Handling
    # =========================================================================
    
    def _handle_parse_error(
        self,
        raw_data: str,
        sensor_id: str,
        error: json.JSONDecodeError,
    ) -> SensorMessage:
        """Handle JSON parsing errors gracefully.
        
        Args:
            raw_data: The raw string that failed to parse.
            sensor_id: Source sensor.
            error: The parse error.
            
        Returns:
            SensorMessage with error information.
        """
        self._total_errors += 1
        
        validation_error = ValidationError(
            code="JSON_PARSE_ERROR",
            message=f"Failed to parse JSON: {error.msg}",
            severity=ValidationSeverity.CRITICAL,
            suggestion="Check that sensor is sending valid JSON",
            actual=raw_data[:200] if len(raw_data) > 200 else raw_data,
        )
        
        validation = ValidationResult(
            is_valid=False,
            errors=[validation_error],
            raw_data={"raw_string": raw_data[:1000]},
        )
        
        logger.error(f"[{sensor_id}] JSON parse error: {error.msg}")
        logger.debug(f"[{sensor_id}] Raw data: {raw_data[:500]}")
        
        self._notify_user(
            "ERROR",
            f"Sensor {sensor_id} sent invalid JSON",
        )
        
        # Return an empty message with error info
        return SensorMessage(
            message_id=f"error_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            sensor_type=SensorType.UNKNOWN,
            sensor_id=sensor_id,
            validation=validation,
        )
    
    def _log_validation_errors(
        self,
        sensor_id: str,
        validation: ValidationResult,
    ) -> None:
        """Log validation errors and track history.
        
        Args:
            sensor_id: Source sensor.
            validation: Validation result with errors.
        """
        for error in validation.errors:
            # Log based on severity
            if error.severity == ValidationSeverity.CRITICAL:
                logger.error(f"[{sensor_id}] CRITICAL: {error.code} - {error.message}")
            elif error.severity == ValidationSeverity.ERROR:
                logger.warning(f"[{sensor_id}] ERROR: {error.code} - {error.message}")
            elif error.severity == ValidationSeverity.WARNING:
                logger.warning(f"[{sensor_id}] WARNING: {error.code} - {error.message}")
            else:
                logger.debug(f"[{sensor_id}] INFO: {error.code} - {error.message}")
            
            # Track error history
            self._error_history[sensor_id].append(error)
            if len(self._error_history[sensor_id]) > self._max_error_history:
                self._error_history[sensor_id].pop(0)
            
            if error.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR):
                self._total_errors += 1
    
    # =========================================================================
    # Logging
    # =========================================================================
    
    def _log_raw(
        self,
        direction: str,
        sensor_id: str,
        data: dict[str, Any],
    ) -> None:
        """Log raw sensor data for debugging.
        
        Args:
            direction: "RECEIVED" or "SENT".
            sensor_id: Source/destination sensor.
            data: The data to log.
        """
        # Truncate large fields
        log_data = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 10:
                log_data[key] = f"[{len(value)} items]"
            elif isinstance(value, str) and len(value) > 500:
                log_data[key] = f"{value[:500]}... ({len(value)} chars)"
            else:
                log_data[key] = value
        
        logger.debug(f"[{sensor_id}] {direction}: {json.dumps(log_data, default=str, indent=2)}")
    
    def _log_processed(
        self,
        sensor_id: str,
        message: SensorMessage,
        validation: ValidationResult,
    ) -> None:
        """Log processed message summary.
        
        Args:
            sensor_id: Source sensor.
            message: Processed message.
            validation: Validation result.
        """
        status = "✓" if validation.is_valid else "✗"
        
        parts = [
            f"[{sensor_id}] {status}",
            f"type={message.sensor_type.value}",
        ]
        
        if message.location:
            loc = message.location
            if loc.room_label:
                parts.append(f"room={loc.room_label}")
            if loc.position:
                parts.append(f"pos=({loc.position[0]:.1f},{loc.position[1]:.1f},{loc.position[2]:.1f})")
        
        if message.entities:
            parts.append(f"entities={len(message.entities)}")
        
        if message.events:
            parts.append(f"events={len(message.events)}")
        
        parts.append(f"time={message.processing_time_ms:.1f}ms")
        
        if not validation.is_valid:
            parts.append(f"errors={validation.error_summary}")
        
        logger.info(" | ".join(parts))
    
    # =========================================================================
    # User Notification
    # =========================================================================
    
    def _notify_user(self, severity: str, message: str) -> None:
        """Notify user of an issue requiring attention.
        
        Args:
            severity: "INFO", "WARNING", "ERROR", or "CRITICAL".
            message: Human-readable message.
        """
        if self._on_user_notification:
            try:
                self._on_user_notification(severity, message)
            except Exception as e:
                logger.error(f"User notification callback failed: {e}")
        else:
            # Default: log with appropriate level
            if severity == "CRITICAL":
                logger.critical(f"USER ALERT: {message}")
            elif severity == "ERROR":
                logger.error(f"USER ALERT: {message}")
            elif severity == "WARNING":
                logger.warning(f"USER ALERT: {message}")
            else:
                logger.info(f"USER ALERT: {message}")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics.
        
        Returns:
            Dict with stats like message counts, error rates, etc.
        """
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        sensor_stats = {}
        for sensor_id, reg in self._sensors.items():
            sensor_stats[sensor_id] = {
                "type": reg.sensor_type.value,
                "status": reg.status.value,
                "frames_received": reg.frames_received,
                "errors": reg.errors_count,
                "success_rate": (
                    (reg.frames_received - reg.errors_count) / reg.frames_received
                    if reg.frames_received > 0 else 1.0
                ),
            }
        
        return {
            "uptime_seconds": uptime,
            "total_messages": self._total_messages,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / self._total_messages if self._total_messages > 0 else 0.0,
            "sensors": sensor_stats,
            "last_message_time": self._last_message_time.isoformat() if self._last_message_time else None,
        }
    
    def get_error_history(self, sensor_id: str | None = None) -> list[ValidationError]:
        """Get error history for a sensor or all sensors.
        
        Args:
            sensor_id: Specific sensor or None for all.
            
        Returns:
            List of ValidationErrors.
        """
        if sensor_id:
            return list(self._error_history.get(sensor_id, []))
        else:
            all_errors = []
            for errors in self._error_history.values():
                all_errors.extend(errors)
            return all_errors
    
    def clear_error_history(self, sensor_id: str | None = None) -> None:
        """Clear error history.
        
        Args:
            sensor_id: Specific sensor or None for all.
        """
        if sensor_id:
            self._error_history[sensor_id].clear()
        else:
            self._error_history.clear()
    
    # =========================================================================
    # Sensor Capabilities Query
    # =========================================================================
    
    def get_sensors_by_capability(
        self,
        capability: SensorCapability,
    ) -> list[SensorRegistration]:
        """Get all sensors that provide a specific capability.
        
        Args:
            capability: The capability to filter by.
            
        Returns:
            List of sensors with that capability.
        """
        return [
            reg for reg in self._sensors.values()
            if capability in reg.capabilities
        ]
    
    def can_answer_question(self, question: str) -> list[str]:
        """Check which sensors can help answer a fundamental question.
        
        Args:
            question: "where", "around", or "happening".
            
        Returns:
            List of sensor IDs that can help.
        """
        capability_map = {
            "where": [
                SensorCapability.PROVIDES_LOCATION,
                SensorCapability.PROVIDES_POSE,
                SensorCapability.PROVIDES_ROOM_ID,
                SensorCapability.PROVIDES_ODOMETRY,
            ],
            "around": [
                SensorCapability.PROVIDES_ENTITIES,
                SensorCapability.PROVIDES_DEPTH,
                SensorCapability.PROVIDES_POINTCLOUD,
            ],
            "happening": [
                SensorCapability.PROVIDES_EVENTS,
                SensorCapability.PROVIDES_STATE_CHANGES,
                SensorCapability.PROVIDES_MOTION,
                SensorCapability.PROVIDES_AUDIO,
            ],
        }
        
        question_lower = question.lower()
        for key in capability_map:
            if key in question_lower:
                capabilities = capability_map[key]
                sensors = []
                for cap in capabilities:
                    for reg in self.get_sensors_by_capability(cap):
                        if reg.sensor_id not in sensors:
                            sensors.append(reg.sensor_id)
                return sensors
        
        return list(self._sensors.keys())
