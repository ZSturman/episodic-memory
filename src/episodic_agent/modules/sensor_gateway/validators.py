"""Sensor data validators for the Sensor Gateway.

Validators check incoming sensor data for correctness and completeness,
and can apply automatic corrections where appropriate.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from episodic_agent.modules.sensor_gateway.types import (
    SensorType,
    ValidationError,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


class SensorValidator(ABC):
    """Base class for sensor data validators.
    
    Validators are responsible for:
    1. Checking required fields are present
    2. Validating field types and ranges
    3. Applying automatic corrections where safe
    4. Generating helpful error messages
    """
    
    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """The sensor type this validator handles."""
        ...
    
    @abstractmethod
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate sensor data.
        
        Args:
            data: Raw sensor data to validate.
            
        Returns:
            ValidationResult with status and any errors.
        """
        ...
    
    def _check_required_fields(
        self,
        data: dict[str, Any],
        required: list[str],
        errors: list[ValidationError],
    ) -> bool:
        """Check that required fields are present.
        
        Args:
            data: Data to check.
            required: List of required field names.
            errors: List to append errors to.
            
        Returns:
            True if all required fields present.
        """
        all_present = True
        for field_name in required:
            if field_name not in data:
                errors.append(ValidationError(
                    code="MISSING_REQUIRED_FIELD",
                    message=f"Required field '{field_name}' is missing",
                    severity=ValidationSeverity.ERROR,
                    field_path=field_name,
                    suggestion=f"Ensure the sensor provides '{field_name}' in its output",
                ))
                all_present = False
            elif data[field_name] is None:
                errors.append(ValidationError(
                    code="NULL_REQUIRED_FIELD",
                    message=f"Required field '{field_name}' is null",
                    severity=ValidationSeverity.ERROR,
                    field_path=field_name,
                    suggestion=f"Provide a valid value for '{field_name}'",
                ))
                all_present = False
        
        return all_present
    
    def _check_type(
        self,
        data: dict[str, Any],
        field_name: str,
        expected_type: type | tuple[type, ...],
        errors: list[ValidationError],
        allow_none: bool = True,
    ) -> bool:
        """Check that a field has the expected type.
        
        Args:
            data: Data containing the field.
            field_name: Name of field to check.
            expected_type: Expected type or tuple of types.
            errors: List to append errors to.
            allow_none: Whether None is acceptable.
            
        Returns:
            True if type is correct.
        """
        if field_name not in data:
            return True  # Missing fields handled by required check
        
        value = data[field_name]
        if value is None and allow_none:
            return True
        
        if not isinstance(value, expected_type):
            type_name = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
            errors.append(ValidationError(
                code="INVALID_TYPE",
                message=f"Field '{field_name}' has invalid type",
                severity=ValidationSeverity.ERROR,
                field_path=field_name,
                expected=type_name,
                actual=type(value).__name__,
            ))
            return False
        
        return True
    
    def _check_range(
        self,
        data: dict[str, Any],
        field_name: str,
        errors: list[ValidationError],
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> bool:
        """Check that a numeric field is within range.
        
        Args:
            data: Data containing the field.
            field_name: Name of field to check.
            errors: List to append errors to.
            min_val: Minimum allowed value (inclusive).
            max_val: Maximum allowed value (inclusive).
            
        Returns:
            True if value is in range.
        """
        if field_name not in data or data[field_name] is None:
            return True
        
        value = data[field_name]
        if not isinstance(value, (int, float)):
            return True  # Type check handles this
        
        if min_val is not None and value < min_val:
            errors.append(ValidationError(
                code="VALUE_BELOW_MINIMUM",
                message=f"Field '{field_name}' is below minimum",
                severity=ValidationSeverity.WARNING,
                field_path=field_name,
                expected=f">= {min_val}",
                actual=value,
            ))
            return False
        
        if max_val is not None and value > max_val:
            errors.append(ValidationError(
                code="VALUE_ABOVE_MAXIMUM",
                message=f"Field '{field_name}' exceeds maximum",
                severity=ValidationSeverity.WARNING,
                field_path=field_name,
                expected=f"<= {max_val}",
                actual=value,
            ))
            return False
        
        return True


class UnityValidator(SensorValidator):
    """Validator for Unity WebSocket sensor data.
    
    Validates frames against the Unity sensor protocol.
    
    ARCHITECTURAL INVARIANT: Unity sends GUIDs and observables only, no semantic labels.
    Labels are learned from user interaction and owned by the backend.
    """
    
    EXPECTED_PROTOCOL = "1.0.0"
    REQUIRED_FIELDS = ["protocol_version", "frame_id", "timestamp"]
    # REMOVED: current_room_label - backend owns all semantic labels
    OPTIONAL_FIELDS = ["camera_pose", "current_room", "current_room_guid", 
                       "entities", "state_changes"]
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.UNITY_WEBSOCKET
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate Unity sensor frame.
        
        Args:
            data: Raw Unity frame data.
            
        Returns:
            ValidationResult with status, errors, and corrected data.
        """
        errors: list[ValidationError] = []
        corrected = dict(data)  # Start with copy
        
        # Check required fields
        if not self._check_required_fields(data, self.REQUIRED_FIELDS, errors):
            # Try to provide defaults for missing fields
            if "frame_id" not in data:
                corrected["frame_id"] = 0
                errors.append(ValidationError(
                    code="AUTO_CORRECTED",
                    message="Missing frame_id defaulted to 0",
                    severity=ValidationSeverity.INFO,
                    field_path="frame_id",
                ))
            
            if "timestamp" not in data:
                corrected["timestamp"] = datetime.now().timestamp()
                errors.append(ValidationError(
                    code="AUTO_CORRECTED",
                    message="Missing timestamp set to current time",
                    severity=ValidationSeverity.WARNING,
                    field_path="timestamp",
                ))
            
            if "protocol_version" not in data:
                corrected["protocol_version"] = self.EXPECTED_PROTOCOL
                errors.append(ValidationError(
                    code="AUTO_CORRECTED",
                    message=f"Missing protocol_version defaulted to {self.EXPECTED_PROTOCOL}",
                    severity=ValidationSeverity.WARNING,
                    field_path="protocol_version",
                ))
        
        # Validate protocol version
        protocol = data.get("protocol_version")
        if protocol and protocol != self.EXPECTED_PROTOCOL:
            errors.append(ValidationError(
                code="PROTOCOL_MISMATCH",
                message="Protocol version mismatch",
                severity=ValidationSeverity.WARNING,
                field_path="protocol_version",
                expected=self.EXPECTED_PROTOCOL,
                actual=protocol,
                suggestion="Update Unity or Python client to match versions",
            ))
        
        # Validate frame_id type
        if "frame_id" in data:
            frame_id = data["frame_id"]
            # Handle case where frame_id is a string (common error)
            if isinstance(frame_id, str):
                try:
                    corrected["frame_id"] = int(frame_id)
                    errors.append(ValidationError(
                        code="AUTO_CORRECTED",
                        message="Converted frame_id from string to int",
                        severity=ValidationSeverity.INFO,
                        field_path="frame_id",
                        expected="int",
                        actual="str",
                    ))
                except ValueError:
                    errors.append(ValidationError(
                        code="INVALID_FRAME_ID",
                        message="frame_id cannot be converted to integer",
                        severity=ValidationSeverity.ERROR,
                        field_path="frame_id",
                        expected="int",
                        actual=frame_id,
                    ))
        
        # Validate timestamp
        if "timestamp" in data:
            timestamp = data["timestamp"]
            if isinstance(timestamp, str):
                # Try to parse ISO format
                try:
                    parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    corrected["timestamp"] = parsed.timestamp()
                    errors.append(ValidationError(
                        code="AUTO_CORRECTED",
                        message="Converted timestamp from ISO string to Unix timestamp",
                        severity=ValidationSeverity.INFO,
                        field_path="timestamp",
                    ))
                except ValueError:
                    errors.append(ValidationError(
                        code="INVALID_TIMESTAMP",
                        message="Cannot parse timestamp string",
                        severity=ValidationSeverity.ERROR,
                        field_path="timestamp",
                        expected="Unix timestamp or ISO format",
                        actual=timestamp,
                    ))
        
        # Validate camera_pose structure
        if "camera_pose" in data and data["camera_pose"]:
            camera_pose = data["camera_pose"]
            if isinstance(camera_pose, dict):
                self._validate_camera_pose(camera_pose, errors)
            else:
                errors.append(ValidationError(
                    code="INVALID_TYPE",
                    message="camera_pose must be an object",
                    severity=ValidationSeverity.WARNING,
                    field_path="camera_pose",
                    expected="object with position, rotation, forward",
                    actual=type(camera_pose).__name__,
                ))
        
        # Validate entities
        if "entities" in data:
            entities = data["entities"]
            if not isinstance(entities, list):
                errors.append(ValidationError(
                    code="INVALID_TYPE",
                    message="entities must be an array",
                    severity=ValidationSeverity.ERROR,
                    field_path="entities",
                    expected="array",
                    actual=type(entities).__name__,
                ))
                corrected["entities"] = []
            else:
                corrected["entities"] = []
                for i, entity in enumerate(entities):
                    if isinstance(entity, dict):
                        valid_entity = self._validate_entity(entity, i, errors)
                        corrected["entities"].append(valid_entity)
                    else:
                        errors.append(ValidationError(
                            code="INVALID_ENTITY",
                            message=f"Entity at index {i} is not an object",
                            severity=ValidationSeverity.WARNING,
                            field_path=f"entities[{i}]",
                            expected="object",
                            actual=type(entity).__name__,
                        ))
        
        # Validate state_changes
        if "state_changes" in data and data["state_changes"]:
            changes = data["state_changes"]
            if not isinstance(changes, list):
                errors.append(ValidationError(
                    code="INVALID_TYPE",
                    message="state_changes must be an array",
                    severity=ValidationSeverity.WARNING,
                    field_path="state_changes",
                ))
                corrected["state_changes"] = []
        
        # Determine if valid (no errors or only warnings/info)
        is_valid = not any(
            e.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) 
            for e in errors
        )
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            corrected_data=corrected,
            raw_data=data,
        )
    
    def _validate_camera_pose(
        self,
        pose: dict[str, Any],
        errors: list[ValidationError],
    ) -> None:
        """Validate camera pose structure."""
        for vector_name in ["position", "rotation", "forward"]:
            if vector_name in pose:
                vector = pose[vector_name]
                if isinstance(vector, dict):
                    for axis in ["x", "y", "z"]:
                        if axis not in vector:
                            errors.append(ValidationError(
                                code="MISSING_AXIS",
                                message=f"camera_pose.{vector_name} missing '{axis}' axis",
                                severity=ValidationSeverity.WARNING,
                                field_path=f"camera_pose.{vector_name}.{axis}",
                            ))
                        elif not isinstance(vector.get(axis), (int, float)):
                            errors.append(ValidationError(
                                code="INVALID_AXIS_TYPE",
                                message=f"camera_pose.{vector_name}.{axis} must be a number",
                                severity=ValidationSeverity.WARNING,
                                field_path=f"camera_pose.{vector_name}.{axis}",
                                expected="number",
                                actual=type(vector.get(axis)).__name__,
                            ))
    
    def _validate_entity(
        self,
        entity: dict[str, Any],
        index: int,
        errors: list[ValidationError],
    ) -> dict[str, Any]:
        """Validate a single entity and return corrected version."""
        corrected = dict(entity)
        
        # Required entity fields
        if "guid" not in entity:
            corrected["guid"] = f"unknown_{index}"
            errors.append(ValidationError(
                code="MISSING_ENTITY_GUID",
                message=f"Entity at index {index} missing guid",
                severity=ValidationSeverity.WARNING,
                field_path=f"entities[{index}].guid",
                suggestion="Entity will be assigned a temporary ID",
            ))
        
        if "label" not in entity:
            corrected["label"] = "unknown"
        
        if "category" not in entity:
            corrected["category"] = "unknown"
        
        # Validate position if present
        if "position" in entity:
            pos = entity["position"]
            if isinstance(pos, dict):
                for axis in ["x", "y", "z"]:
                    if axis not in pos:
                        corrected.setdefault("position", {})[axis] = 0.0
        
        return corrected


class GenericValidator(SensorValidator):
    """Generic validator for unknown or custom sensor types.
    
    Performs minimal validation, accepting most data formats.
    """
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.UNKNOWN
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Perform minimal validation on generic sensor data.
        
        Args:
            data: Raw sensor data.
            
        Returns:
            ValidationResult (usually valid unless data is empty).
        """
        errors: list[ValidationError] = []
        
        if not data:
            errors.append(ValidationError(
                code="EMPTY_DATA",
                message="Sensor data is empty",
                severity=ValidationSeverity.ERROR,
            ))
            return ValidationResult(
                is_valid=False,
                errors=errors,
                raw_data=data,
            )
        
        # Check for common fields we might expect
        if "timestamp" not in data:
            errors.append(ValidationError(
                code="MISSING_TIMESTAMP",
                message="No timestamp field - timing may be unreliable",
                severity=ValidationSeverity.WARNING,
                suggestion="Add a 'timestamp' field for accurate timing",
            ))
        
        return ValidationResult(
            is_valid=True,
            errors=errors,
            corrected_data=data,
            raw_data=data,
        )


class LidarValidator(SensorValidator):
    """Validator for LiDAR sensor data."""
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.LIDAR
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate LiDAR point cloud data."""
        errors: list[ValidationError] = []
        corrected = dict(data)
        
        # Check for point cloud data
        if "points" not in data and "point_cloud" not in data:
            errors.append(ValidationError(
                code="MISSING_POINT_CLOUD",
                message="No point cloud data found",
                severity=ValidationSeverity.ERROR,
                suggestion="Provide 'points' or 'point_cloud' array",
            ))
            return ValidationResult(
                is_valid=False,
                errors=errors,
                raw_data=data,
            )
        
        points_key = "points" if "points" in data else "point_cloud"
        points = data[points_key]
        
        if not isinstance(points, list):
            errors.append(ValidationError(
                code="INVALID_POINT_CLOUD",
                message="Point cloud must be an array",
                severity=ValidationSeverity.ERROR,
                field_path=points_key,
            ))
            return ValidationResult(
                is_valid=False,
                errors=errors,
                raw_data=data,
            )
        
        # Validate point structure (sample first few points)
        if points:
            sample = points[:min(10, len(points))]
            for i, point in enumerate(sample):
                if not isinstance(point, (list, dict)):
                    errors.append(ValidationError(
                        code="INVALID_POINT_FORMAT",
                        message=f"Point at index {i} has invalid format",
                        severity=ValidationSeverity.WARNING,
                        field_path=f"{points_key}[{i}]",
                        expected="[x, y, z] array or {x, y, z} object",
                    ))
                    break
        
        is_valid = not any(e.severity == ValidationSeverity.ERROR for e in errors)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            corrected_data=corrected,
            raw_data=data,
        )


class AudioValidator(SensorValidator):
    """Validator for audio/microphone sensor data."""
    
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.MICROPHONE
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate audio sensor data."""
        errors: list[ValidationError] = []
        corrected = dict(data)
        
        # Check for audio data in various formats
        has_audio = any(key in data for key in [
            "samples", "audio", "waveform", "spectrogram", "transcript"
        ])
        
        if not has_audio:
            errors.append(ValidationError(
                code="MISSING_AUDIO_DATA",
                message="No audio data found",
                severity=ValidationSeverity.ERROR,
                suggestion="Provide 'samples', 'audio', 'waveform', or 'transcript'",
            ))
            return ValidationResult(
                is_valid=False,
                errors=errors,
                raw_data=data,
            )
        
        # Validate sample rate if present
        if "sample_rate" in data:
            rate = data["sample_rate"]
            if not isinstance(rate, (int, float)) or rate <= 0:
                errors.append(ValidationError(
                    code="INVALID_SAMPLE_RATE",
                    message="Invalid sample rate",
                    severity=ValidationSeverity.WARNING,
                    field_path="sample_rate",
                    expected="positive number",
                    actual=rate,
                ))
        
        is_valid = not any(e.severity == ValidationSeverity.ERROR for e in errors)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            corrected_data=corrected,
            raw_data=data,
        )


# Registry of validators by sensor type
VALIDATORS: dict[SensorType, type[SensorValidator]] = {
    SensorType.UNITY_WEBSOCKET: UnityValidator,
    SensorType.LIDAR: LidarValidator,
    SensorType.MICROPHONE: AudioValidator,
    SensorType.ULTRASOUND: AudioValidator,  # Similar validation
    SensorType.UNKNOWN: GenericValidator,
    SensorType.CUSTOM: GenericValidator,
}


def get_validator(sensor_type: SensorType) -> SensorValidator:
    """Get the appropriate validator for a sensor type.
    
    Args:
        sensor_type: The type of sensor to validate.
        
    Returns:
        A validator instance for that sensor type.
    """
    validator_class = VALIDATORS.get(sensor_type, GenericValidator)
    return validator_class()
