"""Sensor Gateway - Universal sensor data validation and preprocessing.

This module provides a unified interface for handling sensor data from multiple
sources (Unity, audio, lidar, ultrasound, etc.) and converts them to a common
format for the episodic memory system.

The gateway answers the three fundamental questions:
1. "Where am I?" - Spatial/location information
2. "What's around?" - Entity/object detection
3. "What's happening/can happen?" - Events and predictions
"""

from episodic_agent.modules.sensor_gateway.gateway import SensorGateway
from episodic_agent.modules.sensor_gateway.types import (
    SensorType,
    SensorMessage,
    SensorStatus,
    ValidationResult,
    ValidationError,
    SensorCapability,
)
from episodic_agent.modules.sensor_gateway.validators import (
    SensorValidator,
    UnityValidator,
    GenericValidator,
)
from episodic_agent.modules.sensor_gateway.handlers import (
    SensorHandler,
    UnitySensorHandler,
    GenericSensorHandler,
)
from episodic_agent.modules.sensor_gateway.visual_client import (
    VisualAttentionManager,
    VisualFeatureExtractor,
    VisualRingBuffer,
    VisualStreamClient,
    create_visual_client,
)

__all__ = [
    # Core gateway
    "SensorGateway",
    # Types
    "SensorType",
    "SensorMessage",
    "SensorStatus",
    "ValidationResult",
    "ValidationError",
    "SensorCapability",
    # Validators
    "SensorValidator",
    "UnityValidator",
    "GenericValidator",
    # Handlers
    "SensorHandler",
    "UnitySensorHandler",
    "GenericSensorHandler",
    # Visual channel (Phase 5)
    "VisualAttentionManager",
    "VisualFeatureExtractor",
    "VisualRingBuffer",
    "VisualStreamClient",
    "create_visual_client",
]
