"""Protocol Tests: Message Schemas + Wire Format.

Tests for protocol message schemas, validation, serialization,
and architectural invariant enforcement.

Run with: pytest tests/test_protocol.py -v
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import pytest

from episodic_agent.schemas.protocol import (
    CapabilitiesReport,
    EntityUpdate,
    ErrorSeverity,
    FrameAck,
    Handshake,
    LabelConfidence,
    LabelRequest,
    LabelResponse,
    LabelResponseType,
    LabelTargetType,
    LocationUpdate,
    MessageType,
    ProtocolError,
    ProtocolMessage,
    SensorCapability,
    StreamControl,
    StreamControlCommand,
)


# =============================================================================
# MESSAGE TYPE TESTS
# =============================================================================


class TestMessageType:
    """Tests for MessageType enumeration."""

    def test_all_message_types_defined(self):
        """Verify all expected message types exist."""
        expected_types = [
            "sensor_frame",
            "capabilities_report",
            "visual_summary",
            "visual_focus",
            "label_response",
            "error",
            "frame_ack",
            "stream_control",
            "focus_request",
            "label_request",
            "entity_update",
            "location_update",
            "heartbeat",
            "handshake",
        ]
        
        actual_types = [mt.value for mt in MessageType]
        for expected in expected_types:
            assert expected in actual_types, f"Missing message type: {expected}"

    def test_sensor_to_backend_types(self):
        """Verify sensor→backend message types."""
        sensor_types = [
            MessageType.SENSOR_FRAME,
            MessageType.CAPABILITIES_REPORT,
            MessageType.VISUAL_SUMMARY,
            MessageType.VISUAL_FOCUS,
            MessageType.LABEL_RESPONSE,
            MessageType.ERROR,
        ]
        for mt in sensor_types:
            assert mt is not None
            assert mt.value != ""

    def test_backend_to_sensor_types(self):
        """Verify backend→sensor message types."""
        backend_types = [
            MessageType.FRAME_ACK,
            MessageType.STREAM_CONTROL,
            MessageType.FOCUS_REQUEST,
            MessageType.LABEL_REQUEST,
            MessageType.ENTITY_UPDATE,
            MessageType.LOCATION_UPDATE,
        ]
        for mt in backend_types:
            assert mt is not None
            assert mt.value != ""

    def test_bidirectional_types(self):
        """Verify bidirectional message types."""
        bidir_types = [
            MessageType.HEARTBEAT,
            MessageType.HANDSHAKE,
        ]
        for mt in bidir_types:
            assert mt is not None
            assert mt.value != ""


# =============================================================================
# SENSOR CAPABILITY TESTS
# =============================================================================


class TestSensorCapability:
    """Tests for SensorCapability enumeration."""

    def test_visual_capabilities(self):
        """Verify visual capability types."""
        visual = [
            SensorCapability.RGB_CAMERA,
            SensorCapability.DEPTH_CAMERA,
            SensorCapability.STEREO_CAMERA,
            SensorCapability.ZOOM,
            SensorCapability.FOCUS,
            SensorCapability.PAN_TILT,
        ]
        for cap in visual:
            assert cap is not None

    def test_spatial_capabilities(self):
        """Verify spatial capability types."""
        spatial = [
            SensorCapability.ODOMETRY,
            SensorCapability.IMU,
            SensorCapability.GPS,
            SensorCapability.LIDAR,
        ]
        for cap in spatial:
            assert cap is not None

    def test_detection_capabilities(self):
        """Verify detection capability types."""
        detection = [
            SensorCapability.BOUNDING_BOXES,
            SensorCapability.SEGMENTATION,
            SensorCapability.TRACKING,
        ]
        for cap in detection:
            assert cap is not None


# =============================================================================
# CAPABILITIES REPORT TESTS
# =============================================================================


class TestCapabilitiesReport:
    """Tests for CapabilitiesReport schema."""

    def test_minimal_capabilities_report(self):
        """Create minimal capabilities report."""
        report = CapabilitiesReport(
            sensor_id="test-001",
            sensor_type="camera",
        )
        
        assert report.sensor_id == "test-001"
        assert report.sensor_type == "camera"
        assert report.sensor_version == "1.0.0"
        assert report.capabilities == []
        assert report.max_fps == 30.0
        assert report.compute_available is False

    def test_full_capabilities_report(self):
        """Create fully-specified capabilities report."""
        report = CapabilitiesReport(
            sensor_id="unity-sim-001",
            sensor_type="camera",
            sensor_version="2.0.0",
            capabilities=[
                SensorCapability.RGB_CAMERA,
                SensorCapability.DEPTH_CAMERA,
                SensorCapability.BOUNDING_BOXES,
            ],
            max_resolution=(1920, 1080),
            min_resolution=(320, 240),
            max_fps=60.0,
            min_fps=5.0,
            compute_available=True,
            compute_tflops=5.0,
            buffer_size_mb=100,
            supports_visual_channel=True,
            supports_focus_requests=True,
            extras={"hardware": "GPU", "model": "test"},
        )
        
        assert len(report.capabilities) == 3
        assert report.max_resolution == (1920, 1080)
        assert report.compute_tflops == 5.0
        assert report.extras["hardware"] == "GPU"

    def test_capabilities_report_serialization(self):
        """Test JSON serialization."""
        report = CapabilitiesReport(
            sensor_id="test-001",
            sensor_type="camera",
            capabilities=[SensorCapability.RGB_CAMERA],
        )
        
        # To JSON
        json_str = report.model_dump_json()
        assert "test-001" in json_str
        assert "rgb_camera" in json_str
        
        # From JSON
        data = json.loads(json_str)
        restored = CapabilitiesReport.model_validate(data)
        assert restored.sensor_id == report.sensor_id

    def test_capabilities_as_list(self):
        """Capabilities should be a list for multiple values."""
        report = CapabilitiesReport(
            sensor_id="test",
            sensor_type="robot",
            capabilities=[
                SensorCapability.RGB_CAMERA,
                SensorCapability.LIDAR,
                SensorCapability.ODOMETRY,
            ],
        )
        
        assert len(report.capabilities) == 3
        assert SensorCapability.LIDAR in report.capabilities


# =============================================================================
# STREAM CONTROL TESTS
# =============================================================================


class TestStreamControl:
    """Tests for StreamControl schema."""

    def test_start_command(self):
        """Test start streaming command."""
        cmd = StreamControl(command=StreamControlCommand.START)
        assert cmd.command == StreamControlCommand.START
        assert cmd.resolution is None

    def test_set_resolution_command(self):
        """Test resolution change command."""
        cmd = StreamControl(
            command=StreamControlCommand.SET_RESOLUTION,
            resolution=(640, 480),
            request_id="cmd-001",
        )
        
        assert cmd.command == StreamControlCommand.SET_RESOLUTION
        assert cmd.resolution == (640, 480)
        assert cmd.request_id == "cmd-001"

    def test_set_fps_command(self):
        """Test FPS change command."""
        cmd = StreamControl(
            command=StreamControlCommand.SET_FPS,
            fps=15.0,
        )
        
        assert cmd.fps == 15.0

    def test_set_crop_command(self):
        """Test crop region command."""
        cmd = StreamControl(
            command=StreamControlCommand.SET_CROP,
            crop_region=(100, 50, 200, 150),
            duration_seconds=5.0,
        )
        
        assert cmd.crop_region == (100, 50, 200, 150)
        assert cmd.duration_seconds == 5.0

    def test_all_stream_commands(self):
        """Verify all stream control commands exist."""
        commands = [
            StreamControlCommand.START,
            StreamControlCommand.STOP,
            StreamControlCommand.PAUSE,
            StreamControlCommand.RESUME,
            StreamControlCommand.SET_RESOLUTION,
            StreamControlCommand.SET_FPS,
            StreamControlCommand.SET_CROP,
            StreamControlCommand.CLEAR_CROP,
            StreamControlCommand.ENABLE_SUMMARY,
            StreamControlCommand.DISABLE_SUMMARY,
            StreamControlCommand.REQUEST_KEYFRAME,
        ]
        
        assert len(commands) == 11
        for cmd in commands:
            assert cmd.value != ""


# =============================================================================
# LABEL REQUEST/RESPONSE TESTS
# =============================================================================


class TestLabelRequest:
    """Tests for LabelRequest schema."""

    def test_minimal_label_request(self):
        """Create minimal label request."""
        request = LabelRequest(
            request_id="req-001",
            target_type=LabelTargetType.ENTITY,
            target_id="entity-abc",
            confidence=LabelConfidence.LOW,
        )
        
        assert request.request_id == "req-001"
        assert request.target_type == LabelTargetType.ENTITY
        assert request.confidence == LabelConfidence.LOW
        assert request.current_label is None
        assert request.timeout_seconds == 30.0

    def test_full_label_request(self):
        """Create fully-specified label request."""
        request = LabelRequest(
            request_id="req-002",
            target_type=LabelTargetType.LOCATION,
            target_id="loc-xyz",
            confidence=LabelConfidence.MEDIUM,
            current_label="kitchen?",
            alternative_labels=["dining room", "cooking area"],
            thumbnail_base64="<base64>",
            bounding_box=(10, 20, 100, 80),
            description="What room is this?",
            timeout_seconds=60.0,
        )
        
        assert request.current_label == "kitchen?"
        assert len(request.alternative_labels) == 2
        assert request.bounding_box == (10, 20, 100, 80)

    def test_all_target_types(self):
        """Verify all label target types."""
        targets = [
            LabelTargetType.ENTITY,
            LabelTargetType.LOCATION,
            LabelTargetType.EVENT,
            LabelTargetType.RELATION,
        ]
        
        assert len(targets) == 4

    def test_all_confidence_levels(self):
        """Verify all confidence levels."""
        levels = [
            LabelConfidence.LOW,
            LabelConfidence.MEDIUM,
            LabelConfidence.HIGH,
        ]
        
        assert len(levels) == 3


class TestLabelResponse:
    """Tests for LabelResponse schema."""

    def test_provided_response(self):
        """Test user-provided label response."""
        response = LabelResponse(
            request_id="req-001",
            response_type=LabelResponseType.PROVIDED,
            label="coffee mug",
            confidence=1.0,
            response_time_ms=2500.0,
        )
        
        assert response.response_type == LabelResponseType.PROVIDED
        assert response.label == "coffee mug"
        assert response.confidence == 1.0

    def test_confirmed_response(self):
        """Test user-confirmed label response."""
        response = LabelResponse(
            request_id="req-002",
            response_type=LabelResponseType.CONFIRMED,
            label="kitchen",
        )
        
        assert response.response_type == LabelResponseType.CONFIRMED

    def test_rejected_response(self):
        """Test user-rejected response."""
        response = LabelResponse(
            request_id="req-003",
            response_type=LabelResponseType.REJECTED,
            notes="None of these are correct",
        )
        
        assert response.response_type == LabelResponseType.REJECTED
        assert response.label is None
        assert response.notes == "None of these are correct"

    def test_timeout_response(self):
        """Test timeout response."""
        response = LabelResponse(
            request_id="req-004",
            response_type=LabelResponseType.TIMEOUT,
        )
        
        assert response.response_type == LabelResponseType.TIMEOUT
        assert response.label is None

    def test_skipped_response(self):
        """Test skipped response."""
        response = LabelResponse(
            request_id="req-005",
            response_type=LabelResponseType.SKIPPED,
        )
        
        assert response.response_type == LabelResponseType.SKIPPED

    def test_confidence_bounds(self):
        """Test confidence value validation."""
        # Valid confidence
        response = LabelResponse(
            request_id="req-006",
            response_type=LabelResponseType.PROVIDED,
            label="test",
            confidence=0.5,
        )
        assert 0.0 <= response.confidence <= 1.0
        
        # Edge cases
        response_low = LabelResponse(
            request_id="req-007",
            response_type=LabelResponseType.PROVIDED,
            label="test",
            confidence=0.0,
        )
        assert response_low.confidence == 0.0
        
        response_high = LabelResponse(
            request_id="req-008",
            response_type=LabelResponseType.PROVIDED,
            label="test",
            confidence=1.0,
        )
        assert response_high.confidence == 1.0


# =============================================================================
# ENTITY/LOCATION UPDATE TESTS
# =============================================================================


class TestEntityUpdate:
    """Tests for EntityUpdate schema."""

    def test_minimal_entity_update(self):
        """Create minimal entity update."""
        update = EntityUpdate(entity_id="entity-001")
        
        assert update.entity_id == "entity-001"
        assert update.label == "unknown"
        assert update.labels == []
        assert update.confidence == 0.0
        assert update.visible is True

    def test_full_entity_update(self):
        """Create fully-specified entity update."""
        update = EntityUpdate(
            entity_id="entity-002",
            label="coffee mug",
            labels=["coffee mug", "mug", "cup"],
            confidence=0.85,
            visible=True,
            state="full",
            relative_position=(1.2, -0.3, 0.5),
        )
        
        assert update.label == "coffee mug"
        assert len(update.labels) == 3
        assert update.state == "full"
        assert update.relative_position == (1.2, -0.3, 0.5)


class TestLocationUpdate:
    """Tests for LocationUpdate schema."""

    def test_minimal_location_update(self):
        """Create minimal location update."""
        update = LocationUpdate(location_id="loc-001")
        
        assert update.location_id == "loc-001"
        assert update.label == "unknown"
        assert update.is_stable is True

    def test_uncertain_location_update(self):
        """Create uncertain location update."""
        update = LocationUpdate(
            location_id="loc-002",
            label="kitchen?",
            labels=["kitchen?", "dining area?"],
            confidence=0.4,
            is_stable=False,
            uncertainty_reason="Motion/perception conflict",
        )
        
        assert update.is_stable is False
        assert update.uncertainty_reason == "Motion/perception conflict"


# =============================================================================
# FRAME ACK TESTS
# =============================================================================


class TestFrameAck:
    """Tests for FrameAck schema."""

    def test_simple_ack(self):
        """Test simple frame acknowledgment."""
        ack = FrameAck(frame_id=42)
        
        assert ack.frame_id == 42
        assert ack.processed is True
        assert ack.entities_detected == 0

    def test_detailed_ack(self):
        """Test detailed frame acknowledgment."""
        ack = FrameAck(
            frame_id=100,
            processed=True,
            processing_time_ms=15.5,
            entities_detected=5,
            events_detected=2,
        )
        
        assert ack.processing_time_ms == 15.5
        assert ack.entities_detected == 5
        assert ack.events_detected == 2


# =============================================================================
# HANDSHAKE TESTS
# =============================================================================


class TestHandshake:
    """Tests for Handshake schema."""

    def test_sensor_handshake(self):
        """Test sensor-side handshake."""
        handshake = Handshake(
            role="sensor",
            identity="unity-sim-001",
        )
        
        assert handshake.role == "sensor"
        assert handshake.protocol_version == "1.0.0"
        assert handshake.session_id is None

    def test_backend_handshake(self):
        """Test backend-side handshake."""
        handshake = Handshake(
            role="backend",
            identity="episodic-agent",
            session_id="session-abc123",
        )
        
        assert handshake.role == "backend"
        assert handshake.session_id == "session-abc123"


# =============================================================================
# ERROR TESTS
# =============================================================================


class TestProtocolError:
    """Tests for ProtocolError schema."""

    def test_warning_error(self):
        """Test warning severity error."""
        error = ProtocolError(
            severity=ErrorSeverity.WARNING,
            code="LOW_BANDWIDTH",
            message="Connection bandwidth below optimal",
        )
        
        assert error.severity == ErrorSeverity.WARNING
        assert error.recoverable is True

    def test_critical_error(self):
        """Test critical severity error."""
        error = ProtocolError(
            severity=ErrorSeverity.CRITICAL,
            code="CONNECTION_LOST",
            message="Sensor connection lost",
            recoverable=False,
            suggested_action="Check sensor power and network",
        )
        
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.recoverable is False
        assert error.suggested_action is not None

    def test_error_with_context(self):
        """Test error with related message context."""
        error = ProtocolError(
            severity=ErrorSeverity.ERROR,
            code="CAPABILITY_NOT_SUPPORTED",
            message="Zoom capability not available",
            related_message_id="cmd-001",
            details={"requested": "zoom", "available": ["rgb_camera"]},
        )
        
        assert error.related_message_id == "cmd-001"
        assert "requested" in error.details


# =============================================================================
# PROTOCOL MESSAGE ENVELOPE TESTS
# =============================================================================


class TestProtocolMessage:
    """Tests for ProtocolMessage envelope."""

    def test_create_message(self):
        """Test creating protocol message with factory method."""
        request = LabelRequest(
            request_id="req-001",
            target_type=LabelTargetType.ENTITY,
            target_id="entity-abc",
            confidence=LabelConfidence.LOW,
        )
        
        message = ProtocolMessage.create(
            message_type=MessageType.LABEL_REQUEST,
            payload=request,
            source="backend",
        )
        
        assert message.message_type == MessageType.LABEL_REQUEST
        assert message.source == "backend"
        assert message.message_id is not None
        assert "request_id" in message.payload

    def test_message_with_correlation(self):
        """Test message with correlation ID."""
        message = ProtocolMessage.create(
            message_type=MessageType.LABEL_RESPONSE,
            payload={"request_id": "req-001", "label": "test"},
            source="sensor-001",
            correlation_id="session-abc",
            in_reply_to="msg-001",
        )
        
        assert message.correlation_id == "session-abc"
        assert message.in_reply_to == "msg-001"

    def test_message_serialization(self):
        """Test message serialization to JSON."""
        message = ProtocolMessage.create(
            message_type=MessageType.STREAM_CONTROL,
            payload=StreamControl(command=StreamControlCommand.START),
            source="backend",
        )
        
        # To JSON
        json_str = message.model_dump_json()
        assert "stream_control" in json_str
        assert "backend" in json_str
        
        # From JSON
        data = json.loads(json_str)
        restored = ProtocolMessage.model_validate(data)
        assert restored.message_type == MessageType.STREAM_CONTROL

    def test_message_id_is_uuid(self):
        """Message ID should be a valid UUID."""
        message = ProtocolMessage.create(
            message_type=MessageType.HEARTBEAT,
            payload={},
        )
        
        # Should not raise
        uuid.UUID(message.message_id)


# =============================================================================
# ARCHITECTURAL INVARIANT TESTS
# =============================================================================


class TestProtocolInvariants:
    """Tests verifying architectural invariants in protocol design."""

    def test_no_absolute_coordinates_in_entity_update(self):
        """EntityUpdate uses relative_position, not world coordinates.
        
        INVARIANT: Protocol is sensor-agnostic - no absolute world coordinates.
        """
        update = EntityUpdate(
            entity_id="entity-001",
            relative_position=(1.2, -0.3, 0.5),
        )
        
        # The field is named "relative_position", not "world_position"
        assert hasattr(update, "relative_position")
        assert not hasattr(update, "world_position")
        assert not hasattr(update, "absolute_position")

    def test_labels_require_user_flow(self):
        """Labels enter system ONLY through LabelRequest/LabelResponse.
        
        INVARIANT: Labels come from users, not assumptions.
        """
        # LabelRequest is how backend asks for labels
        request = LabelRequest(
            request_id="req-001",
            target_type=LabelTargetType.ENTITY,
            target_id="entity-001",
            confidence=LabelConfidence.LOW,
        )
        
        # LabelResponse is how labels enter
        response = LabelResponse(
            request_id="req-001",
            response_type=LabelResponseType.PROVIDED,
            label="user-provided-label",
        )
        
        # Verify the flow exists
        assert request.request_id == response.request_id
        assert response.label is not None

    def test_capabilities_before_features(self):
        """Backend must check capabilities before using features.
        
        INVARIANT: Backend must not assume capabilities.
        """
        # Sensor reports what it can do
        report = CapabilitiesReport(
            sensor_id="test-001",
            sensor_type="camera",
            capabilities=[SensorCapability.RGB_CAMERA],  # No ZOOM
            supports_focus_requests=False,
        )
        
        # Backend should check before requesting zoom
        assert SensorCapability.ZOOM not in report.capabilities
        assert not report.supports_focus_requests
        
        # Focus request exists in protocol but should be gated by capability check
        # (This is architectural - the check happens in application code)

    def test_stream_control_backend_authority(self):
        """Backend controls the stream, not sensor.
        
        INVARIANT: Backend controls the stream, not sensor.
        """
        # All stream control commands come FROM backend TO sensor
        commands = [
            StreamControl(command=StreamControlCommand.START),
            StreamControl(command=StreamControlCommand.SET_RESOLUTION, resolution=(640, 480)),
            StreamControl(command=StreamControlCommand.ENABLE_SUMMARY),
        ]
        
        # These are all valid - backend has authority
        for cmd in commands:
            assert cmd.command is not None

    def test_message_envelope_traceability(self):
        """Every message has ID and timestamp for auditability.
        
        INVARIANT: All decisions are logged (messages enable this).
        """
        message = ProtocolMessage.create(
            message_type=MessageType.LABEL_REQUEST,
            payload={},
        )
        
        # Required for audit trail
        assert message.message_id is not None
        assert message.timestamp is not None
        assert message.message_type is not None

    def test_no_predefined_semantic_labels(self):
        """Protocol doesn't include predefined semantic labels.
        
        INVARIANT: No pre-wired semantics.
        """
        # LabelTargetType is structural (what KIND of thing), not semantic
        types = [lt.value for lt in LabelTargetType]
        assert "entity" in types  # Structural: it's an entity
        assert "kitchen" not in types  # Semantic: what kind of entity
        
        # LabelConfidence is about confidence level, not semantic meaning
        levels = [lc.value for lc in LabelConfidence]
        assert "low" in levels
        assert "furniture" not in levels

    def test_error_is_recoverable_by_default(self):
        """Errors default to recoverable to prevent over-reaction.
        
        Supports graceful degradation in real-world deployments.
        """
        error = ProtocolError(
            severity=ErrorSeverity.ERROR,
            code="TEST",
            message="Test error",
        )
        
        assert error.recoverable is True  # Default


# =============================================================================
# SERIALIZATION ROUND-TRIP TESTS
# =============================================================================


class TestSerializationRoundTrip:
    """Tests for complete serialization round-trips."""

    def test_capabilities_report_roundtrip(self):
        """CapabilitiesReport survives JSON round-trip."""
        original = CapabilitiesReport(
            sensor_id="test-001",
            sensor_type="camera",
            capabilities=[SensorCapability.RGB_CAMERA, SensorCapability.TRACKING],
            max_resolution=(1920, 1080),
            compute_available=True,
        )
        
        json_str = original.model_dump_json()
        restored = CapabilitiesReport.model_validate_json(json_str)
        
        assert restored.sensor_id == original.sensor_id
        assert len(restored.capabilities) == 2
        assert restored.max_resolution == (1920, 1080)

    def test_label_request_roundtrip(self):
        """LabelRequest survives JSON round-trip."""
        original = LabelRequest(
            request_id="req-001",
            target_type=LabelTargetType.ENTITY,
            target_id="entity-abc",
            confidence=LabelConfidence.MEDIUM,
            current_label="mug?",
            alternative_labels=["cup", "glass"],
        )
        
        json_str = original.model_dump_json()
        restored = LabelRequest.model_validate_json(json_str)
        
        assert restored.request_id == original.request_id
        assert restored.target_type == LabelTargetType.ENTITY
        assert restored.current_label == "mug?"

    def test_protocol_message_roundtrip(self):
        """ProtocolMessage envelope survives JSON round-trip."""
        original = ProtocolMessage.create(
            message_type=MessageType.ENTITY_UPDATE,
            payload=EntityUpdate(
                entity_id="entity-001",
                label="test",
                labels=["test", "alias"],
            ),
            source="backend",
            correlation_id="corr-001",
        )
        
        json_str = original.model_dump_json()
        restored = ProtocolMessage.model_validate_json(json_str)
        
        assert restored.message_type == MessageType.ENTITY_UPDATE
        assert restored.correlation_id == "corr-001"
        assert "entity_id" in restored.payload


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_capabilities_list(self):
        """Sensor with no special capabilities."""
        report = CapabilitiesReport(
            sensor_id="basic",
            sensor_type="simple",
            capabilities=[],
        )
        
        assert len(report.capabilities) == 0

    def test_label_response_without_label(self):
        """Responses that don't provide a label."""
        for response_type in [
            LabelResponseType.REJECTED,
            LabelResponseType.TIMEOUT,
            LabelResponseType.SKIPPED,
        ]:
            response = LabelResponse(
                request_id="req-001",
                response_type=response_type,
            )
            # These response types don't require a label
            assert response.label is None

    def test_entity_update_without_position(self):
        """Entity update when position is unknown."""
        update = EntityUpdate(
            entity_id="entity-001",
            label="somewhere",
            visible=False,  # Not visible, no position
            relative_position=None,
        )
        
        assert update.relative_position is None

    def test_location_update_with_uncertainty(self):
        """Location update with instability."""
        update = LocationUpdate(
            location_id="loc-001",
            label="??",
            confidence=0.2,
            is_stable=False,
            uncertainty_reason="Ambiguous visual match",
        )
        
        assert update.is_stable is False
        assert update.uncertainty_reason is not None

    def test_error_with_empty_details(self):
        """Error without extra details."""
        error = ProtocolError(
            severity=ErrorSeverity.WARNING,
            code="SIMPLE",
            message="Simple warning",
        )
        
        assert error.details == {}
        assert error.related_message_id is None


# =============================================================================
# INTEGRATION SCENARIO TESTS
# =============================================================================


class TestIntegrationScenarios:
    """Tests for realistic protocol usage scenarios."""

    def test_sensor_connection_flow(self):
        """Complete sensor connection handshake flow."""
        # 1. Sensor sends handshake
        sensor_handshake = Handshake(
            role="sensor",
            identity="unity-sim-001",
        )
        
        # 2. Backend responds with handshake
        backend_handshake = Handshake(
            role="backend",
            identity="episodic-agent",
            session_id=str(uuid.uuid4()),
        )
        
        # 3. Sensor sends capabilities
        capabilities = CapabilitiesReport(
            sensor_id="unity-sim-001",
            sensor_type="camera",
            capabilities=[
                SensorCapability.RGB_CAMERA,
                SensorCapability.BOUNDING_BOXES,
            ],
            supports_visual_channel=True,
        )
        
        # 4. Backend sends stream control
        control = StreamControl(command=StreamControlCommand.START)
        
        # Verify complete flow
        assert sensor_handshake.role == "sensor"
        assert backend_handshake.session_id is not None
        assert len(capabilities.capabilities) > 0
        assert control.command == StreamControlCommand.START

    def test_label_acquisition_flow(self):
        """Complete label acquisition flow."""
        # 1. Backend detects unknown entity
        entity_id = "entity-" + str(uuid.uuid4())[:8]
        
        # 2. Backend sends label request
        request = LabelRequest(
            request_id="req-" + str(uuid.uuid4())[:8],
            target_type=LabelTargetType.ENTITY,
            target_id=entity_id,
            confidence=LabelConfidence.LOW,
            description="What is this object?",
        )
        
        # 3. User provides label
        response = LabelResponse(
            request_id=request.request_id,
            response_type=LabelResponseType.PROVIDED,
            label="coffee mug",
            confidence=1.0,
            response_time_ms=3500.0,
        )
        
        # 4. Backend sends entity update
        update = EntityUpdate(
            entity_id=entity_id,
            label=response.label,
            labels=[response.label],
            confidence=response.confidence,
        )
        
        # Verify flow
        assert request.request_id == response.request_id
        assert update.label == "coffee mug"

    def test_error_recovery_flow(self):
        """Error handling and recovery flow."""
        # 1. Backend requests unsupported feature
        control = StreamControl(
            command=StreamControlCommand.SET_RESOLUTION,
            resolution=(4096, 2160),  # 4K might not be supported
            request_id="cmd-001",
        )
        
        # 2. Sensor reports error
        error = ProtocolError(
            severity=ErrorSeverity.ERROR,
            code="RESOLUTION_NOT_SUPPORTED",
            message="Maximum resolution is 1920x1080",
            related_message_id=control.request_id,
            details={"max_width": 1920, "max_height": 1080},
            recoverable=True,
            suggested_action="Use 1920x1080 or lower",
        )
        
        # 3. Backend adjusts request
        retry = StreamControl(
            command=StreamControlCommand.SET_RESOLUTION,
            resolution=(1920, 1080),
            request_id="cmd-002",
        )
        
        # Verify flow
        assert error.recoverable is True
        assert "1920" in error.suggested_action
        assert retry.resolution[0] <= 1920


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
