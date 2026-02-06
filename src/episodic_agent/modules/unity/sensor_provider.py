"""Unity WebSocket sensor provider for real-time frame streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable

from episodic_agent.core.interfaces import SensorProvider
from episodic_agent.schemas import SensorFrame

# Optional gateway integration for validation/preprocessing
try:
    from episodic_agent.modules.sensor_gateway import (
        SensorGateway,
        SensorType,
        SensorMessage,
    )
    GATEWAY_AVAILABLE = True
except ImportError:
    GATEWAY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Separate logger for detailed data logging (can be enabled independently)
data_logger = logging.getLogger(f"{__name__}.data")

# Protocol version we expect from Unity
EXPECTED_PROTOCOL_VERSION = "1.0.0"
# Only truly required fields - room/location info is optional since real-world sensors
# may not know the current room (agent must discover/learn rooms from sensor data)
REQUIRED_FIELDS = ["protocol_version", "frame_id", "timestamp"]
# Optional fields that enhance the frame but aren't required for basic operation
# REMOVED: current_room_label - backend owns all semantic labels, Unity only sends GUIDs
OPTIONAL_FIELDS = ["current_room", "current_room_guid", "entities", "camera_pose"]


class ConnectionState:
    """Track connection state for the sensor provider."""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class UnityWebSocketSensorProvider(SensorProvider):
    """WebSocket sensor provider that connects to Unity's sensor stream.
    
    Features:
    - Automatic reconnection on disconnect
    - Graceful handling of dropped frames
    - Protocol version validation
    - Frame buffering with overflow protection
    - Thread-safe frame queue for sync access from agent loop
    - Optional SensorGateway integration for validation
    - Detailed data logging for debugging
    
    The provider runs the WebSocket client in a background thread,
    exposing a synchronous get_frame() interface for the orchestrator.
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8765",
        buffer_size: int = 100,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = -1,  # -1 = infinite
        on_connect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        validate_protocol: bool = True,
        use_gateway: bool = True,  # Enable gateway validation
        log_raw_data: bool = False,  # Log raw Unity data for debugging
        on_validation_error: Callable[[str], None] | None = None,  # Error callback
    ) -> None:
        """Initialize the Unity WebSocket sensor provider.
        
        Args:
            ws_url: WebSocket URL to connect to (ws://host:port).
            buffer_size: Maximum frames to buffer (oldest dropped on overflow).
            reconnect_delay: Seconds to wait between reconnect attempts.
            max_reconnect_attempts: Max reconnects (-1 for infinite).
            on_connect: Optional callback when connected.
            on_disconnect: Optional callback when disconnected.
            validate_protocol: Whether to validate protocol version.
            use_gateway: Whether to use SensorGateway for validation.
            log_raw_data: Whether to log raw Unity data (verbose).
            on_validation_error: Callback for validation errors.
        """
        self._ws_url = ws_url
        self._buffer_size = buffer_size
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._validate_protocol = validate_protocol
        self._log_raw_data = log_raw_data
        self._on_validation_error = on_validation_error
        
        # Thread-safe frame buffer
        self._frame_buffer: deque[SensorFrame] = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()
        
        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._last_frame_id: int | None = None
        self._last_frame_time: datetime | None = None
        self._dropped_frames = 0
        self._reconnect_count = 0
        
        # Background thread management
        self._running = False
        self._ws_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        # Frame availability event (for blocking get_frame)
        self._frame_available = threading.Event()
        
        # Initialize gateway for validation if available and enabled
        self._gateway: SensorGateway | None = None
        self._use_gateway = use_gateway
        if use_gateway and GATEWAY_AVAILABLE:
            self._gateway = SensorGateway(
                on_user_notification=self._handle_gateway_notification,
                log_raw_data=log_raw_data,
            )
            self._gateway.register_sensor(
                "unity_main",
                SensorType.UNITY_WEBSOCKET,
            )
            logger.info("SensorGateway initialized for Unity data validation")
        elif use_gateway and not GATEWAY_AVAILABLE:
            logger.warning("SensorGateway requested but not available, using basic validation")
        
        # Data logging state
        self._frames_logged = 0
        self._validation_errors = 0

    @property
    def state(self) -> str:
        """Get current connection state."""
        with self._state_lock:
            return self._state

    @property
    def last_frame_id(self) -> int | None:
        """Get the ID of the last received frame."""
        return self._last_frame_id

    @property
    def last_frame_time(self) -> datetime | None:
        """Get the timestamp of the last received frame."""
        return self._last_frame_time

    @property
    def dropped_frames(self) -> int:
        """Get count of dropped frames (buffer overflow or gaps)."""
        return self._dropped_frames

    @property
    def reconnect_count(self) -> int:
        """Get count of reconnection attempts."""
        return self._reconnect_count

    @property
    def validation_errors(self) -> int:
        """Get count of validation errors."""
        return self._validation_errors

    @property
    def gateway(self) -> "SensorGateway | None":
        """Get the sensor gateway (if enabled)."""
        return self._gateway

    def _handle_gateway_notification(self, severity: str, message: str) -> None:
        """Handle notifications from the gateway.
        
        Args:
            severity: Notification severity level.
            message: Notification message.
        """
        if severity in ("CRITICAL", "ERROR"):
            logger.error(f"[GATEWAY] {message}")
            if self._on_validation_error:
                try:
                    self._on_validation_error(message)
                except Exception as e:
                    logger.warning(f"Validation error callback failed: {e}")
        elif severity == "WARNING":
            logger.warning(f"[GATEWAY] {message}")
        else:
            logger.info(f"[GATEWAY] {message}")

    def start(self) -> None:
        """Start the WebSocket connection in a background thread."""
        if self._running:
            return
            
        self._running = True
        self._stop_event.clear()
        self._ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
        self._ws_thread.start()
        logger.info(f"Unity sensor provider starting, connecting to {self._ws_url}")

    def stop(self) -> None:
        """Stop the WebSocket connection and background thread."""
        self._running = False
        self._stop_event.set()
        
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
        
        self._ws_thread = None
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("Unity sensor provider stopped")

    def _set_state(self, state: str) -> None:
        """Set connection state thread-safely."""
        with self._state_lock:
            old_state = self._state
            self._state = state
        
        if old_state != state:
            logger.info(f"Connection state: {old_state} -> {state}")

    def _run_ws_loop(self) -> None:
        """Background thread: run the asyncio WebSocket event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._ws_connect_loop())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            loop.close()

    async def _ws_connect_loop(self) -> None:
        """Async connection loop with reconnection logic."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets library required: pip install websockets")
            self._set_state(ConnectionState.DISCONNECTED)
            return
        
        attempt = 0
        
        while self._running and not self._stop_event.is_set():
            try:
                self._set_state(ConnectionState.CONNECTING if attempt == 0 
                               else ConnectionState.RECONNECTING)
                
                async with websockets.connect(self._ws_url) as ws:
                    self._set_state(ConnectionState.CONNECTED)
                    attempt = 0  # Reset on successful connect
                    
                    if self._on_connect:
                        try:
                            self._on_connect()
                        except Exception as e:
                            logger.warning(f"on_connect callback error: {e}")
                    
                    await self._receive_loop(ws)
                    
            except Exception as e:
                logger.warning(f"WebSocket connection error: {e}")
                self._set_state(ConnectionState.DISCONNECTED)
                
                if self._on_disconnect:
                    try:
                        self._on_disconnect()
                    except Exception as e2:
                        logger.warning(f"on_disconnect callback error: {e2}")
                
                # Check reconnect limits
                attempt += 1
                self._reconnect_count += 1
                
                if self._max_reconnect_attempts >= 0 and attempt > self._max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts ({self._max_reconnect_attempts}) exceeded")
                    break
                
                if self._running and not self._stop_event.is_set():
                    logger.info(f"Reconnecting in {self._reconnect_delay}s (attempt {attempt})...")
                    await asyncio.sleep(self._reconnect_delay)

    async def _receive_loop(self, ws) -> None:
        """Receive messages from WebSocket and buffer frames."""
        while self._running and not self._stop_event.is_set():
            try:
                # Use a longer timeout and handle gracefully
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                self._handle_message(message)
            except asyncio.TimeoutError:
                # Check if we should stop
                continue
            except TypeError as e:
                # TypeError during receive - log details and retry
                logger.warning(f"WebSocket receive TypeError: {e}")
                data_logger.error(f"[WS] TypeError in receive: {e}")
                # Try to continue without explicit timeout
                try:
                    message = await ws.recv()
                    self._handle_message(message)
                except Exception as inner_e:
                    logger.warning(f"Receive error after retry: {inner_e}")
                    break
            except Exception as e:
                logger.warning(f"Receive error: {e}")
                # Log the actual exception type for debugging
                data_logger.debug(f"Receive exception details: {type(e).__name__}: {e}")
                break

    def _handle_message(self, message: str) -> None:
        """Parse and validate a received message, queue the frame.
        
        Uses SensorGateway for validation if available, otherwise
        falls back to basic validation.
        """
        # Log raw incoming data if enabled
        if self._log_raw_data:
            self._frames_logged += 1
            data_logger.info(f"[UNITY→PYTHON] Frame {self._frames_logged}: {message[:500]}{'...' if len(message) > 500 else ''}")
        
        # Parse JSON
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            data_logger.error(f"[UNITY→PYTHON] Invalid JSON: {message[:200]}")
            self._validation_errors += 1
            return
        
        # Only handle sensor frames (have frame_id)
        if "frame_id" not in data:
            # Might be a command response, log for debugging
            data_logger.debug(f"[UNITY→PYTHON] Non-frame message: {list(data.keys())}")
            return
        
        # Log frame summary
        # ARCHITECTURAL INVARIANT: room label comes from backend, Unity sends GUID only
        frame_id = data.get("frame_id", "?")
        room_guid = data.get("current_room_guid") or data.get("current_room", "unknown")
        entities_count = len(data.get("entities", []))
        state_changes = len(data.get("state_changes", []))
        
        data_logger.debug(
            f"[UNITY→PYTHON] Frame #{frame_id}: room_guid={room_guid}, "
            f"entities={entities_count}, state_changes={state_changes}"
        )
        
        # Use gateway for validation if available
        if self._gateway:
            gateway_message = self._gateway.process(data, sensor_id="unity_main")
            
            if gateway_message.validation and not gateway_message.validation.is_valid:
                self._validation_errors += 1
                # Log validation errors
                for error in gateway_message.validation.errors:
                    data_logger.warning(f"[VALIDATION] {error.code}: {error.message}")
                
                # Use corrected data if available
                if gateway_message.validation.corrected_data:
                    data = gateway_message.validation.corrected_data
                    data_logger.info("[VALIDATION] Using auto-corrected data")
                else:
                    # Skip frame if validation failed with no correction
                    logger.warning(f"Skipping frame {frame_id} due to validation errors")
                    return
        else:
            # Fall back to basic validation
            if not self._validate_frame_fields(data):
                self._validation_errors += 1
                return
        
        # Validate protocol version
        if self._validate_protocol:
            protocol = data.get("protocol_version", "")
            if protocol != EXPECTED_PROTOCOL_VERSION:
                logger.warning(f"Protocol version mismatch: expected {EXPECTED_PROTOCOL_VERSION}, got {protocol}")
                # Continue anyway but log warning
        
        # Check for dropped frames (gaps in frame_id)
        frame_id = data["frame_id"]
        if self._last_frame_id is not None:
            gap = frame_id - self._last_frame_id - 1
            if gap > 0:
                self._dropped_frames += gap
                logger.debug(f"Frame gap detected: {gap} frames dropped")
        
        # Convert to SensorFrame
        sensor_frame = self._convert_to_sensor_frame(data)
        
        # Update tracking
        self._last_frame_id = frame_id
        self._last_frame_time = sensor_frame.timestamp
        
        # Add to buffer (thread-safe)
        with self._buffer_lock:
            if len(self._frame_buffer) == self._buffer_size:
                self._dropped_frames += 1  # About to drop oldest
            self._frame_buffer.append(sensor_frame)
        
        # Signal frame available
        self._frame_available.set()

    def _validate_frame_fields(self, data: dict[str, Any]) -> bool:
        """Validate that required fields are present."""
        missing = [f for f in REQUIRED_FIELDS if f not in data]
        if missing:
            logger.warning(f"Frame missing required fields: {missing}")
            return False
        return True

    def _convert_to_sensor_frame(self, data: dict[str, Any]) -> SensorFrame:
        """Convert Unity JSON frame to SensorFrame contract.
        
        Maps Unity fields to the standard SensorFrame, storing Unity-specific
        data in extras for the perception module.
        
        Note: Room/location information is optional. In real-world scenarios,
        the agent must discover and learn room boundaries from sensor patterns,
        not rely on pre-defined room labels. The model's view of the world
        should only be updated through sensor observations and user input,
        never through hardcoded IDs or labels.
        """
        # Parse timestamp - Unity sends ISO 8601 string, but also handle Unix float
        timestamp_val = data.get("timestamp")
        if isinstance(timestamp_val, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_val)
            except (ValueError, TypeError):
                timestamp = datetime.now()
        elif isinstance(timestamp_val, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp_val)
        else:
            timestamp = datetime.now()
        
        # Extract camera pose for raw_data
        camera_pose = data.get("camera_pose", data.get("camera", {}))
        
        # Build raw_data with core sensor info
        raw_data = {
            "camera_position": camera_pose.get("position"),
            "camera_rotation": camera_pose.get("rotation"),
            "camera_forward": camera_pose.get("forward"),
        }
        
        # Handle room info - support both field naming conventions
        # Unity sends: current_room_guid (GUID only, no label - backend owns labels)
        # Also support: current_room (for backward compatibility)
        # Note: Room info is OPTIONAL - real-world sensors may not know the current room
        current_room_guid = data.get("current_room_guid") or data.get("current_room")
        # REMOVED: current_room_label - backend learns labels from user interaction
        
        # Store sensor data in extras for perception module to process
        # The perception module should use these as observations, not as ground truth
        # ARCHITECTURAL INVARIANT: Unity provides GUIDs and observables only, no semantic labels
        extras = {
            "protocol_version": data.get("protocol_version"),
            # Room observation (GUID only) - agent must learn/verify label from user
            "current_room_guid": current_room_guid,  
            # Legacy field name for backward compatibility
            "current_room": current_room_guid,
            # Entity observations (positions, states) - no labels, backend learns them
            "entities": data.get("entities", []),
            "state_changes": data.get("state_changes", []),
            "camera_pose": camera_pose,
        }
        
        return SensorFrame(
            frame_id=data["frame_id"],
            timestamp=timestamp,
            raw_data=raw_data,
            sensor_type="unity_websocket",
            extras=extras,
        )

    # =========================================================================
    # SensorProvider interface
    # =========================================================================

    def get_frame(self, timeout: float = 5.0) -> SensorFrame:
        """Get the next sensor frame (blocking).
        
        Args:
            timeout: Maximum time to wait for a frame.
            
        Returns:
            The next available SensorFrame.
            
        Raises:
            TimeoutError: If no frame available within timeout.
            RuntimeError: If provider not started.
        """
        if not self._running:
            # Auto-start for convenience
            self.start()
            # Wait a bit for initial connection
            time.sleep(0.5)
        
        start_time = time.perf_counter()
        
        while True:
            # Check buffer
            with self._buffer_lock:
                if self._frame_buffer:
                    return self._frame_buffer.popleft()
            
            # Wait for frame with remaining timeout
            elapsed = time.perf_counter() - start_time
            remaining = timeout - elapsed
            
            if remaining <= 0:
                raise TimeoutError(f"No frame received within {timeout}s")
            
            # Wait for signal or timeout
            self._frame_available.clear()
            if not self._frame_available.wait(timeout=min(remaining, 0.5)):
                # Check if we should keep waiting
                if time.perf_counter() - start_time >= timeout:
                    raise TimeoutError(f"No frame received within {timeout}s")

    def has_frames(self) -> bool:
        """Check if more frames are available.
        
        For a live WebSocket stream, this returns True as long as
        we're connected or have buffered frames.
        """
        with self._buffer_lock:
            if self._frame_buffer:
                return True
        
        return self.state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING, 
                              ConnectionState.RECONNECTING)

    def reset(self) -> None:
        """Reset the sensor provider (clear buffer, restart connection)."""
        with self._buffer_lock:
            self._frame_buffer.clear()
        
        self._last_frame_id = None
        self._last_frame_time = None
        self._dropped_frames = 0
        
        # Restart connection
        was_running = self._running
        if was_running:
            self.stop()
            self.start()

    def get_status(self) -> dict[str, Any]:
        """Get detailed connection status for display.
        
        Returns:
            Dict with connection state, frame info, validation stats, etc.
        """
        with self._buffer_lock:
            buffer_count = len(self._frame_buffer)
        
        status = {
            "state": self.state,
            "ws_url": self._ws_url,
            "last_frame_id": self._last_frame_id,
            "last_frame_time": self._last_frame_time.isoformat() if self._last_frame_time else None,
            "dropped_frames": self._dropped_frames,
            "reconnect_count": self._reconnect_count,
            "buffer_count": buffer_count,
            "buffer_size": self._buffer_size,
            "validation_errors": self._validation_errors,
            "gateway_enabled": self._gateway is not None,
        }
        
        # Add gateway stats if available
        if self._gateway:
            gateway_stats = self._gateway.get_stats()
            status["gateway_stats"] = {
                "total_messages": gateway_stats.get("total_messages", 0),
                "total_errors": gateway_stats.get("total_errors", 0),
                "error_rate": gateway_stats.get("error_rate", 0.0),
            }
        
        return status

    def get_recent_errors(self, count: int = 10) -> list[dict]:
        """Get recent validation errors for debugging.
        
        Args:
            count: Maximum errors to return.
            
        Returns:
            List of recent validation errors.
        """
        if not self._gateway:
            return []
        
        errors = self._gateway.get_error_history("unity_main")
        return [
            {
                "code": e.code,
                "message": e.message,
                "severity": e.severity.value,
                "field": e.field_path,
                "suggestion": e.suggestion,
            }
            for e in errors[-count:]
        ]

    def __enter__(self):
        """Context manager entry - start connection."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop connection."""
        self.stop()
        return False
