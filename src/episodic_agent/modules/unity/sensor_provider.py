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

logger = logging.getLogger(__name__)

# Protocol version we expect from Unity
EXPECTED_PROTOCOL_VERSION = "1.0.0"
# Only truly required fields - room/location info is optional since real-world sensors
# may not know the current room (agent must discover/learn rooms from sensor data)
REQUIRED_FIELDS = ["protocol_version", "frame_id", "timestamp"]
# Optional fields that enhance the frame but aren't required for basic operation
OPTIONAL_FIELDS = ["current_room", "current_room_guid", "current_room_label", "entities", "camera_pose"]


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
        """
        self._ws_url = ws_url
        self._buffer_size = buffer_size
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_attempts = max_reconnect_attempts
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._validate_protocol = validate_protocol
        
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
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                self._handle_message(message)
            except asyncio.TimeoutError:
                # Check if we should stop
                continue
            except Exception as e:
                logger.warning(f"Receive error: {e}")
                break

    def _handle_message(self, message: str) -> None:
        """Parse and validate a received message, queue the frame."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return
        
        # Only handle sensor frames (have frame_id)
        if "frame_id" not in data:
            # Might be a command response, ignore
            return
        
        # Validate required fields
        if not self._validate_frame_fields(data):
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
        # Parse timestamp
        timestamp = datetime.fromtimestamp(data.get("timestamp", time.time()))
        
        # Extract camera pose for raw_data
        camera_pose = data.get("camera_pose", data.get("camera", {}))
        
        # Build raw_data with core sensor info
        raw_data = {
            "camera_position": camera_pose.get("position"),
            "camera_rotation": camera_pose.get("rotation"),
            "camera_forward": camera_pose.get("forward"),
        }
        
        # Handle room info - support both field naming conventions
        # Unity sends: current_room_guid, current_room_label
        # Also support: current_room (for backward compatibility)
        # Note: Room info is OPTIONAL - real-world sensors may not know the current room
        current_room_guid = data.get("current_room_guid") or data.get("current_room")
        current_room_label = data.get("current_room_label")
        
        # Store sensor data in extras for perception module to process
        # The perception module should use these as observations, not as ground truth
        extras = {
            "protocol_version": data.get("protocol_version"),
            # Room observation (if available from sensor) - agent must verify/learn this
            "current_room_guid": current_room_guid,  
            "current_room_label": current_room_label,
            # Legacy field name for backward compatibility
            "current_room": current_room_guid,
            # Entity observations (positions, states) - not identities
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
            Dict with connection state, frame info, etc.
        """
        with self._buffer_lock:
            buffer_count = len(self._frame_buffer)
        
        return {
            "state": self.state,
            "ws_url": self._ws_url,
            "last_frame_id": self._last_frame_id,
            "last_frame_time": self._last_frame_time.isoformat() if self._last_frame_time else None,
            "dropped_frames": self._dropped_frames,
            "reconnect_count": self._reconnect_count,
            "buffer_count": buffer_count,
            "buffer_size": self._buffer_size,
        }

    def __enter__(self):
        """Context manager entry - start connection."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop connection."""
        self.stop()
        return False
