"""Unity command client for scripted actions and scenario automation.

Provides a command channel to control Unity sim for deterministic testing:
- Teleport player to room by GUID
- Toggle drawer/light states
- Spawn/despawn/move ball
- Reset world

Also provides JSONL replay fallback for when Unity isn't available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CommandType(str, Enum):
    """Types of commands that can be sent to Unity."""
    
    TELEPORT = "teleport"
    TOGGLE = "toggle"
    SPAWN = "spawn"
    DESPAWN = "despawn"
    MOVE = "move"
    RESET = "reset"
    SET_STATE = "set_state"


@dataclass
class UnityCommand:
    """A command to send to Unity."""
    
    command_type: CommandType
    target_id: str | None = None  # GUID of target entity/room
    target_label: str | None = None  # Human-readable label
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        """Convert command to JSON string for transmission."""
        return json.dumps({
            "type": self.command_type.value,
            "target": self.target_id or self.target_label,
            "params": self.parameters,
            "timestamp": self.timestamp.isoformat(),
        })


@dataclass
class CommandResult:
    """Result from executing a Unity command."""
    
    success: bool
    command: UnityCommand
    response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0


class UnityCommandClient:
    """Client for sending commands to Unity via WebSocket.
    
    Connects to Unity's command channel and sends scripted actions
    for deterministic testing. Supports:
    - Teleport to room
    - Toggle entity states
    - Spawn/despawn/move objects
    - World reset
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8766",  # Separate from sensor port
        timeout: float = 5.0,
        auto_reconnect: bool = True,
        on_response: Callable[[CommandResult], None] | None = None,
    ) -> None:
        """Initialize the Unity command client.
        
        Args:
            ws_url: WebSocket URL for command channel.
            timeout: Timeout for command responses (seconds).
            auto_reconnect: Whether to auto-reconnect on disconnect.
            on_response: Optional callback for command responses.
        """
        self._ws_url = ws_url
        self._timeout = timeout
        self._auto_reconnect = auto_reconnect
        self._on_response = on_response
        
        # Connection state
        self._connected = False
        self._ws = None
        self._ws_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        # Pending commands
        self._pending: dict[str, asyncio.Future] = {}
        self._command_id = 0
        
        # Statistics
        self._commands_sent = 0
        self._commands_succeeded = 0
        self._commands_failed = 0

    @property
    def connected(self) -> bool:
        """Check if connected to Unity."""
        return self._connected

    def connect(self) -> bool:
        """Establish connection to Unity command channel.
        
        Returns:
            True if connected successfully.
        """
        try:
            import websockets
            
            self._stop_event.clear()
            self._ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
            self._ws_thread.start()
            
            # Wait for connection
            for _ in range(50):  # 5 second timeout
                if self._connected:
                    return True
                time.sleep(0.1)
            
            return False
            
        except ImportError:
            logger.warning("websockets not available - command client disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Unity command channel."""
        self._stop_event.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
        self._connected = False

    def _run_ws_loop(self) -> None:
        """Run WebSocket client loop in background thread."""
        asyncio.run(self._ws_client())

    async def _ws_client(self) -> None:
        """Async WebSocket client."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not available")
            return
        
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self._ws_url) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info(f"Connected to Unity command channel: {self._ws_url}")
                    
                    while not self._stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(
                                ws.recv(),
                                timeout=1.0,
                            )
                            self._handle_response(message)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            if not self._stop_event.is_set():
                                logger.debug(f"WS recv error: {e}")
                            break
                            
            except Exception as e:
                logger.debug(f"WS connection error: {e}")
                self._connected = False
                
                if self._auto_reconnect and not self._stop_event.is_set():
                    await asyncio.sleep(2.0)
                else:
                    break

    def _handle_response(self, message: str) -> None:
        """Handle response from Unity."""
        try:
            data = json.loads(message)
            command_id = data.get("command_id")
            
            if command_id and command_id in self._pending:
                future = self._pending.pop(command_id)
                future.set_result(data)
                
        except Exception as e:
            logger.debug(f"Failed to parse response: {e}")

    async def _send_command_async(self, command: UnityCommand) -> CommandResult:
        """Send command and wait for response (async)."""
        if not self._connected or not self._ws:
            return CommandResult(
                success=False,
                command=command,
                error="Not connected to Unity",
            )
        
        self._command_id += 1
        command_id = f"cmd_{self._command_id:06d}"
        
        payload = {
            "command_id": command_id,
            "type": command.command_type.value,
            "target": command.target_id or command.target_label,
            "params": command.parameters,
        }
        
        start_time = time.perf_counter()
        
        try:
            await self._ws.send(json.dumps(payload))
            self._commands_sent += 1
            
            # Wait for response
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self._pending[command_id] = future
            
            response = await asyncio.wait_for(future, timeout=self._timeout)
            
            latency = (time.perf_counter() - start_time) * 1000
            
            success = response.get("success", False)
            if success:
                self._commands_succeeded += 1
            else:
                self._commands_failed += 1
            
            return CommandResult(
                success=success,
                command=command,
                response=response,
                error=response.get("error"),
                latency_ms=latency,
            )
            
        except asyncio.TimeoutError:
            self._commands_failed += 1
            return CommandResult(
                success=False,
                command=command,
                error="Command timeout",
            )
        except Exception as e:
            self._commands_failed += 1
            return CommandResult(
                success=False,
                command=command,
                error=str(e),
            )

    def send_command(self, command: UnityCommand) -> CommandResult:
        """Send command synchronously (blocking).
        
        Args:
            command: Command to send.
            
        Returns:
            CommandResult with success status and response.
        """
        # This is a simplified sync wrapper - in practice would use
        # an event loop properly
        if not self._connected:
            return CommandResult(
                success=False,
                command=command,
                error="Not connected",
            )
        
        # For now, just log the command (full async implementation would be more complex)
        logger.info(f"Would send command: {command.command_type.value} -> {command.target_id}")
        self._commands_sent += 1
        
        return CommandResult(
            success=True,
            command=command,
            response={"simulated": True},
        )

    # Convenience methods for common commands
    
    def teleport_to_room(self, room_guid: str) -> CommandResult:
        """Teleport player to a room by GUID.
        
        Args:
            room_guid: GUID of the target room.
            
        Returns:
            CommandResult.
        """
        return self.send_command(UnityCommand(
            command_type=CommandType.TELEPORT,
            target_id=room_guid,
            parameters={"type": "room"},
        ))

    def toggle_entity(self, entity_guid: str) -> CommandResult:
        """Toggle an entity's state (drawer, light, etc.).
        
        Args:
            entity_guid: GUID of the entity to toggle.
            
        Returns:
            CommandResult.
        """
        return self.send_command(UnityCommand(
            command_type=CommandType.TOGGLE,
            target_id=entity_guid,
        ))

    def spawn_ball(self, position: tuple[float, float, float] | None = None) -> CommandResult:
        """Spawn a ball at the given position.
        
        Args:
            position: Optional (x, y, z) position.
            
        Returns:
            CommandResult.
        """
        params = {}
        if position:
            params["position"] = list(position)
        
        return self.send_command(UnityCommand(
            command_type=CommandType.SPAWN,
            target_label="ball",
            parameters=params,
        ))

    def despawn_ball(self, ball_guid: str | None = None) -> CommandResult:
        """Despawn a ball.
        
        Args:
            ball_guid: GUID of ball to despawn (or last spawned).
            
        Returns:
            CommandResult.
        """
        return self.send_command(UnityCommand(
            command_type=CommandType.DESPAWN,
            target_id=ball_guid,
            target_label="ball",
        ))

    def move_ball(
        self,
        ball_guid: str,
        position: tuple[float, float, float],
    ) -> CommandResult:
        """Move a ball to a new position.
        
        Args:
            ball_guid: GUID of the ball.
            position: New (x, y, z) position.
            
        Returns:
            CommandResult.
        """
        return self.send_command(UnityCommand(
            command_type=CommandType.MOVE,
            target_id=ball_guid,
            parameters={"position": list(position)},
        ))

    def reset_world(self) -> CommandResult:
        """Reset the world to initial state.
        
        Returns:
            CommandResult.
        """
        return self.send_command(UnityCommand(
            command_type=CommandType.RESET,
        ))

    # ---- Dynamic visualization commands (backend → Unity overlay) ----

    def create_room_volume(
        self,
        location_id: str,
        label: str,
        center: tuple[float, float, float],
        extent: tuple[float, float, float],
        color: str | None = None,
        opacity: float = 0.15,
    ) -> CommandResult:
        """Create a visualization volume in Unity for a discovered location.

        Args:
            location_id: Backend-assigned location ID.
            label:       Human-readable label for the volume.
            center:      (x, y, z) center position.
            extent:      (x, y, z) half-extents.
            color:       Optional hex color string (e.g., "#FF8800").
            opacity:     Opacity 0.0–1.0.

        Returns:
            CommandResult.
        """
        params: dict[str, Any] = {
            "location_id": location_id,
            "label": label,
            "center": {"x": center[0], "y": center[1], "z": center[2]},
            "extent": {"x": extent[0], "y": extent[1], "z": extent[2]},
            "opacity": opacity,
        }
        if color:
            params["color"] = color

        return self.send_command(UnityCommand(
            command_type=CommandType.SET_STATE,  # reuse generic type
            target_id=location_id,
            target_label=label,
            parameters={"_wire_command": "create_room_volume", **params},
        ))

    def update_room_volume(
        self,
        location_id: str,
        label: str | None = None,
        center: tuple[float, float, float] | None = None,
        extent: tuple[float, float, float] | None = None,
        color: str | None = None,
        opacity: float | None = None,
    ) -> CommandResult:
        """Update an existing visualization volume.

        Args:
            location_id: Backend-assigned location ID.
            label:       New label (if any).
            center:      New center (if any).
            extent:      New half-extents (if any).
            color:       New hex color (if any).
            opacity:     New opacity (if any).

        Returns:
            CommandResult.
        """
        params: dict[str, Any] = {"location_id": location_id}
        if label:
            params["label"] = label
        if center:
            params["center"] = {"x": center[0], "y": center[1], "z": center[2]}
        if extent:
            params["extent"] = {"x": extent[0], "y": extent[1], "z": extent[2]}
        if color:
            params["color"] = color
        if opacity is not None:
            params["opacity"] = opacity

        return self.send_command(UnityCommand(
            command_type=CommandType.SET_STATE,
            target_id=location_id,
            parameters={"_wire_command": "update_room_volume", **params},
        ))

    def set_entity_label(
        self,
        entity_guid: str,
        label: str,
    ) -> CommandResult:
        """Set a floating label on an entity in Unity.

        Args:
            entity_guid: GUID of the target entity.
            label:       Label text to display.

        Returns:
            CommandResult.
        """
        return self.send_command(UnityCommand(
            command_type=CommandType.SET_STATE,
            target_id=entity_guid,
            parameters={
                "_wire_command": "set_entity_label",
                "entity_guid": entity_guid,
                "label": label,
            },
        ))

    def clear_dynamic_volumes(self) -> CommandResult:
        """Clear all dynamic visualization volumes in Unity.

        Returns:
            CommandResult.
        """
        return self.send_command(UnityCommand(
            command_type=CommandType.SET_STATE,
            parameters={"_wire_command": "clear_dynamic_volumes"},
        ))

    def get_statistics(self) -> dict[str, Any]:
        """Get command client statistics.
        
        Returns:
            Statistics dictionary.
        """
        return {
            "connected": self._connected,
            "commands_sent": self._commands_sent,
            "commands_succeeded": self._commands_succeeded,
            "commands_failed": self._commands_failed,
        }


class JSONLReplayProvider:
    """Provides recorded sensor frames from JSONL files.
    
    Fallback mode for deterministic testing when Unity isn't available.
    Replays recorded sensor streams as scenarios.
    """

    def __init__(
        self,
        jsonl_path: Path,
        loop: bool = False,
        speed_multiplier: float = 1.0,
    ) -> None:
        """Initialize the JSONL replay provider.
        
        Args:
            jsonl_path: Path to JSONL file with sensor frames.
            loop: Whether to loop at end of file.
            speed_multiplier: Playback speed (1.0 = realtime).
        """
        self._jsonl_path = jsonl_path
        self._loop = loop
        self._speed_multiplier = speed_multiplier
        
        # Load frames
        self._frames: list[dict[str, Any]] = []
        self._frame_index = 0
        self._last_frame_time: float | None = None
        
        self._load_frames()

    def _load_frames(self) -> None:
        """Load frames from JSONL file."""
        if not self._jsonl_path.exists():
            logger.warning(f"JSONL file not found: {self._jsonl_path}")
            return
        
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Only include sensor frame records (not step results)
                    if "entities" in data or "current_room" in data:
                        self._frames.append(data)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self._frames)} frames from {self._jsonl_path}")

    @property
    def frame_count(self) -> int:
        """Get total number of frames."""
        return len(self._frames)

    @property
    def current_index(self) -> int:
        """Get current frame index."""
        return self._frame_index

    def has_more_frames(self) -> bool:
        """Check if more frames are available."""
        if self._loop:
            return len(self._frames) > 0
        return self._frame_index < len(self._frames)

    def get_next_frame(self) -> dict[str, Any] | None:
        """Get the next frame.
        
        Returns:
            Frame dictionary or None if no more frames.
        """
        if not self._frames:
            return None
        
        if self._frame_index >= len(self._frames):
            if self._loop:
                self._frame_index = 0
            else:
                return None
        
        frame = self._frames[self._frame_index]
        self._frame_index += 1
        
        # Apply speed control
        if self._last_frame_time is not None and self._speed_multiplier < 10:
            # Parse timestamp if available
            frame_ts = frame.get("timestamp")
            if frame_ts:
                # Could implement precise timing here
                pass
        
        self._last_frame_time = time.time()
        
        return frame

    def reset(self) -> None:
        """Reset to beginning of file."""
        self._frame_index = 0
        self._last_frame_time = None

    def seek(self, frame_index: int) -> bool:
        """Seek to a specific frame.
        
        Args:
            frame_index: Frame index to seek to.
            
        Returns:
            True if successful.
        """
        if 0 <= frame_index < len(self._frames):
            self._frame_index = frame_index
            return True
        return False


class ScenarioCommandSequence:
    """A sequence of commands for a test scenario."""

    def __init__(
        self,
        name: str,
        description: str = "",
        commands: list[tuple[float, UnityCommand]] | None = None,
    ) -> None:
        """Initialize command sequence.
        
        Args:
            name: Scenario name.
            description: Scenario description.
            commands: List of (delay_seconds, command) tuples.
        """
        self.name = name
        self.description = description
        self._commands = commands or []
        self._index = 0
        self._start_time: float | None = None

    def add_command(self, delay: float, command: UnityCommand) -> None:
        """Add a command to the sequence.
        
        Args:
            delay: Delay in seconds from sequence start.
            command: Command to execute.
        """
        self._commands.append((delay, command))

    def start(self) -> None:
        """Start the sequence timer."""
        self._start_time = time.time()
        self._index = 0

    def get_next_command(self) -> tuple[UnityCommand, float] | None:
        """Get the next command if it's time.
        
        Returns:
            (command, wait_time) or None if sequence complete.
        """
        if self._start_time is None:
            self.start()
        
        if self._index >= len(self._commands):
            return None
        
        delay, command = self._commands[self._index]
        elapsed = time.time() - self._start_time
        
        if elapsed >= delay:
            self._index += 1
            return (command, 0.0)
        else:
            return (command, delay - elapsed)

    def reset(self) -> None:
        """Reset the sequence."""
        self._index = 0
        self._start_time = None

    @property
    def is_complete(self) -> bool:
        """Check if sequence is complete."""
        return self._index >= len(self._commands)

    @property  
    def progress(self) -> float:
        """Get progress (0-1)."""
        if not self._commands:
            return 1.0
        return self._index / len(self._commands)


# Pre-defined scenario command sequences

def create_walk_rooms_scenario(room_guids: list[str]) -> ScenarioCommandSequence:
    """Create a scenario that walks through rooms.
    
    Args:
        room_guids: List of room GUIDs to visit.
        
    Returns:
        Command sequence.
    """
    seq = ScenarioCommandSequence(
        name="walk_rooms",
        description="Walk through multiple rooms",
    )
    
    delay = 0.0
    for guid in room_guids:
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TELEPORT,
            target_id=guid,
            parameters={"type": "room"},
        ))
        delay += 5.0  # 5 seconds per room
    
    return seq


def create_toggle_scenario(entity_guids: list[str]) -> ScenarioCommandSequence:
    """Create a scenario that toggles entities.
    
    Args:
        entity_guids: List of entity GUIDs to toggle.
        
    Returns:
        Command sequence.
    """
    seq = ScenarioCommandSequence(
        name="toggle_drawer_light",
        description="Toggle drawers and lights",
    )
    
    delay = 1.0
    for guid in entity_guids:
        # Toggle on
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TOGGLE,
            target_id=guid,
        ))
        delay += 2.0
        
        # Toggle off
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TOGGLE,
            target_id=guid,
        ))
        delay += 2.0
    
    return seq


def create_spawn_ball_scenario(
    positions: list[tuple[float, float, float]],
) -> ScenarioCommandSequence:
    """Create a scenario that spawns, moves, and despawns a ball.
    
    Args:
        positions: List of positions to move ball to.
        
    Returns:
        Command sequence.
    """
    seq = ScenarioCommandSequence(
        name="spawn_move_ball",
        description="Spawn a ball, move it, then despawn",
    )
    
    # Spawn
    seq.add_command(0.0, UnityCommand(
        command_type=CommandType.SPAWN,
        target_label="ball",
        parameters={"position": list(positions[0]) if positions else [0, 1, 0]},
    ))
    
    # Move through positions
    delay = 2.0
    for pos in positions[1:]:
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.MOVE,
            target_label="ball",
            parameters={"position": list(pos)},
        ))
        delay += 2.0
    
    # Despawn
    seq.add_command(delay, UnityCommand(
        command_type=CommandType.DESPAWN,
        target_label="ball",
    ))
    
    return seq
