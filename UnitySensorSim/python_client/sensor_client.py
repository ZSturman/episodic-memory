"""
Python WebSocket client for Unity Sensor Simulator.

This module provides a client to receive sensor frames from the Unity
sensor simulator and optionally send commands for testing.

Usage:
    python sensor_client.py [--port 8765] [--verbose]

Or programmatically:
    from sensor_client import SensorClient
    
    async def handle_frame(frame):
        print(f"Room: {frame['current_room_label']}")
    
    client = SensorClient(on_frame=handle_frame)
    await client.connect()
"""

import asyncio
import json
import uuid
from typing import Callable, Optional, Any
from dataclasses import dataclass


@dataclass
class Vector3:
    """3D vector."""
    x: float
    y: float
    z: float
    
    @classmethod
    def from_dict(cls, d: dict) -> "Vector3":
        return cls(x=d.get("x", 0), y=d.get("y", 0), z=d.get("z", 0))


@dataclass
class CameraPose:
    """Camera position and orientation."""
    position: Vector3
    rotation: Vector3
    forward: Vector3
    
    @classmethod
    def from_dict(cls, d: dict) -> "CameraPose":
        return cls(
            position=Vector3.from_dict(d.get("position", {})),
            rotation=Vector3.from_dict(d.get("rotation", {})),
            forward=Vector3.from_dict(d.get("forward", {}))
        )


@dataclass
class EntityData:
    """Entity information from sensor frame."""
    guid: str
    label: str
    category: str
    position: Vector3
    rotation: Optional[Vector3] = None
    state: str = "default"
    visible: bool = True
    distance: float = 0.0
    room_guid: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "EntityData":
        return cls(
            guid=d.get("guid", ""),
            label=d.get("label", ""),
            category=d.get("category", ""),
            position=Vector3.from_dict(d.get("position", {})),
            rotation=Vector3.from_dict(d.get("rotation", {})) if d.get("rotation") else None,
            state=d.get("state", "default"),
            visible=d.get("visible", True),
            distance=d.get("distance", 0.0),
            room_guid=d.get("room_guid")
        )


@dataclass
class StateChange:
    """State change event."""
    entity_guid: str
    change_type: str
    old_value: str
    new_value: str
    timestamp: float
    
    @classmethod
    def from_dict(cls, d: dict) -> "StateChange":
        return cls(
            entity_guid=d.get("entity_guid", ""),
            change_type=d.get("change_type", ""),
            old_value=d.get("old_value", ""),
            new_value=d.get("new_value", ""),
            timestamp=d.get("timestamp", 0.0)
        )


@dataclass
class SensorFrame:
    """Complete sensor frame from Unity."""
    protocol_version: str
    frame_id: int
    timestamp: float
    camera_pose: CameraPose
    current_room: Optional[str]
    current_room_label: Optional[str]
    entities: list[EntityData]
    state_changes: list[StateChange]
    
    @classmethod
    def from_dict(cls, d: dict) -> "SensorFrame":
        return cls(
            protocol_version=d.get("protocol_version", "1.0.0"),
            frame_id=d.get("frame_id", 0),
            timestamp=d.get("timestamp", 0.0),
            camera_pose=CameraPose.from_dict(d.get("camera_pose", {})),
            current_room=d.get("current_room"),
            current_room_label=d.get("current_room_label"),
            entities=[EntityData.from_dict(e) for e in d.get("entities", [])],
            state_changes=[StateChange.from_dict(c) for c in d.get("state_changes", [])]
        )


@dataclass
class CommandResponse:
    """Response from Unity for a command."""
    command_id: str
    success: bool
    error_message: Optional[str] = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "CommandResponse":
        return cls(
            command_id=d.get("command_id", ""),
            success=d.get("success", False),
            error_message=d.get("error_message")
        )


class SensorClient:
    """
    WebSocket client for Unity Sensor Simulator.
    
    Args:
        host: WebSocket server host
        port: WebSocket server port
        on_frame: Callback for sensor frames
        on_response: Callback for command responses
        on_connect: Callback when connected
        on_disconnect: Callback when disconnected
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        on_frame: Optional[Callable[[SensorFrame], Any]] = None,
        on_response: Optional[Callable[[CommandResponse], Any]] = None,
        on_connect: Optional[Callable[[], Any]] = None,
        on_disconnect: Optional[Callable[[], Any]] = None
    ):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        
        self.on_frame = on_frame
        self.on_response = on_response
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        
        self._websocket = None
        self._running = False
        self._pending_commands: dict[str, asyncio.Future] = {}
    
    async def connect(self):
        """Connect to the Unity sensor simulator and start receiving."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets library required: pip install websockets")
        
        self._running = True
        
        while self._running:
            try:
                async with websockets.connect(self.uri) as ws:
                    self._websocket = ws
                    
                    if self.on_connect:
                        await self._call_handler(self.on_connect)
                    
                    await self._receive_loop()
                    
            except Exception as e:
                print(f"[SensorClient] Connection error: {e}")
                self._websocket = None
                
                if self.on_disconnect:
                    await self._call_handler(self.on_disconnect)
                
                if self._running:
                    print("[SensorClient] Reconnecting in 2 seconds...")
                    await asyncio.sleep(2)
    
    async def _receive_loop(self):
        """Main receive loop."""
        while self._running and self._websocket:
            try:
                message = await self._websocket.recv()
                await self._handle_message(message)
            except Exception as e:
                print(f"[SensorClient] Receive error: {e}")
                break
    
    async def _handle_message(self, message: str):
        """Handle incoming message."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            print(f"[SensorClient] JSON parse error: {e}")
            return
        
        # Check if it's a sensor frame or command response
        if "frame_id" in data:
            # Sensor frame
            frame = SensorFrame.from_dict(data)
            if self.on_frame:
                await self._call_handler(self.on_frame, frame)
        
        elif "command_id" in data:
            # Command response
            response = CommandResponse.from_dict(data)
            
            # Check for pending command
            if response.command_id in self._pending_commands:
                future = self._pending_commands.pop(response.command_id)
                future.set_result(response)
            
            if self.on_response:
                await self._call_handler(self.on_response, response)
    
    async def _call_handler(self, handler: Callable, *args):
        """Call a handler, supporting both sync and async."""
        result = handler(*args)
        if asyncio.iscoroutine(result):
            await result
    
    async def send_command(
        self,
        command: str,
        parameters: Optional[dict] = None,
        wait_response: bool = True,
        timeout: float = 5.0
    ) -> Optional[CommandResponse]:
        """
        Send a command to Unity.
        
        Args:
            command: Command type (e.g., "teleport_player")
            parameters: Command parameters
            wait_response: Whether to wait for response
            timeout: Response timeout in seconds
            
        Returns:
            CommandResponse if wait_response is True, else None
        """
        if not self._websocket:
            raise RuntimeError("Not connected")
        
        command_id = str(uuid.uuid4())[:8]
        
        message = {
            "command": command,
            "command_id": command_id
        }
        
        if parameters:
            message["parameters"] = parameters
        
        # Set up response future if waiting
        if wait_response:
            future = asyncio.Future()
            self._pending_commands[command_id] = future
        
        # Send command
        await self._websocket.send(json.dumps(message))
        
        if not wait_response:
            return None
        
        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_commands.pop(command_id, None)
            raise TimeoutError(f"Command {command_id} timed out")
    
    # Convenience methods for common commands
    
    async def teleport_player(self, room_guid: str) -> CommandResponse:
        """Teleport player to a room."""
        return await self.send_command(
            "teleport_player",
            {"room_guid": room_guid}
        )
    
    async def toggle_interactable(
        self,
        entity_guid: str,
        target_state: Optional[str] = None
    ) -> CommandResponse:
        """Toggle an interactable or set specific state."""
        params = {"entity_guid": entity_guid}
        if target_state:
            params["target_state"] = target_state
        return await self.send_command("toggle_interactable", params)
    
    async def spawn_ball(
        self,
        position: Optional[tuple[float, float, float]] = None
    ) -> CommandResponse:
        """Spawn a ball, optionally at a specific position."""
        params = {}
        if position:
            params["position"] = {"x": position[0], "y": position[1], "z": position[2]}
        return await self.send_command("spawn_ball", params)
    
    async def despawn_ball(self) -> CommandResponse:
        """Despawn the ball."""
        return await self.send_command("despawn_ball")
    
    async def move_ball(self, position: tuple[float, float, float]) -> CommandResponse:
        """Move the ball to a position."""
        return await self.send_command(
            "move_ball",
            {"position": {"x": position[0], "y": position[1], "z": position[2]}}
        )
    
    async def reset_world(self) -> CommandResponse:
        """Reset the world to initial state."""
        return await self.send_command("reset_world")
    
    def stop(self):
        """Stop the client."""
        self._running = False


# Example usage
async def main():
    """Example sensor client usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unity Sensor Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    frame_count = 0
    last_room = None
    
    async def on_frame(frame: SensorFrame):
        nonlocal frame_count, last_room
        frame_count += 1
        
        # Report room changes
        if frame.current_room != last_room:
            print(f"\n📍 Room changed: {frame.current_room_label or 'Outside'}")
            last_room = frame.current_room
        
        # Report state changes
        for change in frame.state_changes:
            print(f"   ⚡ {change.entity_guid}: {change.old_value} → {change.new_value}")
        
        # Periodic status
        if args.verbose and frame_count % 50 == 0:
            visible = [e for e in frame.entities if e.visible]
            print(f"   Frame #{frame.frame_id}: {len(visible)} visible entities")
    
    async def on_response(response: CommandResponse):
        status = "✓" if response.success else "✗"
        msg = response.error_message or "OK"
        print(f"   {status} Command {response.command_id}: {msg}")
    
    def on_connect():
        print("🔗 Connected to Unity Sensor Simulator")
    
    def on_disconnect():
        print("🔌 Disconnected")
    
    client = SensorClient(
        host=args.host,
        port=args.port,
        on_frame=on_frame,
        on_response=on_response,
        on_connect=on_connect,
        on_disconnect=on_disconnect
    )
    
    print(f"Connecting to ws://{args.host}:{args.port}...")
    
    try:
        await client.connect()
    except KeyboardInterrupt:
        print("\nStopping...")
        client.stop()


if __name__ == "__main__":
    asyncio.run(main())
