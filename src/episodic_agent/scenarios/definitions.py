"""Predefined test scenarios.

Contains implementations of standard test scenarios:
- walk_rooms: Visit multiple rooms
- toggle_drawer_light: Toggle drawers and lights
- spawn_move_ball: Spawn, move, despawn ball
- mixed: Combination of above
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from episodic_agent.scenarios.runner import Scenario, register_scenario

if TYPE_CHECKING:
    from episodic_agent.modules.unity.command_client import ScenarioCommandSequence


@register_scenario
class WalkRoomsScenario(Scenario):
    """Scenario that walks through multiple rooms.
    
    Tests:
    - Location detection
    - Entity recognition in each room
    - Episode boundary on location change
    """

    def __init__(
        self,
        room_guids: list[str] | None = None,
        dwell_time: float = 5.0,
    ) -> None:
        """Initialize walk rooms scenario.
        
        Args:
            room_guids: List of room GUIDs to visit.
            dwell_time: Time to spend in each room (seconds).
        """
        super().__init__(
            name="scenario_walk_rooms",
            description="Walk through multiple rooms to test location detection",
        )
        # Default room GUIDs (would be configured per Unity scene)
        self._room_guids = room_guids or [
            "room_living",
            "room_kitchen", 
            "room_bedroom",
            "room_bathroom",
        ]
        self._dwell_time = dwell_time

    def get_command_sequence(self) -> "ScenarioCommandSequence | None":
        """Get command sequence for walking rooms."""
        from episodic_agent.modules.unity.command_client import (
            create_walk_rooms_scenario,
        )
        return create_walk_rooms_scenario(self._room_guids)

    def get_replay_path(self) -> Path | None:
        """Get replay file for this scenario."""
        # Check for pre-recorded scenario
        path = Path("scenarios/replay/walk_rooms.jsonl")
        if path.exists():
            return path
        return None

    def get_max_steps(self) -> int:
        """Maximum steps based on room count."""
        return len(self._room_guids) * int(self._dwell_time * 10) + 100


@register_scenario
class ToggleDrawerLightScenario(Scenario):
    """Scenario that toggles drawers and lights.
    
    Tests:
    - State change detection
    - Event recognition
    - Delta detection
    """

    def __init__(
        self,
        entity_guids: list[str] | None = None,
    ) -> None:
        """Initialize toggle scenario.
        
        Args:
            entity_guids: List of entity GUIDs to toggle.
        """
        super().__init__(
            name="scenario_toggle_drawer_light",
            description="Toggle drawers and lights to test state change detection",
        )
        self._entity_guids = entity_guids or [
            "drawer_01",
            "light_01",
            "drawer_02",
        ]

    def get_command_sequence(self) -> "ScenarioCommandSequence | None":
        """Get command sequence for toggling entities."""
        from episodic_agent.modules.unity.command_client import (
            create_toggle_scenario,
        )
        return create_toggle_scenario(self._entity_guids)

    def get_replay_path(self) -> Path | None:
        """Get replay file for this scenario."""
        path = Path("scenarios/replay/toggle_drawer_light.jsonl")
        if path.exists():
            return path
        return None

    def get_max_steps(self) -> int:
        """Maximum steps."""
        return 200


@register_scenario
class SpawnMoveBallScenario(Scenario):
    """Scenario that spawns, moves, and despawns a ball.
    
    Tests:
    - Entity appearance detection
    - Movement detection
    - Entity disappearance detection
    - Prediction error on unexpected entity
    """

    def __init__(
        self,
        positions: list[tuple[float, float, float]] | None = None,
    ) -> None:
        """Initialize spawn ball scenario.
        
        Args:
            positions: List of positions to move ball through.
        """
        super().__init__(
            name="scenario_spawn_move_ball",
            description="Spawn, move, and despawn a ball to test dynamic entity detection",
        )
        self._positions = positions or [
            (0.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 1.0, 2.0),
            (0.0, 1.0, 2.0),
        ]

    def get_command_sequence(self) -> "ScenarioCommandSequence | None":
        """Get command sequence for ball scenario."""
        from episodic_agent.modules.unity.command_client import (
            create_spawn_ball_scenario,
        )
        return create_spawn_ball_scenario(self._positions)

    def get_replay_path(self) -> Path | None:
        """Get replay file for this scenario."""
        path = Path("scenarios/replay/spawn_move_ball.jsonl")
        if path.exists():
            return path
        return None

    def get_max_steps(self) -> int:
        """Maximum steps."""
        return 150


@register_scenario
class MixedScenario(Scenario):
    """Combined scenario with room changes, toggles, and ball.
    
    Tests all systems together for comprehensive evaluation.
    """

    def __init__(self) -> None:
        """Initialize mixed scenario."""
        super().__init__(
            name="scenario_mixed",
            description="Combined test with rooms, toggles, and ball spawning",
        )

    def get_command_sequence(self) -> "ScenarioCommandSequence | None":
        """Get combined command sequence."""
        from episodic_agent.modules.unity.command_client import (
            CommandType,
            ScenarioCommandSequence,
            UnityCommand,
        )
        
        seq = ScenarioCommandSequence(
            name="mixed",
            description="Combined scenario",
        )
        
        delay = 0.0
        
        # Phase 1: Walk to living room
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TELEPORT,
            target_label="room_living",
        ))
        delay += 3.0
        
        # Phase 2: Toggle light
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TOGGLE,
            target_label="light_01",
        ))
        delay += 2.0
        
        # Phase 3: Spawn ball
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.SPAWN,
            target_label="ball",
            parameters={"position": [1.0, 1.0, 1.0]},
        ))
        delay += 2.0
        
        # Phase 4: Move ball
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.MOVE,
            target_label="ball",
            parameters={"position": [2.0, 1.0, 2.0]},
        ))
        delay += 2.0
        
        # Phase 5: Walk to kitchen
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TELEPORT,
            target_label="room_kitchen",
        ))
        delay += 3.0
        
        # Phase 6: Toggle drawer
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TOGGLE,
            target_label="drawer_01",
        ))
        delay += 2.0
        
        # Phase 7: Toggle drawer back
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TOGGLE,
            target_label="drawer_01",
        ))
        delay += 2.0
        
        # Phase 8: Walk back to living room
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TELEPORT,
            target_label="room_living",
        ))
        delay += 3.0
        
        # Phase 9: Despawn ball
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.DESPAWN,
            target_label="ball",
        ))
        delay += 2.0
        
        # Phase 10: Toggle light back
        seq.add_command(delay, UnityCommand(
            command_type=CommandType.TOGGLE,
            target_label="light_01",
        ))
        
        return seq

    def get_replay_path(self) -> Path | None:
        """Get replay file for this scenario."""
        path = Path("scenarios/replay/mixed.jsonl")
        if path.exists():
            return path
        return None

    def get_max_steps(self) -> int:
        """Maximum steps."""
        return 300


# Scenario name mappings
SCENARIO_NAMES = {
    "walk_rooms": WalkRoomsScenario,
    "scenario_walk_rooms": WalkRoomsScenario,
    "toggle": ToggleDrawerLightScenario,
    "toggle_drawer_light": ToggleDrawerLightScenario,
    "scenario_toggle_drawer_light": ToggleDrawerLightScenario,
    "spawn_ball": SpawnMoveBallScenario,
    "spawn_move_ball": SpawnMoveBallScenario,
    "scenario_spawn_move_ball": SpawnMoveBallScenario,
    "mixed": MixedScenario,
    "scenario_mixed": MixedScenario,
}


def get_scenario(name: str) -> Scenario:
    """Get a scenario instance by name.
    
    Args:
        name: Scenario name (case-insensitive).
        
    Returns:
        Scenario instance.
        
    Raises:
        ValueError: If scenario not found.
    """
    name_lower = name.lower()
    
    if name_lower in SCENARIO_NAMES:
        return SCENARIO_NAMES[name_lower]()
    
    # Check exact class name
    for cls in [WalkRoomsScenario, ToggleDrawerLightScenario, SpawnMoveBallScenario, MixedScenario]:
        if cls.__name__.lower() == name_lower:
            return cls()
    
    available = list(set(SCENARIO_NAMES.keys()))
    raise ValueError(f"Unknown scenario: {name}. Available: {available}")


def list_scenarios() -> list[tuple[str, str]]:
    """List available scenarios.
    
    Returns:
        List of (name, description) tuples.
    """
    seen = set()
    results = []
    
    for name, cls in SCENARIO_NAMES.items():
        if cls not in seen:
            instance = cls()
            results.append((instance.name, instance.description))
            seen.add(cls)
    
    return results
