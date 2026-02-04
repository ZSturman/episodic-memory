# Creating Custom Scenarios

Guide for creating your own test scenarios.

## Scenario Structure

A scenario consists of:

1. **Metadata** - Name, description
2. **Command sequence** - Actions to perform in Unity
3. **Wait periods** - Time for agent to process
4. **Validation** - Assertions on results

---

## Basic Scenario

```python
from episodic_agent.scenarios.runner import Scenario, register_scenario

@register_scenario
class MyCustomScenario(Scenario):
    """My custom test scenario."""
    
    def __init__(self, custom_param: str = "default"):
        super().__init__(
            name="my_custom_scenario",
            description="Description of what this tests",
        )
        self._custom_param = custom_param
    
    def get_command_sequence(self):
        """Define commands to send to Unity."""
        from episodic_agent.modules.unity.command_client import CommandSequence
        
        return CommandSequence([
            # Teleport to starting position
            {
                "command": "teleport_player",
                "parameters": {"room_guid": "room_living"}
            },
            # Wait for frames
            {
                "command": "wait",
                "parameters": {"seconds": 3}
            },
            # Perform action
            {
                "command": "toggle_interactable",
                "parameters": {
                    "entity_guid": "door_001",
                    "target_state": "Open"
                }
            },
            # Wait and observe
            {
                "command": "wait",
                "parameters": {"seconds": 2}
            },
        ])
    
    def get_max_steps(self) -> int:
        """Maximum steps for this scenario."""
        return 100
    
    def get_replay_path(self):
        """Optional: Path to replay file for offline testing."""
        from pathlib import Path
        path = Path(f"recordings/{self.name}.jsonl")
        return path if path.exists() else None
```

---

## Command Reference

### teleport_player

Move player to a room.

```python
{
    "command": "teleport_player",
    "parameters": {
        "room_guid": "room_kitchen_001"
    }
}
```

### toggle_interactable

Change entity state.

```python
{
    "command": "toggle_interactable",
    "parameters": {
        "entity_guid": "door_front_001",
        "target_state": "Open"  # or "Closed", "On", "Off"
    }
}
```

### spawn_ball

Create a ball at position.

```python
{
    "command": "spawn_ball",
    "parameters": {
        "position": {"x": 0, "y": 1, "z": 0}
    }
}
```

### move_ball

Move existing ball.

```python
{
    "command": "move_ball",
    "parameters": {
        "position": {"x": 5, "y": 1, "z": 3}
    }
}
```

### despawn_ball

Remove ball.

```python
{
    "command": "despawn_ball",
    "parameters": {}
}
```

### reset_world

Reset to initial state.

```python
{
    "command": "reset_world",
    "parameters": {}
}
```

### wait

Pause command execution.

```python
{
    "command": "wait",
    "parameters": {
        "seconds": 5.0
    }
}
```

---

## Adding Validation

Override `validate_result` to check scenario outcomes:

```python
class MyScenario(Scenario):
    ...
    
    def validate_result(self, result) -> list[str]:
        """Return list of error messages (empty = success)."""
        errors = []
        
        # Check episode count
        if result.episode_count < 3:
            errors.append(
                f"Expected at least 3 episodes, got {result.episode_count}"
            )
        
        # Check specific location was detected
        locations = {e.location_label for e in result.episodes}
        if "Kitchen" not in locations:
            errors.append("Kitchen location was never detected")
        
        # Check events were captured
        total_events = sum(len(e.events) for e in result.episodes)
        if total_events < 2:
            errors.append(f"Expected ≥2 events, got {total_events}")
        
        # Check command success
        if result.commands_failed > 0:
            errors.append(
                f"{result.commands_failed} commands failed"
            )
        
        return errors
```

---

## Parameterized Scenarios

Create configurable scenarios:

```python
@register_scenario
class MultiRoomScenario(Scenario):
    def __init__(
        self,
        room_sequence: list[str] = None,
        dwell_time: float = 5.0,
    ):
        super().__init__(
            name="multi_room",
            description="Visit multiple rooms in sequence",
        )
        self._rooms = room_sequence or [
            "room_living",
            "room_kitchen",
            "room_bedroom",
        ]
        self._dwell = dwell_time
    
    def get_command_sequence(self):
        from episodic_agent.modules.unity.command_client import CommandSequence
        
        commands = []
        for room in self._rooms:
            commands.append({
                "command": "teleport_player",
                "parameters": {"room_guid": room}
            })
            commands.append({
                "command": "wait",
                "parameters": {"seconds": self._dwell}
            })
        
        return CommandSequence(commands)
    
    def get_max_steps(self) -> int:
        return int(len(self._rooms) * self._dwell * 15)  # 15 fps buffer
```

---

## Registering Scenarios

### Using Decorator

```python
@register_scenario
class MyScenario(Scenario):
    ...
```

This automatically registers the scenario with its name.

### Manual Registration

```python
from episodic_agent.scenarios.runner import SCENARIOS

class MyScenario(Scenario):
    ...

SCENARIOS["my_scenario"] = MyScenario
```

---

## Running Custom Scenarios

After creating and registering:

```bash
# If registered with decorator
python -m episodic_agent scenario my_custom_scenario --profile unity_full

# With parameters (via code)
# Edit scenarios/definitions.py to instantiate with custom params
```

---

## Scenario File Organization

```
src/episodic_agent/scenarios/
├── __init__.py
├── runner.py          # Scenario runner framework
├── definitions.py     # Built-in scenarios
└── custom/            # Custom scenarios (optional)
    ├── __init__.py
    └── my_scenarios.py
```

Import custom scenarios in `__init__.py`:

```python
# scenarios/__init__.py
from .definitions import *
from .custom.my_scenarios import *
```

---

## Debugging Scenarios

### Verbose Mode

```bash
python -m episodic_agent scenario my_scenario --verbose
```

### Single Command at a Time

```python
def get_command_sequence(self):
    return CommandSequence([
        {"command": "teleport_player", "parameters": {"room_guid": "room_living"}},
        {"command": "wait", "parameters": {"seconds": 30}},  # Long wait to debug
    ])
```

### Check Unity Console

Watch for:
- Command received messages
- Error messages
- Entity not found warnings

---

## Example: Stress Test Scenario

```python
@register_scenario
class StressTestScenario(Scenario):
    """Rapid teleportation stress test."""
    
    def __init__(self, iterations: int = 20):
        super().__init__(
            name="stress_test",
            description="Rapid location changes for stress testing",
        )
        self._iterations = iterations
    
    def get_command_sequence(self):
        from episodic_agent.modules.unity.command_client import CommandSequence
        
        rooms = ["room_living", "room_kitchen", "room_bedroom", "room_bathroom"]
        commands = []
        
        for i in range(self._iterations):
            room = rooms[i % len(rooms)]
            commands.extend([
                {"command": "teleport_player", "parameters": {"room_guid": room}},
                {"command": "wait", "parameters": {"seconds": 0.5}},
            ])
        
        return CommandSequence(commands)
    
    def get_max_steps(self) -> int:
        return self._iterations * 10
    
    def validate_result(self, result) -> list[str]:
        errors = []
        
        # Should have roughly as many episodes as room changes
        expected_episodes = self._iterations - 1  # -1 for initial
        if result.episode_count < expected_episodes * 0.8:
            errors.append(
                f"Expected ~{expected_episodes} episodes, got {result.episode_count}"
            )
        
        return errors
```

---

## Best Practices

1. **Clear naming** - Scenario name should describe what it tests
2. **Atomic actions** - Each command tests one thing
3. **Appropriate waits** - Give agent time to process
4. **Meaningful validation** - Check specific expected outcomes
5. **Document expectations** - Comment what each step should produce
6. **Parameterize** - Allow customization for different test cases
7. **Include replay path** - Enable offline testing
