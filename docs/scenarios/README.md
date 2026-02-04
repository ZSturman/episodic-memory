# Test Scenarios

Documentation for the automated test scenario framework.

## Overview

Scenarios are automated test sequences that:
- Send commands to Unity
- Capture agent responses
- Validate behavior
- Generate metrics

## Available Scenarios

| Scenario | Description | Tests |
|----------|-------------|-------|
| `walk_rooms` | Visit multiple rooms | Location detection, boundary triggers |
| `toggle_drawer_light` | Toggle object states | State change detection, events |
| `spawn_move_ball` | Create and move ball | Delta detection, appearance/disappearance |
| `mixed` | All combined | End-to-end validation |

---

## Running Scenarios

### Basic Usage

```bash
# Run specific scenario
python -m episodic_agent scenario walk_rooms --profile unity_full

# Run mixed (all scenarios)
python -m episodic_agent scenario mixed --profile unity_full
```

### Options

```bash
python -m episodic_agent scenario <name> \
    --profile unity_full \           # Module profile
    --output runs/my_test/ \         # Output directory
    --unity-ws ws://localhost:8765 \ # WebSocket URL
    --quiet                          # Less output
```

### Offline Replay

Run without Unity using recorded data:

```bash
python -m episodic_agent scenario walk_rooms --replay recordings/walk.jsonl
```

---

## Scenario Details

### walk_rooms

Tests location detection and episode segmentation.

**Sequence**:
1. Teleport to Living Room
2. Wait 5 seconds (capture frames)
3. Teleport to Kitchen
4. Wait 5 seconds
5. Teleport to Bedroom
6. Wait 5 seconds
7. Return to Living Room
8. Wait 5 seconds

**Expected Results**:
- 4 location changes detected
- 4 episodes created (one per room)
- Location labels learned/recognized

---

### toggle_drawer_light

Tests state change detection and event recognition.

**Sequence**:
1. Position in Living Room
2. Toggle drawer Open
3. Wait 2 seconds
4. Toggle drawer Closed
5. Toggle light On
6. Wait 2 seconds
7. Toggle light Off

**Expected Results**:
- 4 state changes detected
- 4 events logged
- Delta types: `state_changed`

---

### spawn_move_ball

Tests entity appearance, movement, and disappearance.

**Sequence**:
1. Position in Living Room
2. Spawn ball at (0, 1, 0)
3. Wait 2 seconds
4. Move ball to (3, 1, 0)
5. Wait 2 seconds
6. Move ball to (5, 1, 5)
7. Wait 2 seconds
8. Despawn ball

**Expected Results**:
- 1 appearance delta (spawn)
- 2 movement deltas
- 1 disappearance delta (despawn)
- Events for each

---

### mixed

Combines all scenarios for comprehensive testing.

**Sequence**:
1. Run walk_rooms
2. Run toggle_drawer_light
3. Run spawn_move_ball

**Expected Results**:
- All individual scenario expectations
- Multiple episode boundaries
- Varied event types

---

## Scenario Results

Each scenario creates output files:

```
runs/20260202_scenario_mixed/
├── run.jsonl              # Step-by-step log
├── episodes.jsonl         # Frozen episodes
├── nodes.jsonl            # Learned graph nodes
├── edges.jsonl            # Graph edges
├── metrics.json           # Computed metrics
├── scenario_result.json   # Scenario-specific results
├── report.txt             # Text summary
└── report.html            # Visual report
```

### scenario_result.json

```json
{
  "scenario_name": "mixed",
  "start_time": "2026-02-02T10:00:00",
  "end_time": "2026-02-02T10:05:00",
  "total_steps": 523,
  "total_episodes": 8,
  "commands_sent": 15,
  "commands_succeeded": 15,
  "commands_failed": 0,
  "assertions": {
    "passed": 12,
    "failed": 0
  }
}
```

---

## Creating Custom Scenarios

See [Custom Scenarios](CUSTOM.md) for detailed guide.

### Quick Example

```python
from episodic_agent.scenarios.runner import Scenario, register_scenario

@register_scenario
class MyScenario(Scenario):
    def __init__(self):
        super().__init__(
            name="my_scenario",
            description="Custom test scenario",
        )
    
    def get_command_sequence(self):
        from episodic_agent.modules.unity.command_client import CommandSequence
        
        return CommandSequence([
            {"command": "teleport_player", "parameters": {"room_guid": "room_001"}},
            {"command": "wait", "parameters": {"seconds": 3}},
            {"command": "toggle_interactable", "parameters": {"entity_guid": "door_001"}},
        ])
    
    def get_max_steps(self) -> int:
        return 200
```

---

## Validation

Scenarios can include assertions:

```python
class MyScenario(Scenario):
    def validate_result(self, result: ScenarioResult) -> list[str]:
        errors = []
        
        if result.episode_count < 2:
            errors.append(f"Expected ≥2 episodes, got {result.episode_count}")
        
        if not any(e.location_label == "Kitchen" for e in result.episodes):
            errors.append("Never detected Kitchen location")
        
        return errors
```

---

## Metrics

Scenarios compute metrics automatically:

| Metric | Description |
|--------|-------------|
| `location_accuracy` | Correct location matches / total |
| `event_detection_rate` | Events detected / state changes |
| `episode_rate` | Episodes / minute |
| `question_rate` | User prompts / minute |

See report for detailed breakdown.

---

## Best Practices

1. **Start with stub profile** - Verify scenario logic without Unity
2. **Use auto-label mode** - Faster testing without prompts
3. **Check Unity Console** - Look for errors during scenarios
4. **Review reports** - Use HTML reports for visual analysis
5. **Record baseline** - Save JSONL for regression testing
