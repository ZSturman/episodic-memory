## Phase 6: Spreading Activation, Predictions & Test Harness

Phase 6 completes the episodic memory system with associative retrieval, predictions, and automated testing.

### Features

- **Spreading Activation Retrieval**: Query memory by spreading activation from cues
- **Prediction Module**: Generate predictions from graph statistics, compute prediction errors
- **Enhanced Boundary Detection**: Use prediction error and salient events for segmentation
- **Unity Command Client**: Scripted scenario execution via commands
- **Scenario Framework**: Automated end-to-end testing with pre-defined scenarios
- **Metrics Evaluation**: Comprehensive metrics for location, entity, event, and memory
- **Report Tool**: Generate text and HTML reports from run data

### Full Demo Flow

1. **Start Unity Simulator**
   ```bash
   # In Unity Editor, enter Play mode
   # WebSocket server starts on ws://localhost:8765
   ```

2. **Run Automated Scenario**
   ```bash
   # Run the mixed scenario (walks + toggles + spawns)
   python -m episodic_agent scenario mixed --profile unity_full
   
   # Or run specific scenarios
   python -m episodic_agent scenario walk_rooms --profile unity_full
   python -m episodic_agent scenario toggle_drawer_light --profile unity_full
   python -m episodic_agent scenario spawn_move_ball --profile unity_full
   ```

3. **View Generated Report**
   ```bash
   # Reports are auto-generated after scenarios
   # Or generate manually:
   python -m episodic_agent report runs/20260202_scenario_mixed
   
   # Open HTML report in browser
   python -m episodic_agent report runs/20260202_scenario_mixed --html
   ```

4. **Offline Testing (No Unity)**
   ```bash
   # Replay from recorded JSONL file
   python -m episodic_agent scenario walk_rooms --replay replay.jsonl
   ```

### CLI Commands (Phase 6)

```bash
# List all available profiles
python -m episodic_agent profiles

# Run scenario with full features
python -m episodic_agent scenario <name> [options]

# Generate report from run folder
python -m episodic_agent report <run_folder> [options]
```

#### Scenario Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | `unity_full` | Profile to use |
| `--output` | `runs/` | Output directory |
| `--unity-ws` | `ws://localhost:8765` | Unity WebSocket URL |
| `--replay` | None | JSONL file for offline replay |
| `--quiet` | `false` | Suppress per-step output |

#### Available Scenarios

| Name | Description |
|------|-------------|
| `walk_rooms` | Walk through all rooms |
| `toggle_drawer_light` | Toggle drawer and lights |
| `spawn_move_ball` | Spawn and move a ball |
| `mixed` | All above combined |

### Profiles

| Profile | Description |
|---------|-------------|
| `stub` | All stub modules (testing only) |
| `unity_cheat` | Unity with cheat perception |
| `unity_full` | **Phase 6**: Full features with spreading activation, hysteresis boundary, prediction |

### Report Output

The report tool generates:
- **Text Report** (`report.txt`): Human-readable summary
- **HTML Report** (`report.html`): Visual dashboard with metrics

Includes:
- Summary metrics (steps, episodes, duration)
- Location recognition accuracy
- Event detection rates
- Memory growth statistics
- Episode timeline

## Troubleshooting

### Connection Issues

**Problem**: Connection failed or WebSocket connection refused

**Solution**:
1. Ensure Unity is running and in Play mode
2. Check WebSocket server is on port 8765
3. Verify firewall allows localhost connections
4. Try explicit URL: `--unity-ws ws://127.0.0.1:8765`

### Protocol Mismatch

**Problem**: Invalid frame format or Schema validation failed

**Solution**:
1. Ensure Unity and Python use matching protocol schemas
2. Check `UnitySensorSim/protocol/sensor_frame_schema.json`
3. Update Unity scripts if schema changed
4. Enable verbose logging: `--verbose`

### No Frames Received

**Problem**: Agent running but no frame updates

**Solution**:
1. Move around in Unity to trigger frame updates
2. Check Unity console for errors
3. Verify `SensorStreamer` component is active
4. Reduce FPS if system overloaded: `--fps 5`

### Location Not Learned

**Problem**: Re-entering room prompts for label again

**Solution**:
1. Check `nodes.jsonl` for existing location nodes
2. Verify GUID stability (same GUID for same room)
3. Check room volume colliders in Unity
4. Use `--auto-label` for automatic labeling

### Scenario Errors

**Problem**: Scenario fails with Command timeout or WebSocket closed

**Solution**:
1. Ensure Unity is responsive (not paused)
2. Check network connectivity
3. Try offline replay: `--replay replay.jsonl`
4. Reduce scenario speed if needed

### Report Generation Fails

**Problem**: Error generating report or empty metrics

**Solution**:
1. Ensure run completed successfully
2. Check for required files: `run.jsonl`, `episodes.jsonl`
3. Run metrics computation first if missing `metrics.json`
4. Check file permissions on run directory

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific phase tests
pytest tests/test_phase6.py -v

# Run with coverage
pytest --cov=episodic_agent tests/
```

### Creating Custom Scenarios

```python
from episodic_agent.scenarios import Scenario, ScenarioStep

class MyScenario(Scenario):
    @property
    def name(self) -> str:
        return "my_scenario"
    
    @property
    def description(self) -> str:
        return "Custom scenario"
    
    def get_steps(self) -> list[ScenarioStep]:
        return [
            ScenarioStep(
                name="teleport_home",
                command_type="teleport",
                command_args={"position": {"x": 0, "y": 0, "z": 0}},
                wait_frames=10,
            ),
            # ... more steps
        ]
```

### Adding New Profiles

Edit `src/episodic_agent/utils/profiles.py`:

```python
MY_PROFILE = ProfileConfig(
    name="my_profile",
    description="Custom profile",
    sensor_provider="UnityWebSocketSensorProvider",
    perception="PerceptionUnityCheat",
    # ... other modules
    retriever="SpreadingActivationRetriever",
    boundary_detector="HysteresisBoundaryDetector",
    parameters={
        "spreading_decay": 0.9,
        "boundary_high_threshold": 0.8,
    },
)

PROFILES["my_profile"] = MY_PROFILE
```
