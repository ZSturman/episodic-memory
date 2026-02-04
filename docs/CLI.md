# CLI Reference

Complete command-line interface documentation for the Episodic Memory Agent.

## Overview

```bash
python -m episodic_agent <command> [options]
```

## Commands

### run

Run the agent loop continuously.

```bash
python -m episodic_agent run [OPTIONS]
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | `stub` | Module profile to use |
| `--steps` | `100` | Number of steps (0 = infinite) |
| `--fps` | `10` | Target frames per second |
| `--unity-ws` | `ws://localhost:8765` | Unity WebSocket URL |
| `--output` | `runs/` | Output directory |
| `--seed` | None | Random seed for reproducibility |
| `--freeze-interval` | `50` | Steps between automatic freezes |
| `--auto-label` | `false` | Auto-generate labels without prompting |
| `--verbose` | `false` | Enable debug logging |

#### Examples

```bash
# Basic stub mode
python -m episodic_agent run --profile stub --steps 200

# Unity with auto-labeling
python -m episodic_agent run --profile unity_full --auto-label --steps 0

# Reproducible run with seed
python -m episodic_agent run --profile stub --seed 42 --steps 100

# Debug mode
python -m episodic_agent run --profile unity_cheat --verbose
```

---

### scenario

Run a predefined test scenario.

```bash
python -m episodic_agent scenario <name> [OPTIONS]
```

#### Scenario Names

| Name | Description |
|------|-------------|
| `walk_rooms` | Visit multiple rooms |
| `toggle_drawer_light` | Toggle drawers and lights |
| `spawn_move_ball` | Spawn and move a ball |
| `mixed` | All scenarios combined |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile` | `unity_full` | Module profile to use |
| `--output` | `runs/` | Output directory |
| `--unity-ws` | `ws://localhost:8765` | Unity WebSocket URL |
| `--replay` | None | JSONL file for offline replay |
| `--quiet` | `false` | Suppress per-step output |

#### Examples

```bash
# Run mixed scenario
python -m episodic_agent scenario mixed --profile unity_full

# Run with specific output directory
python -m episodic_agent scenario walk_rooms --output results/exp1/

# Offline replay (no Unity needed)
python -m episodic_agent scenario walk_rooms --replay recordings/walk.jsonl

# Quiet mode (less output)
python -m episodic_agent scenario mixed --quiet
```

---

### report

Generate report from run data.

```bash
python -m episodic_agent report <run_folder> [OPTIONS]
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--html` | `false` | Generate HTML report |
| `--output` | Same as input | Output path |
| `--compute-metrics` | `false` | Compute metrics before report |

#### Examples

```bash
# Text report
python -m episodic_agent report runs/20260202_174416

# HTML report
python -m episodic_agent report runs/20260202_174416 --html

# Compute metrics and generate report
python -m episodic_agent report runs/20260202_174416 --compute-metrics --html
```

---

### profiles

List available module profiles.

```bash
python -m episodic_agent profiles
```

#### Output

```
Available profiles:

  stub          All stub modules for testing
  unity_cheat   Unity with cheat perception (uses GUIDs)
  unity_full    Full features with spreading activation

Use --profile <name> to select a profile.
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `EPISODIC_AGENT_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `EPISODIC_AGENT_OUTPUT_DIR` | Default output directory |

```bash
export EPISODIC_AGENT_LOG_LEVEL=DEBUG
python -m episodic_agent run
```

---

## Output Files

Each run creates a timestamped folder:

```
runs/20260202_174416/
├── run.jsonl          # Step-by-step log
├── episodes.jsonl     # Frozen episodes
├── nodes.jsonl        # Graph nodes
├── edges.jsonl        # Graph edges
├── metrics.json       # Computed metrics
├── report.txt         # Text report (if generated)
└── report.html        # HTML report (if generated)
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Connection error |
| 130 | Interrupted (Ctrl+C) |

---

## Common Patterns

### Development Loop

```bash
# Quick test with stub
python -m episodic_agent run --profile stub --steps 50

# Unity integration test
python -m episodic_agent run --profile unity_cheat --auto-label --steps 100

# Full feature test
python -m episodic_agent scenario mixed --profile unity_full
python -m episodic_agent report runs/<latest> --html
```

### Debugging

```bash
# Verbose output
python -m episodic_agent run --verbose 2>&1 | tee debug.log

# Single step at a time
python -m episodic_agent run --fps 1 --steps 10 --verbose
```

### Batch Testing

```bash
# Run multiple scenarios
for scenario in walk_rooms toggle_drawer_light spawn_move_ball; do
    python -m episodic_agent scenario $scenario --profile unity_full
done

# Generate reports for all runs
for run in runs/*/; do
    python -m episodic_agent report "$run" --html
done
```
