# Visualization & Analysis Guide

This guide explains how to visualize and analyze run data from the episodic memory agent.

## Overview

After running the agent, you'll have logged data in the `runs/` directory. This data can be visualized to:

- Track memory growth over time (nodes, edges, episodes)
- Analyze entity and event frequency
- Visualize location transitions and confidence
- Examine episode boundaries and their triggers
- Generate reports for documentation

## Quick Start

```bash
# Print summary of a run
python -m episodic_agent visualize runs/20260202_174416

# Generate interactive HTML report
python -m episodic_agent visualize runs/20260202_174416 --html

# Export to CSV for external tools
python -m episodic_agent visualize runs/20260202_174416 --csv

# Generate matplotlib plots (requires matplotlib)
python -m episodic_agent visualize runs/20260202_174416 --plot
```

## Run Directory Structure

Each run creates a timestamped directory with:

```
runs/20260202_174416/
├── run.jsonl              # Main log file (JSONL format)
├── episodes.jsonl         # Frozen episodes
├── nodes.jsonl            # Graph nodes
├── edges.jsonl            # Graph edges
├── visualization_data.json # Pre-computed visualization data
├── report.html            # HTML report (if generated)
├── analysis.csv           # CSV export (if generated)
└── report.txt             # Text report
```

## Log Format

Each line in `run.jsonl` is a JSON object with:

```json
{
  "log_version": "1.0",
  "run_id": "20260202_174416",
  "timestamp": "2026-02-02T17:44:16.123456",
  "step_number": 42,
  "frame_id": 42,
  "acf_id": "acf_20260202_174416",
  "location_label": "Kitchen",
  "location_confidence": 0.95,
  "entity_count": 5,
  "event_count": 2,
  "episode_count": 3,
  "boundary_triggered": false,
  "boundary_reason": null,
  "memory_counts": {
    "episodes": 3,
    "nodes": 15,
    "edges": 28
  },
  "extras": {}
}
```

## Visualization Options

### 1. Interactive HTML Report

The HTML report provides:

- **Summary Statistics**: Total steps, episodes, locations, events
- **Memory Growth Chart**: Line chart showing nodes/edges/episodes over time
- **Entity & Event Counts**: Per-step counts plotted over time
- **Location Confidence**: Area chart of confidence values
- **Location Distribution**: Pie chart of time spent in each location
- **Boundary Table**: List of all episode boundaries with reasons

```bash
python -m episodic_agent visualize runs/20260202_174416 --html
# Opens report.html in the run directory
```

### 2. CSV Export

Export data for external tools like Excel, R, or Jupyter notebooks:

```bash
python -m episodic_agent visualize runs/20260202_174416 --csv
# Creates analysis.csv in the run directory
```

CSV columns:
- `step_number`, `timestamp`
- `location_label`, `location_confidence`
- `entity_count`, `event_count`, `episode_count`
- `boundary_triggered`, `boundary_reason`
- `memory_nodes`, `memory_edges`

### 3. Matplotlib Plots (Static)

Generate publication-ready static plots:

```bash
# Install matplotlib first
pip install matplotlib

# Generate plots
python -m episodic_agent visualize runs/20260202_174416 --plot

# Save to file
python -m episodic_agent visualize runs/20260202_174416 --plot -o plots.png
```

## Programmatic Access

### Using RunVisualizer

```python
from episodic_agent.visualize import RunVisualizer

# Load run data
viz = RunVisualizer("runs/20260202_174416")
viz.load()

# Get summary
print(f"Steps: {viz.step_count}")
print(f"Episodes: {viz.episode_count}")

# Print summary to console
viz.print_summary()

# Generate reports
viz.generate_html_report()
viz.export_csv()
```

### Using LogAnalyzer

```python
from episodic_agent.metrics.logging import LogAnalyzer

# Load and analyze log
analyzer = LogAnalyzer("runs/20260202_174416/run.jsonl")
analyzer.load()

# Get time series data
entity_counts = analyzer.get_time_series("entity_count")
# Returns: [(step, value), (step, value), ...]

# Get memory growth
memory = analyzer.get_memory_growth()
# Returns: {"nodes": [...], "edges": [...], "episodes": [...]}

# Get boundary events
boundaries = analyzer.get_boundary_events()
# Returns: [{"step": 50, "reason": "location_change", ...}, ...]

# Get location distribution
dist = analyzer.get_location_distribution()
# Returns: {"Kitchen": 45, "Bedroom": 30, ...}

# Get all plot-ready data
data = analyzer.generate_plot_data()
```

### Using EnhancedLogWriter

For real-time tracking during runs:

```python
from pathlib import Path
from episodic_agent.metrics.logging import EnhancedLogWriter

# Use enhanced writer instead of basic LogWriter
with EnhancedLogWriter(Path("runs/myrun/run.jsonl"), run_id="myrun") as logger:
    for step_result in run_steps():
        logger.write(step_result)
    
    # Access real-time visualization data
    data = logger.get_visualization_data()
    print(f"Episodes so far: {data['metadata']['total_episodes']}")

# Visualization data is automatically saved on close
```

## Custom Analysis

### Loading Raw Data

```python
import json
from pathlib import Path

# Load raw JSONL
records = []
with open("runs/20260202_174416/run.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

# Example: Find all location changes
prev_loc = None
changes = []
for r in records:
    loc = r.get("location_label")
    if loc != prev_loc and prev_loc is not None:
        changes.append({
            "step": r["step_number"],
            "from": prev_loc,
            "to": loc,
        })
    prev_loc = loc

print(f"Location changes: {len(changes)}")
```

### Comparing Multiple Runs

```python
from pathlib import Path
from episodic_agent.visualize import RunVisualizer

runs_dir = Path("runs")
results = []

for run_path in runs_dir.iterdir():
    if run_path.is_dir() and (run_path / "run.jsonl").exists():
        viz = RunVisualizer(run_path)
        viz.load()
        results.append({
            "run": run_path.name,
            "steps": viz.step_count,
            "episodes": viz.episode_count,
        })

# Sort by episode count
results.sort(key=lambda x: x["episodes"], reverse=True)
for r in results:
    print(f"{r['run']}: {r['episodes']} episodes in {r['steps']} steps")
```

## Visualization Data Schema

The `visualization_data.json` file contains:

```json
{
  "metadata": {
    "run_id": "20260202_174416",
    "start_time": "2026-02-02T17:44:16",
    "end_time": "2026-02-02T17:50:00",
    "duration_seconds": 344.5,
    "total_steps": 500,
    "total_episodes": 12,
    "total_location_transitions": 8,
    "total_boundaries": 12
  },
  "time_series": {
    "memory_growth": [
      {"step": 1, "timestamp": "...", "nodes": 0, "edges": 0, "episodes": 0},
      ...
    ],
    "entity_counts": [
      {"step": 1, "timestamp": "...", "count": 3},
      ...
    ],
    "event_counts": [
      {"step": 1, "timestamp": "...", "count": 0},
      ...
    ]
  },
  "transitions": {
    "locations": [
      {"step": 50, "from_location": "Kitchen", "to_location": "Bedroom", "confidence": 0.92},
      ...
    ],
    "boundaries": [
      {"step": 50, "reason": "location_change", "location": "Bedroom", "episode_count": 1},
      ...
    ]
  },
  "episode_timeline": [
    {
      "episode_id": "ep_0001",
      "start_step": 1,
      "end_step": 50,
      "location": "Kitchen",
      "duration_steps": 50,
      "entity_count": 5,
      "event_count": 3,
      "boundary_reason": "location_change"
    },
    ...
  ],
  "summary_stats": {
    "steps_per_minute": 87.2,
    "episodes_per_minute": 2.1,
    "avg_episode_duration": 41.7,
    "cumulative_entities": 523,
    "cumulative_events": 45
  }
}
```

## Tips

1. **Compare Profiles**: Run the same scenario with different profiles and compare visualizations
2. **Track Learning**: Watch memory growth curves to see how fast the agent learns
3. **Debug Boundaries**: Use the boundary table to understand why episodes are being segmented
4. **Export for Papers**: Use matplotlib plots with `--plot -o figure.png` for publications
5. **Batch Analysis**: Write scripts to analyze multiple runs and generate comparison tables
