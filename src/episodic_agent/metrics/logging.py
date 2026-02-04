"""JSONL logging utilities for run output and visualization.

This module provides:
- LogWriter: Core JSONL logging for run data
- EnhancedLogWriter: Extended logging with visualization-ready metrics
- TimeSeriesCollector: In-memory collection for real-time analysis
- LogAnalyzer: Post-hoc analysis and visualization data generation
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from episodic_agent.schemas import StepResult


# =============================================================================
# Core Log Writer
# =============================================================================

class LogWriter:
    """Writes step results to a JSONL log file.
    
    Creates a consistent log format with one JSON object per line.
    """

    def __init__(self, log_path: Path) -> None:
        """Initialize the log writer.
        
        Args:
            log_path: Path to the JSONL log file.
        """
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open file in append mode
        self._file = open(self._log_path, "a", encoding="utf-8")

    def write(self, step_result: StepResult) -> None:
        """Write a step result to the log.
        
        Args:
            step_result: The step result to log.
        """
        log_dict = step_result.to_log_dict()
        line = json.dumps(log_dict, separators=(",", ":"))
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        self._file.close()

    def __enter__(self) -> "LogWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures file is closed."""
        self.close()


# =============================================================================
# Enhanced Log Writer with Visualization Data
# =============================================================================

@dataclass
class TimeSeriesPoint:
    """A single point in a time series for visualization."""
    
    step: int
    timestamp: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTimeline:
    """Timeline entry for episode visualization."""
    
    episode_id: str
    start_step: int
    end_step: int
    location: str
    duration_steps: int
    entity_count: int
    event_count: int
    boundary_reason: str | None = None


class EnhancedLogWriter(LogWriter):
    """Extended log writer with visualization-ready metrics.
    
    In addition to JSONL logging, tracks time series data for:
    - Memory growth (nodes, edges, episodes)
    - Entity counts over time
    - Event frequency
    - Location transitions
    - Boundary triggers
    
    Use `get_visualization_data()` to export data suitable for plotting.
    """

    def __init__(self, log_path: Path, run_id: str = "") -> None:
        """Initialize enhanced log writer.
        
        Args:
            log_path: Path to the JSONL log file.
            run_id: Identifier for this run.
        """
        super().__init__(log_path)
        self._run_id = run_id
        self._start_time = datetime.now()
        
        # Time series collectors
        self._memory_series: list[TimeSeriesPoint] = []
        self._entity_series: list[TimeSeriesPoint] = []
        self._event_series: list[TimeSeriesPoint] = []
        self._location_transitions: list[dict[str, Any]] = []
        self._boundaries: list[dict[str, Any]] = []
        
        # Episode timeline
        self._episodes: list[EpisodeTimeline] = []
        self._current_episode_start: int = 1
        self._current_location: str = "unknown"
        self._prev_location: str = "unknown"
        
        # Cumulative counters
        self._cumulative_entities = 0
        self._cumulative_events = 0
        self._total_steps = 0

    def write(self, step_result: StepResult) -> None:
        """Write step result and collect visualization data.
        
        Args:
            step_result: The step result to log.
        """
        # Write to JSONL (parent behavior)
        super().write(step_result)
        
        # Collect time series data
        self._collect_time_series(step_result)

    def _collect_time_series(self, result: StepResult) -> None:
        """Collect time series data from step result."""
        step = result.step_number
        timestamp = result.timestamp.isoformat()
        self._total_steps = step
        
        # Memory growth
        if result.memory_counts:
            self._memory_series.append(TimeSeriesPoint(
                step=step,
                timestamp=timestamp,
                value=result.memory_counts.episodes,
                metadata={
                    "nodes": result.memory_counts.nodes,
                    "edges": result.memory_counts.edges,
                    "episodes": result.memory_counts.episodes,
                },
            ))
        
        # Entity count
        self._entity_series.append(TimeSeriesPoint(
            step=step,
            timestamp=timestamp,
            value=result.entity_count,
        ))
        self._cumulative_entities += result.entity_count
        
        # Event count
        event_count = result.event_count + result.delta_count
        self._event_series.append(TimeSeriesPoint(
            step=step,
            timestamp=timestamp,
            value=event_count,
        ))
        self._cumulative_events += event_count
        
        # Location transition
        if result.location_label != self._prev_location:
            self._location_transitions.append({
                "step": step,
                "timestamp": timestamp,
                "from_location": self._prev_location,
                "to_location": result.location_label,
                "confidence": result.location_confidence,
            })
            self._prev_location = result.location_label
        
        self._current_location = result.location_label
        
        # Boundary trigger
        if result.boundary_triggered:
            self._boundaries.append({
                "step": step,
                "timestamp": timestamp,
                "reason": result.boundary_reason,
                "location": result.location_label,
                "episode_count": result.episode_count,
            })
            
            # Record episode timeline
            self._episodes.append(EpisodeTimeline(
                episode_id=f"ep_{len(self._episodes) + 1:04d}",
                start_step=self._current_episode_start,
                end_step=step,
                location=result.location_label,
                duration_steps=step - self._current_episode_start + 1,
                entity_count=result.entity_count,
                event_count=result.event_count,
                boundary_reason=result.boundary_reason,
            ))
            self._current_episode_start = step + 1

    def get_visualization_data(self) -> dict[str, Any]:
        """Export all collected data for visualization.
        
        Returns:
            Dictionary with all time series and metadata suitable for plotting.
        """
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()
        
        return {
            "metadata": {
                "run_id": self._run_id,
                "start_time": self._start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "total_steps": self._total_steps,
                "total_episodes": len(self._episodes),
                "total_location_transitions": len(self._location_transitions),
                "total_boundaries": len(self._boundaries),
            },
            "time_series": {
                "memory_growth": [
                    {"step": p.step, "timestamp": p.timestamp, **p.metadata}
                    for p in self._memory_series
                ],
                "entity_counts": [
                    {"step": p.step, "timestamp": p.timestamp, "count": p.value}
                    for p in self._entity_series
                ],
                "event_counts": [
                    {"step": p.step, "timestamp": p.timestamp, "count": p.value}
                    for p in self._event_series
                ],
            },
            "transitions": {
                "locations": self._location_transitions,
                "boundaries": self._boundaries,
            },
            "episode_timeline": [
                {
                    "episode_id": ep.episode_id,
                    "start_step": ep.start_step,
                    "end_step": ep.end_step,
                    "location": ep.location,
                    "duration_steps": ep.duration_steps,
                    "entity_count": ep.entity_count,
                    "event_count": ep.event_count,
                    "boundary_reason": ep.boundary_reason,
                }
                for ep in self._episodes
            ],
            "summary_stats": {
                "steps_per_minute": (self._total_steps / duration * 60) if duration > 0 else 0,
                "episodes_per_minute": (len(self._episodes) / duration * 60) if duration > 0 else 0,
                "avg_episode_duration": (
                    sum(ep.duration_steps for ep in self._episodes) / len(self._episodes)
                    if self._episodes else 0
                ),
                "cumulative_entities": self._cumulative_entities,
                "cumulative_events": self._cumulative_events,
            },
        }

    def save_visualization_data(self, output_path: Path | None = None) -> Path:
        """Save visualization data to JSON file.
        
        Args:
            output_path: Path for output file. If None, uses run directory.
            
        Returns:
            Path to the saved file.
        """
        if output_path is None:
            output_path = self._log_path.parent / "visualization_data.json"
        
        data = self.get_visualization_data()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        return output_path

    def close(self) -> None:
        """Close log file and save visualization data."""
        # Save visualization data automatically
        self.save_visualization_data()
        super().close()


# =============================================================================
# Log Analyzer for Post-hoc Analysis
# =============================================================================

class LogAnalyzer:
    """Analyze completed run logs for visualization.
    
    Load existing JSONL log files and extract visualization-ready data.
    """

    def __init__(self, log_path: Path) -> None:
        """Initialize analyzer with log file path.
        
        Args:
            log_path: Path to JSONL log file.
        """
        self._log_path = log_path
        self._records: list[dict[str, Any]] = []
        self._loaded = False

    def load(self) -> "LogAnalyzer":
        """Load log file into memory.
        
        Returns:
            Self for method chaining.
        """
        self._records = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        self._loaded = True
        return self

    @property
    def record_count(self) -> int:
        """Number of loaded records."""
        return len(self._records)

    def get_time_series(self, field: str) -> list[tuple[int, Any]]:
        """Extract time series for a specific field.
        
        Args:
            field: Field name to extract (e.g., 'entity_count', 'location_label').
            
        Returns:
            List of (step, value) tuples.
        """
        if not self._loaded:
            self.load()
        
        return [
            (r.get("step_number", i), r.get(field))
            for i, r in enumerate(self._records)
            if field in r
        ]

    def get_memory_growth(self) -> dict[str, list[tuple[int, int]]]:
        """Extract memory growth time series.
        
        Returns:
            Dictionary with 'nodes', 'edges', 'episodes' lists.
        """
        if not self._loaded:
            self.load()
        
        result = {
            "nodes": [],
            "edges": [],
            "episodes": [],
        }
        
        for r in self._records:
            step = r.get("step_number", 0)
            counts = r.get("memory_counts", {})
            if counts:
                result["nodes"].append((step, counts.get("nodes", 0)))
                result["edges"].append((step, counts.get("edges", 0)))
                result["episodes"].append((step, counts.get("episodes", 0)))
        
        return result

    def get_boundary_events(self) -> list[dict[str, Any]]:
        """Extract all boundary trigger events.
        
        Returns:
            List of boundary events with step, reason, and location.
        """
        if not self._loaded:
            self.load()
        
        return [
            {
                "step": r.get("step_number"),
                "reason": r.get("boundary_reason"),
                "location": r.get("location_label"),
                "episode_count": r.get("episode_count"),
            }
            for r in self._records
            if r.get("boundary_triggered")
        ]

    def get_location_distribution(self) -> dict[str, int]:
        """Get distribution of steps per location.
        
        Returns:
            Dictionary mapping location label to step count.
        """
        if not self._loaded:
            self.load()
        
        distribution: dict[str, int] = defaultdict(int)
        for r in self._records:
            loc = r.get("location_label", "unknown")
            distribution[loc] += 1
        
        return dict(distribution)

    def get_event_frequency(self, window_size: int = 10) -> list[tuple[int, float]]:
        """Calculate event frequency over sliding window.
        
        Args:
            window_size: Number of steps in sliding window.
            
        Returns:
            List of (step, frequency) tuples.
        """
        if not self._loaded:
            self.load()
        
        event_counts = [r.get("event_count", 0) for r in self._records]
        result = []
        
        for i in range(len(event_counts)):
            start = max(0, i - window_size + 1)
            window = event_counts[start:i + 1]
            avg = sum(window) / len(window) if window else 0
            result.append((i + 1, avg))
        
        return result

    def generate_plot_data(self) -> dict[str, Any]:
        """Generate all data needed for plotting.
        
        Returns:
            Dictionary with plot-ready data structures.
        """
        if not self._loaded:
            self.load()
        
        memory = self.get_memory_growth()
        
        return {
            "steps": [r.get("step_number") for r in self._records],
            "timestamps": [r.get("timestamp") for r in self._records],
            "entity_counts": self.get_time_series("entity_count"),
            "event_counts": self.get_time_series("event_count"),
            "location_labels": self.get_time_series("location_label"),
            "location_confidence": self.get_time_series("location_confidence"),
            "memory_nodes": memory["nodes"],
            "memory_edges": memory["edges"],
            "memory_episodes": memory["episodes"],
            "boundaries": self.get_boundary_events(),
            "location_distribution": self.get_location_distribution(),
            "event_frequency": self.get_event_frequency(),
        }

    def export_csv(self, output_path: Path) -> None:
        """Export core time series to CSV for external tools.
        
        Args:
            output_path: Path for CSV output file.
        """
        if not self._loaded:
            self.load()
        
        headers = [
            "step_number",
            "timestamp",
            "location_label",
            "location_confidence",
            "entity_count",
            "event_count",
            "episode_count",
            "boundary_triggered",
            "boundary_reason",
            "memory_nodes",
            "memory_edges",
        ]
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            
            for r in self._records:
                counts = r.get("memory_counts", {})
                row = [
                    str(r.get("step_number", "")),
                    r.get("timestamp", ""),
                    r.get("location_label", ""),
                    str(r.get("location_confidence", "")),
                    str(r.get("entity_count", "")),
                    str(r.get("event_count", "")),
                    str(r.get("episode_count", "")),
                    str(r.get("boundary_triggered", "")),
                    r.get("boundary_reason", "") or "",
                    str(counts.get("nodes", "")),
                    str(counts.get("edges", "")),
                ]
                f.write(",".join(row) + "\n")
