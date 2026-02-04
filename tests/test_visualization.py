"""Tests for visualization and enhanced logging."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from episodic_agent.metrics.logging import (
    EnhancedLogWriter,
    LogAnalyzer,
    LogWriter,
    TimeSeriesPoint,
    EpisodeTimeline,
)
from episodic_agent.schemas import StepResult
from episodic_agent.schemas.results import MemoryCounts


# =============================================================================
# LogWriter Tests
# =============================================================================

class TestLogWriter:
    """Tests for basic LogWriter."""

    def _make_step_result(self, step: int = 1) -> StepResult:
        """Create a test step result."""
        return StepResult(
            run_id="test_run",
            step_number=step,
            frame_id=step,
            acf_id="acf_test",
            location_label="TestRoom",
            location_confidence=0.9,
            entity_count=2,
            event_count=1,
            episode_count=0,
            boundary_triggered=False,
            memory_counts=MemoryCounts(episodes=0, nodes=5, edges=10),
        )

    def test_creates_file(self):
        """LogWriter creates log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            with LogWriter(log_path) as writer:
                writer.write(self._make_step_result())
            
            assert log_path.exists()

    def test_writes_jsonl(self):
        """LogWriter writes valid JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            with LogWriter(log_path) as writer:
                for i in range(5):
                    writer.write(self._make_step_result(i + 1))
            
            # Read and verify
            with open(log_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 5
            
            # Each line is valid JSON
            for line in lines:
                data = json.loads(line)
                assert "step_number" in data

    def test_context_manager(self):
        """LogWriter works as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            with LogWriter(log_path) as writer:
                writer.write(self._make_step_result())
            
            # File should be closed and flushed
            with open(log_path) as f:
                content = f.read()
            
            assert len(content) > 0


# =============================================================================
# EnhancedLogWriter Tests
# =============================================================================

class TestEnhancedLogWriter:
    """Tests for enhanced log writer with visualization data."""

    def _make_step_result(
        self,
        step: int = 1,
        location: str = "TestRoom",
        boundary: bool = False,
        boundary_reason: str | None = None,
        episode_count: int = 0,
    ) -> StepResult:
        """Create a test step result."""
        return StepResult(
            run_id="test_run",
            step_number=step,
            frame_id=step,
            acf_id="acf_test",
            location_label=location,
            location_confidence=0.9,
            entity_count=2,
            event_count=1,
            episode_count=episode_count,
            boundary_triggered=boundary,
            boundary_reason=boundary_reason,
            memory_counts=MemoryCounts(episodes=episode_count, nodes=5 + step, edges=10 + step * 2),
        )

    def test_collects_time_series(self):
        """Enhanced writer collects time series data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            writer = EnhancedLogWriter(log_path, run_id="test")
            
            for i in range(10):
                writer.write(self._make_step_result(i + 1))
            
            data = writer.get_visualization_data()
            writer.close()
            
            assert len(data["time_series"]["entity_counts"]) == 10
            assert len(data["time_series"]["memory_growth"]) == 10

    def test_tracks_location_transitions(self):
        """Enhanced writer tracks location changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            writer = EnhancedLogWriter(log_path, run_id="test")
            
            # Steps in different locations
            for i, loc in enumerate(["Kitchen", "Kitchen", "Bedroom", "Bedroom", "Kitchen"]):
                writer.write(self._make_step_result(i + 1, location=loc))
            
            data = writer.get_visualization_data()
            writer.close()
            
            transitions = data["transitions"]["locations"]
            # Initial unknown->Kitchen, Kitchen->Bedroom, Bedroom->Kitchen
            assert len(transitions) == 3
            assert transitions[1]["from_location"] == "Kitchen"
            assert transitions[1]["to_location"] == "Bedroom"

    def test_tracks_boundaries(self):
        """Enhanced writer tracks boundary events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            writer = EnhancedLogWriter(log_path, run_id="test")
            
            # Create some steps with boundaries
            writer.write(self._make_step_result(1))
            writer.write(self._make_step_result(2))
            writer.write(self._make_step_result(3, boundary=True, boundary_reason="test", episode_count=1))
            writer.write(self._make_step_result(4))
            writer.write(self._make_step_result(5, boundary=True, boundary_reason="test2", episode_count=2))
            
            data = writer.get_visualization_data()
            writer.close()
            
            boundaries = data["transitions"]["boundaries"]
            assert len(boundaries) == 2

    def test_builds_episode_timeline(self):
        """Enhanced writer builds episode timeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            writer = EnhancedLogWriter(log_path, run_id="test")
            
            # Steps with boundary
            for i in range(5):
                writer.write(self._make_step_result(i + 1))
            writer.write(self._make_step_result(6, boundary=True, boundary_reason="time", episode_count=1))
            
            data = writer.get_visualization_data()
            writer.close()
            
            timeline = data["episode_timeline"]
            assert len(timeline) == 1
            assert timeline[0]["start_step"] == 1
            assert timeline[0]["end_step"] == 6

    def test_saves_visualization_json(self):
        """Enhanced writer saves visualization JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            with EnhancedLogWriter(log_path, run_id="test") as writer:
                for i in range(5):
                    writer.write(self._make_step_result(i + 1))
            
            # Check visualization file was created
            viz_path = Path(tmpdir) / "visualization_data.json"
            assert viz_path.exists()
            
            with open(viz_path) as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "time_series" in data

    def test_summary_stats(self):
        """Enhanced writer calculates summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            writer = EnhancedLogWriter(log_path, run_id="test")
            
            for i in range(10):
                boundary = (i + 1) % 5 == 0
                writer.write(self._make_step_result(
                    i + 1,
                    boundary=boundary,
                    episode_count=((i + 1) // 5) if boundary else (i // 5),
                ))
            
            data = writer.get_visualization_data()
            writer.close()
            
            stats = data["summary_stats"]
            assert "steps_per_minute" in stats
            assert "avg_episode_duration" in stats


# =============================================================================
# LogAnalyzer Tests
# =============================================================================

class TestLogAnalyzer:
    """Tests for log analysis."""

    @pytest.fixture
    def log_file(self):
        """Create a test log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.jsonl"
            
            # Write test data
            with open(log_path, "w") as f:
                for i in range(20):
                    record = {
                        "step_number": i + 1,
                        "timestamp": datetime.now().isoformat(),
                        "location_label": "Kitchen" if i < 10 else "Bedroom",
                        "location_confidence": 0.9,
                        "entity_count": i % 5,
                        "event_count": 1 if i % 3 == 0 else 0,
                        "episode_count": i // 10,
                        "boundary_triggered": i == 10,
                        "boundary_reason": "location_change" if i == 10 else None,
                        "memory_counts": {
                            "nodes": 5 + i,
                            "edges": 10 + i * 2,
                            "episodes": i // 10,
                        },
                    }
                    f.write(json.dumps(record) + "\n")
            
            yield log_path

    def test_load(self, log_file):
        """Analyzer loads log file."""
        analyzer = LogAnalyzer(log_file)
        analyzer.load()
        
        assert analyzer.record_count == 20

    def test_get_time_series(self, log_file):
        """Analyzer extracts time series."""
        analyzer = LogAnalyzer(log_file).load()
        
        entity_counts = analyzer.get_time_series("entity_count")
        
        assert len(entity_counts) == 20
        assert all(isinstance(t, tuple) for t in entity_counts)

    def test_get_memory_growth(self, log_file):
        """Analyzer extracts memory growth."""
        analyzer = LogAnalyzer(log_file).load()
        
        memory = analyzer.get_memory_growth()
        
        assert "nodes" in memory
        assert "edges" in memory
        assert "episodes" in memory
        assert len(memory["nodes"]) == 20

    def test_get_boundary_events(self, log_file):
        """Analyzer extracts boundary events."""
        analyzer = LogAnalyzer(log_file).load()
        
        boundaries = analyzer.get_boundary_events()
        
        assert len(boundaries) == 1
        assert boundaries[0]["reason"] == "location_change"

    def test_get_location_distribution(self, log_file):
        """Analyzer calculates location distribution."""
        analyzer = LogAnalyzer(log_file).load()
        
        dist = analyzer.get_location_distribution()
        
        assert "Kitchen" in dist
        assert "Bedroom" in dist
        assert dist["Kitchen"] == 10
        assert dist["Bedroom"] == 10

    def test_get_event_frequency(self, log_file):
        """Analyzer calculates event frequency."""
        analyzer = LogAnalyzer(log_file).load()
        
        freq = analyzer.get_event_frequency(window_size=5)
        
        assert len(freq) == 20
        assert all(isinstance(f, tuple) for f in freq)

    def test_generate_plot_data(self, log_file):
        """Analyzer generates plot-ready data."""
        analyzer = LogAnalyzer(log_file).load()
        
        data = analyzer.generate_plot_data()
        
        assert "steps" in data
        assert "entity_counts" in data
        assert "memory_nodes" in data
        assert "boundaries" in data
        assert "location_distribution" in data

    def test_export_csv(self, log_file):
        """Analyzer exports to CSV."""
        analyzer = LogAnalyzer(log_file).load()
        
        csv_path = log_file.parent / "analysis.csv"
        analyzer.export_csv(csv_path)
        
        assert csv_path.exists()
        
        with open(csv_path) as f:
            lines = f.readlines()
        
        # Header + 20 rows
        assert len(lines) == 21


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestTimeSeriesPoint:
    """Tests for TimeSeriesPoint dataclass."""

    def test_creation(self):
        """Create time series point."""
        point = TimeSeriesPoint(
            step=1,
            timestamp="2026-02-02T12:00:00",
            value=42.0,
            metadata={"extra": "data"},
        )
        
        assert point.step == 1
        assert point.value == 42.0
        assert point.metadata["extra"] == "data"

    def test_default_metadata(self):
        """Metadata defaults to empty dict."""
        point = TimeSeriesPoint(
            step=1,
            timestamp="2026-02-02T12:00:00",
            value=0.0,
        )
        
        assert point.metadata == {}


class TestEpisodeTimeline:
    """Tests for EpisodeTimeline dataclass."""

    def test_creation(self):
        """Create episode timeline entry."""
        ep = EpisodeTimeline(
            episode_id="ep_001",
            start_step=1,
            end_step=50,
            location="Kitchen",
            duration_steps=50,
            entity_count=5,
            event_count=3,
            boundary_reason="location_change",
        )
        
        assert ep.episode_id == "ep_001"
        assert ep.duration_steps == 50
        assert ep.boundary_reason == "location_change"
