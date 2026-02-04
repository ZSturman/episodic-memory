"""Integration tests for end-to-end functionality."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from episodic_agent.core.orchestrator import AgentOrchestrator
from episodic_agent.memory.episode_store import PersistentEpisodeStore
from episodic_agent.memory.graph_store import LabeledGraphStore
from episodic_agent.modules.stubs import (
    StubACFBuilder,
    StubBoundaryDetector,
    StubDialogManager,
    StubEntityResolver,
    StubEventResolver,
    StubLocationResolver,
    StubPerception,
    StubRetriever,
    StubSensorProvider,
)
from episodic_agent.schemas import (
    ActiveContextFrame,
    Episode,
    ObjectCandidate,
    Percept,
    SensorFrame,
)
from episodic_agent.metrics.logging import LogWriter


# =============================================================================
# End-to-End Stub Profile Tests
# =============================================================================

class TestStubProfileIntegration:
    """Integration tests using stub modules."""

    @pytest.fixture
    def temp_run_dir(self):
        """Create temporary run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def orchestrator(self, temp_run_dir):
        """Create orchestrator with persistent stores."""
        return AgentOrchestrator(
            sensor=StubSensorProvider(max_frames=100),
            perception=StubPerception(),
            acf_builder=StubACFBuilder(),
            location_resolver=StubLocationResolver(),
            entity_resolver=StubEntityResolver(),
            event_resolver=StubEventResolver(),
            retriever=StubRetriever(),
            boundary_detector=StubBoundaryDetector(freeze_interval=10),
            dialog_manager=StubDialogManager(),
            episode_store=PersistentEpisodeStore(temp_run_dir / "episodes.jsonl"),
            graph_store=LabeledGraphStore(
                temp_run_dir / "nodes.jsonl",
                temp_run_dir / "edges.jsonl",
            ),
            run_id="test_integration",
        )

    def test_run_multiple_steps(self, orchestrator):
        """Run multiple steps successfully."""
        results = []
        for _ in range(20):
            results.append(orchestrator.step())
        
        assert len(results) == 20
        assert all(r.step_number > 0 for r in results)

    def test_episodes_created(self, orchestrator):
        """Episodes are created during run."""
        # Run enough steps to trigger boundaries
        for _ in range(25):
            orchestrator.step()
        
        assert orchestrator.episode_count >= 2

    def test_persistence_works(self, orchestrator, temp_run_dir):
        """Data persists to files."""
        for _ in range(15):
            orchestrator.step()
        
        # Check files exist
        assert (temp_run_dir / "episodes.jsonl").exists()

    def test_log_writer(self, temp_run_dir):
        """Log writer creates valid JSONL."""
        log_path = temp_run_dir / "run.jsonl"
        
        with LogWriter(log_path) as writer:
            from episodic_agent.schemas import StepResult
            
            for i in range(5):
                result = StepResult(
                    run_id="test_log_writer",
                    step_number=i + 1,
                    frame_id=i + 1,
                    acf_id=f"acf_{i + 1}",
                    timestamp=datetime.now(),
                    location_label="TestRoom",
                    location_confidence=0.9,
                    entity_count=0,
                    event_count=0,
                    episode_count=0,
                    boundary_triggered=False,
                )
                writer.write(result)
        
        # Verify file content
        with open(log_path) as f:
            lines = f.readlines()
        
        assert len(lines) == 5
        
        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "step_number" in data


# =============================================================================
# Scenario Integration Tests
# =============================================================================

class TestScenarioIntegration:
    """Tests for scenario framework integration."""

    def test_scenario_imports(self):
        """Scenario modules import correctly."""
        from episodic_agent.scenarios.definitions import (
            WalkRoomsScenario,
            ToggleDrawerLightScenario,
            SpawnMoveBallScenario,
            MixedScenario,
        )
        
        assert WalkRoomsScenario is not None
        assert ToggleDrawerLightScenario is not None

    def test_scenario_instantiation(self):
        """Scenarios can be instantiated."""
        from episodic_agent.scenarios.definitions import WalkRoomsScenario
        
        scenario = WalkRoomsScenario()
        
        assert scenario.name == "scenario_walk_rooms"
        assert scenario.description is not None

    def test_scenario_max_steps(self):
        """Scenarios report max steps."""
        from episodic_agent.scenarios.definitions import WalkRoomsScenario
        
        scenario = WalkRoomsScenario()
        max_steps = scenario.get_max_steps()
        
        assert max_steps > 0


# =============================================================================
# Profile System Tests
# =============================================================================

class TestProfileSystem:
    """Tests for the profile configuration system."""

    def test_profile_registry(self):
        """Profiles are registered correctly."""
        from episodic_agent.utils.profiles import PROFILES, ProfileName
        
        assert ProfileName.STUB.value in PROFILES
        assert ProfileName.UNITY_CHEAT.value in PROFILES
        assert ProfileName.UNITY_FULL.value in PROFILES

    def test_get_profile(self):
        """Get profile by name."""
        from episodic_agent.utils.profiles import get_profile
        
        stub_profile = get_profile("stub")
        
        assert stub_profile.name == "stub"
        assert stub_profile.sensor_provider == "StubSensorProvider"

    def test_list_profiles(self):
        """List all available profiles."""
        from episodic_agent.utils.profiles import list_profiles
        
        profiles = list_profiles()
        
        assert len(profiles) >= 3
        # list_profiles returns list of (name, description) tuples
        assert any(p[0] == "stub" for p in profiles)


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    @pytest.fixture
    def run_with_data(self):
        """Create a run directory with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create run.jsonl
            with open(run_dir / "run.jsonl", "w") as f:
                for i in range(10):
                    step = {
                        "step_number": i + 1,
                        "timestamp": datetime.now().isoformat(),
                        "location_label": "TestRoom",
                        "location_confidence": 0.9,
                        "entity_count": 2,
                        "event_count": 0,
                        "episode_count": 0,
                        "boundary_triggered": False,
                    }
                    f.write(json.dumps(step) + "\n")
            
            # Create episodes.jsonl
            with open(run_dir / "episodes.jsonl", "w") as f:
                episode = {
                    "episode_id": "ep_001",
                    "source_acf_id": "acf_001",
                    "location_label": "TestRoom",
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "step_count": 10,
                }
                f.write(json.dumps(episode) + "\n")
            
            yield run_dir

    def test_report_generator_loads_data(self, run_with_data):
        """Report generator loads run data."""
        from episodic_agent.report import ReportGenerator
        
        generator = ReportGenerator(run_with_data)
        
        assert generator._data.get("step_count", 0) > 0

    def test_text_report_generation(self, run_with_data):
        """Text report can be generated."""
        from episodic_agent.report import ReportGenerator
        
        generator = ReportGenerator(run_with_data)
        report = generator.generate_text_report()
        
        assert len(report) > 0
        assert "step" in report.lower() or "episode" in report.lower()


# =============================================================================
# Metrics Evaluation Tests
# =============================================================================

class TestMetricsEvaluation:
    """Tests for metrics computation."""

    def test_location_metrics(self):
        """Location metrics compute correctly."""
        from episodic_agent.metrics.evaluation import LocationMetrics
        
        metrics = LocationMetrics(
            total_steps=100,
            steps_with_location=90,
            steps_with_ground_truth=80,
            correct_matches=70,
            unique_locations=4,
            location_changes=5,
        )
        
        assert metrics.accuracy == 70 / 80
        assert metrics.coverage == 90 / 100

    def test_metrics_to_dict(self):
        """Metrics convert to dict."""
        from episodic_agent.metrics.evaluation import LocationMetrics
        
        metrics = LocationMetrics(
            total_steps=100,
            steps_with_location=90,
        )
        
        data = metrics.to_dict()
        
        assert "total_steps" in data
        assert "accuracy" in data
        assert data["total_steps"] == 100
