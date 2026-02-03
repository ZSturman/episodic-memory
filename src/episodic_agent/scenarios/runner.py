"""Scenario runner for automated testing.

Executes scenarios end-to-end with:
- Unity command client integration (if available)
- JSONL replay fallback
- Metrics collection
- Run folder management
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from episodic_agent.core.orchestrator import AgentOrchestrator
    from episodic_agent.modules.unity.command_client import (
        UnityCommandClient,
        ScenarioCommandSequence,
    )

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for running a scenario."""
    
    name: str
    description: str = ""
    max_steps: int = 500
    fps: float = 10.0
    use_commands: bool = True  # Try Unity commands
    use_replay: bool = True    # Fall back to JSONL replay
    replay_path: Path | None = None
    command_sequence_name: str | None = None
    profile: str = "unity_phase6"
    output_dir: Path = field(default_factory=lambda: Path("runs"))
    auto_label: bool = True
    collect_metrics: bool = True


@dataclass
class ScenarioResult:
    """Result from running a scenario."""
    
    scenario_name: str
    run_id: str
    run_dir: Path
    success: bool
    steps_completed: int
    episodes_created: int
    duration_seconds: float
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "success": self.success,
            "steps_completed": self.steps_completed,
            "episodes_created": self.episodes_created,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "metrics": self.metrics,
        }


class Scenario(ABC):
    """Base class for test scenarios.
    
    Subclasses define specific test scenarios with:
    - Command sequences for Unity
    - Expected behaviors
    - Validation logic
    """

    def __init__(
        self,
        name: str,
        description: str = "",
    ) -> None:
        """Initialize scenario.
        
        Args:
            name: Scenario name.
            description: Scenario description.
        """
        self.name = name
        self.description = description
        self._config: ScenarioConfig | None = None

    @abstractmethod
    def get_command_sequence(self) -> "ScenarioCommandSequence | None":
        """Get the command sequence for this scenario.
        
        Returns:
            Command sequence or None if no commands.
        """
        ...

    @abstractmethod
    def get_replay_path(self) -> Path | None:
        """Get the JSONL replay file path.
        
        Returns:
            Path to replay file or None.
        """
        ...

    def get_config(self) -> ScenarioConfig:
        """Get scenario configuration.
        
        Returns:
            ScenarioConfig for this scenario.
        """
        return ScenarioConfig(
            name=self.name,
            description=self.description,
            max_steps=self.get_max_steps(),
            fps=self.get_target_fps(),
            replay_path=self.get_replay_path(),
        )

    def get_max_steps(self) -> int:
        """Get maximum steps for this scenario."""
        return 500

    def get_target_fps(self) -> float:
        """Get target FPS for this scenario."""
        return 10.0

    def validate_result(
        self,
        result: ScenarioResult,
        orchestrator: "AgentOrchestrator",
    ) -> tuple[bool, list[str]]:
        """Validate scenario result.
        
        Args:
            result: Scenario result.
            orchestrator: Orchestrator with final state.
            
        Returns:
            Tuple of (success, list of validation errors).
        """
        errors = []
        
        # Basic validation
        if result.steps_completed < 10:
            errors.append(f"Too few steps completed: {result.steps_completed}")
        
        if result.episodes_created < 1:
            errors.append("No episodes created")
        
        return (len(errors) == 0, errors)


class ScenarioRunner:
    """Runs scenarios with full orchestrator setup.
    
    Handles:
    - Profile-based module creation
    - Command client integration
    - JSONL replay fallback
    - Metrics collection
    - Run folder management
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        profile: str = "unity_phase6",
        verbose: bool = False,
    ) -> None:
        """Initialize scenario runner.
        
        Args:
            output_dir: Base output directory for runs.
            profile: Profile to use for module creation.
            verbose: Enable verbose logging.
        """
        self._output_dir = output_dir or Path("runs")
        self._profile = profile
        self._verbose = verbose
        
        # Statistics
        self._scenarios_run = 0
        self._scenarios_passed = 0
        self._scenarios_failed = 0

    def run(
        self,
        scenario: Scenario,
        config_override: ScenarioConfig | None = None,
    ) -> ScenarioResult:
        """Run a scenario.
        
        Args:
            scenario: Scenario to run.
            config_override: Optional config override.
            
        Returns:
            ScenarioResult with outcome.
        """
        config = config_override or scenario.get_config()
        config.profile = self._profile
        config.output_dir = self._output_dir
        
        # Generate run ID
        run_id = f"{scenario.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = config.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting scenario: {scenario.name}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Run dir: {run_dir}")
        
        start_time = time.time()
        
        try:
            # Create modules and orchestrator
            orchestrator, modules = self._create_orchestrator(config, run_dir, run_id)
            
            # Get command client if available
            command_client = self._get_command_client(config, modules)
            
            # Get command sequence
            command_sequence = scenario.get_command_sequence()
            if command_sequence and command_client:
                command_sequence.start()
            
            # Run the scenario loop
            steps_completed = 0
            
            from episodic_agent.metrics.logging import LogWriter
            log_path = run_dir / "run.jsonl"
            
            with LogWriter(log_path) as log_writer:
                while steps_completed < config.max_steps:
                    # Check for commands to send
                    if command_sequence and command_client and not command_sequence.is_complete:
                        cmd_result = command_sequence.get_next_command()
                        if cmd_result:
                            cmd, wait = cmd_result
                            if wait <= 0:
                                command_client.send_command(cmd)
                    
                    # Execute one step
                    try:
                        if not orchestrator.has_more_frames():
                            logger.info("No more frames available")
                            break
                        
                        result = orchestrator.step()
                        steps_completed += 1
                        log_writer.write(result)
                        
                        if self._verbose and steps_completed % 50 == 0:
                            logger.info(f"Step {steps_completed}: {result.location_label}")
                        
                    except TimeoutError:
                        if self._verbose:
                            logger.debug("Frame timeout, retrying...")
                        continue
                    except KeyboardInterrupt:
                        logger.info("Interrupted by user")
                        break
                    except Exception as e:
                        logger.warning(f"Step error: {e}")
                        continue
                    
                    # Throttle to target FPS
                    time.sleep(1.0 / config.fps)
            
            duration = time.time() - start_time
            
            # Collect metrics
            metrics = {}
            if config.collect_metrics:
                metrics = self._collect_metrics(orchestrator, modules, run_dir)
            
            # Build result
            result = ScenarioResult(
                scenario_name=scenario.name,
                run_id=run_id,
                run_dir=run_dir,
                success=True,
                steps_completed=steps_completed,
                episodes_created=orchestrator.episode_count,
                duration_seconds=duration,
                metrics=metrics,
            )
            
            # Validate
            success, errors = scenario.validate_result(result, orchestrator)
            if not success:
                result.success = False
                result.error = "; ".join(errors)
            
            self._scenarios_run += 1
            if result.success:
                self._scenarios_passed += 1
            else:
                self._scenarios_failed += 1
            
            # Save result
            self._save_result(result, run_dir)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Scenario failed: {e}")
            
            result = ScenarioResult(
                scenario_name=scenario.name,
                run_id=run_id,
                run_dir=run_dir,
                success=False,
                steps_completed=0,
                episodes_created=0,
                duration_seconds=duration,
                error=str(e),
            )
            
            self._scenarios_run += 1
            self._scenarios_failed += 1
            
            return result
        
        finally:
            # Cleanup
            if 'modules' in locals():
                sensor = modules.get("sensor")
                if hasattr(sensor, "stop"):
                    sensor.stop()

    def _create_orchestrator(
        self,
        config: ScenarioConfig,
        run_dir: Path,
        run_id: str,
    ) -> tuple["AgentOrchestrator", dict[str, Any]]:
        """Create orchestrator with modules.
        
        Args:
            config: Scenario configuration.
            run_dir: Run directory.
            run_id: Run identifier.
            
        Returns:
            Tuple of (orchestrator, modules dict).
        """
        from episodic_agent.core.orchestrator import AgentOrchestrator
        from episodic_agent.utils.profiles import ModuleFactory, get_profile
        
        profile_config = get_profile(config.profile)
        
        factory = ModuleFactory(
            profile=profile_config,
            run_dir=run_dir,
            seed=42,
            auto_label_locations=config.auto_label,
            auto_label_entities=config.auto_label,
        )
        
        modules = factory.create_modules()
        
        orchestrator = AgentOrchestrator(
            sensor=modules["sensor"],
            perception=modules["perception"],
            acf_builder=modules["acf_builder"],
            location_resolver=modules["location_resolver"],
            entity_resolver=modules["entity_resolver"],
            event_resolver=modules["event_resolver"],
            retriever=modules["retriever"],
            boundary_detector=modules["boundary_detector"],
            dialog_manager=modules["dialog_manager"],
            episode_store=modules["episode_store"],
            graph_store=modules["graph_store"],
            run_id=run_id,
        )
        
        return orchestrator, modules

    def _get_command_client(
        self,
        config: ScenarioConfig,
        modules: dict[str, Any],
    ) -> "UnityCommandClient | None":
        """Get command client if available.
        
        Args:
            config: Scenario configuration.
            modules: Created modules.
            
        Returns:
            Command client or None.
        """
        if not config.use_commands:
            return None
        
        try:
            from episodic_agent.modules.unity.command_client import UnityCommandClient
            
            client = UnityCommandClient()
            if client.connect():
                return client
            
        except Exception as e:
            logger.debug(f"Command client not available: {e}")
        
        return None

    def _collect_metrics(
        self,
        orchestrator: "AgentOrchestrator",
        modules: dict[str, Any],
        run_dir: Path,
    ) -> dict[str, Any]:
        """Collect metrics from run.
        
        Args:
            orchestrator: Orchestrator with final state.
            modules: Module instances.
            run_dir: Run directory.
            
        Returns:
            Metrics dictionary.
        """
        from episodic_agent.metrics.evaluation import MetricsCollector
        
        collector = MetricsCollector()
        metrics = collector.collect(orchestrator, modules, run_dir)
        
        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

    def _save_result(
        self,
        result: ScenarioResult,
        run_dir: Path,
    ) -> None:
        """Save scenario result.
        
        Args:
            result: Scenario result.
            run_dir: Run directory.
        """
        result_path = run_dir / "scenario_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_statistics(self) -> dict[str, Any]:
        """Get runner statistics.
        
        Returns:
            Statistics dictionary.
        """
        return {
            "scenarios_run": self._scenarios_run,
            "scenarios_passed": self._scenarios_passed,
            "scenarios_failed": self._scenarios_failed,
            "pass_rate": (
                self._scenarios_passed / self._scenarios_run
                if self._scenarios_run > 0 else 0.0
            ),
        }


# Registry of available scenarios
SCENARIOS: dict[str, type[Scenario]] = {}


def register_scenario(scenario_class: type[Scenario]) -> type[Scenario]:
    """Decorator to register a scenario class."""
    SCENARIOS[scenario_class.__name__] = scenario_class
    return scenario_class


def run_scenario(
    scenario_name: str,
    output_dir: Path | None = None,
    profile: str = "unity_phase6",
    **kwargs: Any,
) -> ScenarioResult:
    """Convenience function to run a scenario by name.
    
    Args:
        scenario_name: Name of the scenario to run.
        output_dir: Output directory.
        profile: Profile to use.
        **kwargs: Additional arguments.
        
    Returns:
        ScenarioResult.
    """
    from episodic_agent.scenarios.definitions import get_scenario
    
    scenario = get_scenario(scenario_name)
    runner = ScenarioRunner(output_dir=output_dir, profile=profile)
    return runner.run(scenario)
