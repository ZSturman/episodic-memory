"""Scenario framework for automated end-to-end testing.

Provides named scenarios that can be run with either:
- Live Unity connection (via command client)
- JSONL replay (for deterministic testing without Unity)

Each scenario produces:
- A run folder with logs, episodes, metrics
- A consistent run ID
"""

from episodic_agent.scenarios.runner import (
    Scenario,
    ScenarioRunner,
    ScenarioResult,
    SCENARIOS,
    run_scenario,
)
from episodic_agent.scenarios.definitions import (
    WalkRoomsScenario,
    ToggleDrawerLightScenario,
    SpawnMoveBallScenario,
    MixedScenario,
)

__all__ = [
    "Scenario",
    "ScenarioRunner",
    "ScenarioResult",
    "SCENARIOS",
    "run_scenario",
    "WalkRoomsScenario",
    "ToggleDrawerLightScenario",
    "SpawnMoveBallScenario",
    "MixedScenario",
]
