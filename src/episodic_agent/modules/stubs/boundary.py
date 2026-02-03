"""Stub boundary detector for Phase 1 testing."""

from __future__ import annotations

from episodic_agent.core.interfaces import BoundaryDetector
from episodic_agent.schemas import ActiveContextFrame
from episodic_agent.utils.config import DEFAULT_FREEZE_INTERVAL


class StubBoundaryDetector(BoundaryDetector):
    """Stub boundary detector that triggers on fixed step intervals.
    
    Freezes an episode every N steps, providing predictable behavior
    for testing and demonstrations.
    """

    def __init__(
        self,
        freeze_interval: int = DEFAULT_FREEZE_INTERVAL,
        seed: int = 42,
    ) -> None:
        """Initialize the stub boundary detector.
        
        Args:
            freeze_interval: Number of steps between episode freezes.
            seed: Random seed (unused in stub, kept for consistency).
        """
        self._freeze_interval = freeze_interval
        self._seed = seed

    def check(self, acf: ActiveContextFrame) -> tuple[bool, str | None]:
        """Check if an episode boundary should be triggered.
        
        Triggers a boundary every freeze_interval steps.
        
        Args:
            acf: Current active context frame.
            
        Returns:
            Tuple of (should_freeze, reason).
        """
        if acf.step_count > 0 and acf.step_count % self._freeze_interval == 0:
            return (True, f"interval_{self._freeze_interval}_steps")
        return (False, None)
