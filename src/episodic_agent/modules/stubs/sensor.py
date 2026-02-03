"""Stub sensor provider for Phase 1 testing."""

from __future__ import annotations

from datetime import datetime

from episodic_agent.core.interfaces import SensorProvider
from episodic_agent.schemas import SensorFrame


class StubSensorProvider(SensorProvider):
    """Synthetic sensor provider that generates deterministic frames.
    
    Produces frames with incrementing frame_id and minimal synthetic
    data. Useful for testing the pipeline without real sensors.
    """

    def __init__(
        self,
        max_frames: int | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize the stub sensor provider.
        
        Args:
            max_frames: Maximum number of frames to produce (None for unlimited).
            seed: Random seed for deterministic behavior.
        """
        self._max_frames = max_frames
        self._seed = seed
        self._frame_count = 0

    def get_frame(self) -> SensorFrame:
        """Get the next synthetic sensor frame.
        
        Returns:
            A SensorFrame with incrementing frame_id.
        """
        self._frame_count += 1
        
        return SensorFrame(
            frame_id=self._frame_count,
            timestamp=datetime.now(),
            raw_data={
                "synthetic": True,
                "seed": self._seed,
                "frame_index": self._frame_count,
            },
            sensor_type="stub",
            extras={},
        )

    def has_frames(self) -> bool:
        """Check if more frames are available.
        
        Returns:
            True if under max_frames limit or unlimited.
        """
        if self._max_frames is None:
            return True
        return self._frame_count < self._max_frames

    def reset(self) -> None:
        """Reset the frame counter to zero."""
        self._frame_count = 0
