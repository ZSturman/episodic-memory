"""JSONL logging utilities for run output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episodic_agent.schemas import StepResult


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
