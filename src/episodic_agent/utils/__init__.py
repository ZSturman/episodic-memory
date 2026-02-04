"""Utility functions and configuration."""

from episodic_agent.utils.config import DEFAULT_EMBEDDING_DIM, DEFAULT_FREEZE_INTERVAL
from episodic_agent.utils.logging import (
    create_session_logger,
    get_logger,
    LogCategory,
    LogEntry,
    LogLevel,
    SessionLogger,
    set_logger,
    StructuredLogger,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_FREEZE_INTERVAL",
    # Phase 7: Structured Logging
    "create_session_logger",
    "get_logger",
    "LogCategory",
    "LogEntry",
    "LogLevel",
    "SessionLogger",
    "set_logger",
    "StructuredLogger",
]
