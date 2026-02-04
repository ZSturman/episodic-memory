"""Structured logging for the episodic memory system.

ARCHITECTURAL INVARIANT: All decisions logged with context for post-run analysis.
Logs are reproducible and traceable.

This module provides:
- LogCategory: Predefined log categories for consistent filtering
- StructuredLogger: Category-prefixed logging with timestamps and context
- SessionLogger: Session-aware logging to files
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

# =============================================================================
# LOG CATEGORIES
# =============================================================================

class LogCategory(str, Enum):
    """Log categories for structured filtering and analysis.
    
    ARCHITECTURAL INVARIANT: Categories are structural, not semantic.
    They describe system components, not content types.
    """
    SENSOR = "SENSOR"              # Sensor data reception and processing
    MEMORY = "MEMORY"              # Memory storage and retrieval operations
    RECOGNITION = "RECOGNITION"    # Recognition and labeling decisions
    CONSOLIDATION = "CONSOLIDATION"  # Memory consolidation operations
    LABEL = "LABEL"                # Label learning and user feedback
    SPATIAL = "SPATIAL"            # Spatial context and landmark operations
    PROTOCOL = "PROTOCOL"          # Wire protocol messages
    ARBITRATION = "ARBITRATION"    # Motion/perception arbitration
    VISUAL = "VISUAL"              # Visual channel processing
    EVENT = "EVENT"                # Event detection and recognition
    RECALL = "RECALL"              # Cued recall operations
    HYPOTHESIS = "HYPOTHESIS"      # Entity hypotheses
    INVARIANT = "INVARIANT"        # Invariant checks and violations
    SYSTEM = "SYSTEM"              # System-level operations


# =============================================================================
# LOG LEVELS
# =============================================================================

class LogLevel(str, Enum):
    """Log levels for filtering."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Map to standard logging levels
_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


# =============================================================================
# LOG ENTRY
# =============================================================================

class LogEntry:
    """A structured log entry.
    
    Contains category, level, message, and optional context fields
    for rich structured logging.
    """
    
    def __init__(
        self,
        category: LogCategory,
        level: LogLevel,
        message: str,
        frame_id: int | None = None,
        confidence: float | None = None,
        entity_id: str | None = None,
        location_id: str | None = None,
        event_type: str | None = None,
        extras: dict[str, Any] | None = None,
        **kwargs: Any,  # Accept arbitrary context
    ):
        self.timestamp = datetime.now()
        self.category = category
        self.level = level
        self.message = message
        self.frame_id = frame_id
        self.confidence = confidence
        self.entity_id = entity_id
        self.location_id = location_id
        self.event_type = event_type
        self.extras = extras or {}
        # Store arbitrary context from kwargs
        self.context = dict(kwargs)
        # Also add named fields to context
        if frame_id is not None:
            self.context["frame_id"] = frame_id
        if confidence is not None:
            self.context["confidence"] = confidence
        if entity_id is not None:
            self.context["entity_id"] = entity_id
        if location_id is not None:
            self.context["location_id"] = location_id
        if event_type is not None:
            self.context["event_type"] = event_type
        self.entity_id = entity_id
        self.location_id = location_id
        self.event_type = event_type
        self.extras = extras or {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "level": self.level.value,
            "message": self.message,
        }
        if self.frame_id is not None:
            d["frame_id"] = self.frame_id
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.entity_id is not None:
            d["entity_id"] = self.entity_id
        if self.location_id is not None:
            d["location_id"] = self.location_id
        if self.event_type is not None:
            d["event_type"] = self.event_type
        if self.extras:
            d["extras"] = self.extras
        if self.context:
            d["context"] = self.context
        return d
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def format_console(self) -> str:
        """Format for console output with category prefix."""
        ts = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{self.category.value}]"
        
        # Add optional context
        context_parts = []
        if self.frame_id is not None:
            context_parts.append(f"frame={self.frame_id}")
        if self.confidence is not None:
            context_parts.append(f"conf={self.confidence:.2f}")
        if self.entity_id:
            context_parts.append(f"entity={self.entity_id[:8]}")
        if self.location_id:
            context_parts.append(f"loc={self.location_id[:8]}")
        
        context_str = f" ({', '.join(context_parts)})" if context_parts else ""
        
        return f"{ts} {prefix:16} {self.message}{context_str}"


# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

class StructuredLogger:
    """Category-prefixed structured logger.
    
    ARCHITECTURAL INVARIANT: All log entries include category prefix
    for consistent filtering and post-run analysis.
    """
    
    def __init__(
        self,
        name: str = "episodic_agent",
        level: LogLevel = LogLevel.INFO,
        min_level: LogLevel | None = None,  # Alias for level
        console_output: bool = True,
        file_output: TextIO | None = None,
        json_output: bool = False,
    ):
        """Initialize the structured logger.
        
        Args:
            name: Logger name
            level: Minimum log level
            min_level: Alias for level (for compatibility)
            console_output: Whether to output to console
            file_output: Optional file handle for output
            json_output: Whether to use JSON format for file output
        """
        self._name = name
        self._level = min_level if min_level is not None else level
        self._console_output = console_output
        self._file_output = file_output
        self._json_output = json_output
        
        # Statistics
        self._counts: dict[LogCategory, int] = {cat: 0 for cat in LogCategory}
        self._error_count = 0
        self._warning_count = 0
        
        # Entry history for filtering/retrieval
        self._entries: list[LogEntry] = []
        self._max_history = 10000
        
        # Category filters (None = all enabled)
        self._enabled_categories: set[LogCategory] | None = None
        self._disabled_categories: set[LogCategory] = set()
    
    def log(
        self,
        category: LogCategory,
        level: LogLevel,
        message: str,
        **kwargs: Any,
    ) -> LogEntry:
        """Log a structured entry.
        
        Args:
            category: Log category
            level: Log level
            message: Log message
            **kwargs: Additional context (frame_id, confidence, etc.)
            
        Returns:
            The created LogEntry
        """
        # Create entry first (even if filtered, for return value)
        entry = LogEntry(category, level, message, **kwargs)
        
        # Check level
        if _LEVEL_MAP[level] < _LEVEL_MAP[self._level]:
            return entry
        
        # Check category filters
        if self._enabled_categories is not None:
            if category not in self._enabled_categories:
                return entry
        if category in self._disabled_categories:
            return entry
        
        # Update statistics
        self._counts[category] += 1
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            self._error_count += 1
        elif level == LogLevel.WARNING:
            self._warning_count += 1
        
        # Store in history
        self._entries.append(entry)
        if len(self._entries) > self._max_history:
            self._entries = self._entries[-self._max_history:]
        
        # Output
        if self._console_output:
            self._write_console(entry)
        
        if self._file_output:
            self._write_file(entry)
        
        return entry
    
    def _write_console(self, entry: LogEntry) -> None:
        """Write entry to console."""
        output = entry.format_console()
        
        # Color by level (if terminal supports it)
        if sys.stdout.isatty():
            colors = {
                LogLevel.DEBUG: "\033[90m",    # Gray
                LogLevel.INFO: "\033[0m",       # Default
                LogLevel.WARNING: "\033[93m",   # Yellow
                LogLevel.ERROR: "\033[91m",     # Red
                LogLevel.CRITICAL: "\033[91;1m",  # Bold red
            }
            reset = "\033[0m"
            output = f"{colors.get(entry.level, '')}{output}{reset}"
        
        print(output)
    
    def _write_file(self, entry: LogEntry) -> None:
        """Write entry to file."""
        if self._json_output:
            self._file_output.write(entry.to_json() + "\n")
        else:
            self._file_output.write(entry.format_console() + "\n")
        self._file_output.flush()
    
    # -------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # -------------------------------------------------------------------------
    
    def debug(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        """Log at DEBUG level."""
        return self.log(category, LogLevel.DEBUG, message, **kwargs)
    
    def info(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        """Log at INFO level."""
        return self.log(category, LogLevel.INFO, message, **kwargs)
    
    def warning(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        """Log at WARNING level."""
        return self.log(category, LogLevel.WARNING, message, **kwargs)
    
    def error(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        """Log at ERROR level."""
        return self.log(category, LogLevel.ERROR, message, **kwargs)
    
    def critical(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        """Log at CRITICAL level."""
        return self.log(category, LogLevel.CRITICAL, message, **kwargs)
    
    # -------------------------------------------------------------------------
    # CATEGORY-SPECIFIC METHODS
    # -------------------------------------------------------------------------
    
    def sensor(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        """Log sensor data."""
        return self.log(LogCategory.SENSOR, level, message, **kwargs)
    
    def memory(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        """Log memory operations."""
        return self.log(LogCategory.MEMORY, level, message, **kwargs)
    
    def recognition(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log recognition decisions."""
        return self.log(LogCategory.RECOGNITION, level, message, **kwargs)
    
    def consolidation(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log consolidation operations."""
        return self.log(LogCategory.CONSOLIDATION, level, message, **kwargs)
    
    def label(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log label learning."""
        return self.log(LogCategory.LABEL, level, message, **kwargs)
    
    def spatial(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        """Log spatial operations."""
        return self.log(LogCategory.SPATIAL, level, message, **kwargs)
    
    def protocol(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        """Log protocol messages."""
        return self.log(LogCategory.PROTOCOL, level, message, **kwargs)
    
    def arbitration(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log arbitration decisions."""
        return self.log(LogCategory.ARBITRATION, level, message, **kwargs)
    
    def visual(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        """Log visual processing."""
        return self.log(LogCategory.VISUAL, level, message, **kwargs)
    
    def event(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log event detection."""
        return self.log(LogCategory.EVENT, level, message, **kwargs)
    
    def recall(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log recall operations."""
        return self.log(LogCategory.RECALL, level, message, **kwargs)
    
    def hypothesis(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log entity hypotheses."""
        return self.log(LogCategory.HYPOTHESIS, level, message, **kwargs)
    
    def invariant(self, message: str, level: LogLevel = LogLevel.WARNING, **kwargs: Any) -> LogEntry:
        """Log invariant checks."""
        return self.log(LogCategory.INVARIANT, level, message, **kwargs)
    
    def system(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        """Log system operations."""
        return self.log(LogCategory.SYSTEM, level, message, **kwargs)
    
    # -------------------------------------------------------------------------
    # INVARIANT LOGGING
    # -------------------------------------------------------------------------
    
    def check_invariant(
        self,
        condition: bool,
        invariant_name: str,
        message: str,
        **kwargs: Any,
    ) -> LogEntry:
        """Check and log an invariant.
        
        Args:
            condition: Whether the invariant holds
            invariant_name: Name of the invariant being checked
            message: Description of the check
            **kwargs: Additional context
            
        Returns:
            The created LogEntry
        """
        if condition:
            return self.invariant(
                f"PASS: {invariant_name} - {message}",
                level=LogLevel.INFO,
                extras={"invariant": invariant_name, "result": "pass", **kwargs.get("extras", {})},
                **{k: v for k, v in kwargs.items() if k != "extras"},
            )
        else:
            return self.invariant(
                f"FAIL: {invariant_name} - {message}",
                level=LogLevel.ERROR,
                extras={"invariant": invariant_name, "result": "fail", **kwargs.get("extras", {})},
                **{k: v for k, v in kwargs.items() if k != "extras"},
            )
    
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    
    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level."""
        self._level = level
    
    def enable_categories(self, categories: list[LogCategory]) -> None:
        """Enable only specific categories."""
        self._enabled_categories = set(categories)
    
    def set_category_filter(self, categories: list[LogCategory]) -> None:
        """Set which categories are enabled (alias for enable_categories)."""
        self._enabled_categories = set(categories)
    
    def disable_categories(self, categories: list[LogCategory]) -> None:
        """Disable specific categories."""
        self._disabled_categories.update(categories)
    
    def enable_all_categories(self) -> None:
        """Enable all categories."""
        self._enabled_categories = None
        self._disabled_categories.clear()
    
    def set_file_output(self, file_output: TextIO | None, json_format: bool = False) -> None:
        """Set file output."""
        self._file_output = file_output
        self._json_output = json_format
    
    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_entries": sum(self._counts.values()),
            "total": sum(self._counts.values()),
            "by_category": {cat: count for cat, count in self._counts.items()},
            "errors": self._error_count,
            "warnings": self._warning_count,
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._counts = {cat: 0 for cat in LogCategory}
        self._error_count = 0
        self._warning_count = 0
    
    def get_recent_entries(self, count: int = 100) -> list[LogEntry]:
        """Get the most recent log entries.
        
        Args:
            count: Number of entries to return
            
        Returns:
            List of most recent LogEntry objects
        """
        return self._entries[-count:]
    
    def filter_by_category(self, category: LogCategory) -> list[LogEntry]:
        """Get all entries of a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of LogEntry objects matching the category
        """
        return [e for e in self._entries if e.category == category]


# =============================================================================
# SESSION LOGGER
# =============================================================================

class SessionLogger:
    """Session-aware logger that writes to session directory.
    
    Creates log files in runs/<session_id>/logs/ with proper structure.
    """
    
    def __init__(
        self,
        session_id: str,
        runs_dir: str | Path = "runs",
        console_output: bool = True,
        level: LogLevel = LogLevel.INFO,
    ):
        """Initialize session logger.
        
        Args:
            session_id: Unique session identifier
            runs_dir: Base directory for runs
            console_output: Whether to output to console
            level: Minimum log level
        """
        self._session_id = session_id
        self._runs_dir = Path(runs_dir)
        
        # Create session directory structure
        self._session_dir = self._runs_dir / session_id
        self._logs_dir = self._session_dir / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log files
        self._main_log_path = self._logs_dir / "main.log"
        self._json_log_path = self._logs_dir / "main.jsonl"
        
        self._main_log_file = open(self._main_log_path, "a", encoding="utf-8")
        self._json_log_file = open(self._json_log_path, "a", encoding="utf-8")
        
        # Create structured loggers
        self._text_logger = StructuredLogger(
            name=f"session_{session_id}",
            level=level,
            console_output=console_output,
            file_output=self._main_log_file,
            json_output=False,
        )
        
        self._json_logger = StructuredLogger(
            name=f"session_{session_id}_json",
            level=level,
            console_output=False,
            file_output=self._json_log_file,
            json_output=True,
        )
        
        # Log session start
        self._text_logger.system(f"Session started: {session_id}")
        self._json_logger.system(f"Session started: {session_id}")
    
    def log(
        self,
        category: LogCategory,
        level: LogLevel,
        message: str,
        **kwargs: Any,
    ) -> LogEntry:
        """Log to both text and JSON outputs."""
        self._text_logger.log(category, level, message, **kwargs)
        return self._json_logger.log(category, level, message, **kwargs)
    
    # Forward convenience methods
    def debug(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        return self.log(category, LogLevel.DEBUG, message, **kwargs)
    
    def info(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        return self.log(category, LogLevel.INFO, message, **kwargs)
    
    def warning(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        return self.log(category, LogLevel.WARNING, message, **kwargs)
    
    def error(self, category: LogCategory, message: str, **kwargs: Any) -> LogEntry:
        return self.log(category, LogLevel.ERROR, message, **kwargs)
    
    # Forward category-specific methods
    def sensor(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.SENSOR, level, message, **kwargs)
    
    def memory(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.MEMORY, level, message, **kwargs)
    
    def recognition(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.RECOGNITION, level, message, **kwargs)
    
    def consolidation(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.CONSOLIDATION, level, message, **kwargs)
    
    def label(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.LABEL, level, message, **kwargs)
    
    def spatial(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.SPATIAL, level, message, **kwargs)
    
    def protocol(self, message: str, level: LogLevel = LogLevel.DEBUG, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.PROTOCOL, level, message, **kwargs)
    
    def arbitration(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.ARBITRATION, level, message, **kwargs)
    
    def event(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.EVENT, level, message, **kwargs)
    
    def recall(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.RECALL, level, message, **kwargs)
    
    def hypothesis(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.HYPOTHESIS, level, message, **kwargs)
    
    def invariant(self, message: str, level: LogLevel = LogLevel.WARNING, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.INVARIANT, level, message, **kwargs)
    
    def system(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any) -> LogEntry:
        return self.log(LogCategory.SYSTEM, level, message, **kwargs)
    
    def check_invariant(
        self,
        condition: bool,
        invariant_name: str,
        message: str,
        **kwargs: Any,
    ) -> bool:
        """Check and log an invariant."""
        self._text_logger.check_invariant(condition, invariant_name, message, **kwargs)
        return self._json_logger.check_invariant(condition, invariant_name, message, **kwargs)
    
    def set_level(self, level: LogLevel) -> None:
        """Set minimum log level."""
        self._text_logger.set_level(level)
        self._json_logger.set_level(level)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get logging statistics."""
        return self._json_logger.get_statistics()
    
    def close(self) -> None:
        """Close log files."""
        self._text_logger.system(f"Session ended: {self._session_id}")
        self._json_logger.system(f"Session ended: {self._session_id}")
        self._main_log_file.close()
        self._json_log_file.close()
    
    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id
    
    @property
    def session_dir(self) -> Path:
        """Get session directory path."""
        return self._session_dir
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self._logs_dir


# =============================================================================
# GLOBAL LOGGER INSTANCE
# =============================================================================

# Default global logger (can be replaced with session logger)
_global_logger: StructuredLogger | SessionLogger | None = None


def get_logger() -> StructuredLogger | SessionLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger()
    return _global_logger


def set_logger(logger: StructuredLogger | SessionLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger


def create_session_logger(
    session_id: str | None = None,
    runs_dir: str | Path = "runs",
    console_output: bool = True,
    level: LogLevel = LogLevel.INFO,
) -> SessionLogger:
    """Create and set a session logger as the global logger.
    
    Args:
        session_id: Session ID (generated if not provided)
        runs_dir: Base directory for runs
        console_output: Whether to output to console
        level: Minimum log level
        
    Returns:
        The created session logger
    """
    global _global_logger
    
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger = SessionLogger(
        session_id=session_id,
        runs_dir=runs_dir,
        console_output=console_output,
        level=level,
    )
    
    _global_logger = logger
    return logger


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LogCategory",
    "LogLevel",
    # Classes
    "LogEntry",
    "StructuredLogger",
    "SessionLogger",
    # Functions
    "get_logger",
    "set_logger",
    "create_session_logger",
]
