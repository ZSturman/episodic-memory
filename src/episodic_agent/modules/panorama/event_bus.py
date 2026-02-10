"""Thread-safe event bus for panorama agent observability.

Provides a central ring buffer of ``PanoramaEvent`` objects that
modules emit and the API server / UI consume.  Designed as a
lightweight, dependency-free pub/sub backbone.

Usage::

    from episodic_agent.modules.panorama.event_bus import PanoramaEventBus

    bus = PanoramaEventBus()
    bus.emit(PanoramaEvent(...))

    # API polling
    events = bus.get_events(since_step=42)
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any, Callable

from episodic_agent.schemas.panorama_events import (
    PanoramaAgentState,
    PanoramaEvent,
    PanoramaEventType,
)

logger = logging.getLogger(__name__)


class PanoramaEventBus:
    """Thread-safe ring buffer of structured panorama events.

    Parameters
    ----------
    max_events : int
        Maximum events retained in memory (oldest are dropped).
    """

    def __init__(self, max_events: int = 1000) -> None:
        self._lock = threading.Lock()
        self._events: deque[PanoramaEvent] = deque(maxlen=max_events)
        self._subscribers: list[Callable[[PanoramaEvent], None]] = []
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def emit(self, event: PanoramaEvent) -> None:
        """Append an event to the buffer and notify subscribers."""
        with self._lock:
            self._events.append(event)
            self._step_counter = max(self._step_counter, event.step)

        # Log at DEBUG for JSONL auditability
        logger.debug(
            "PanoramaEvent: type=%s step=%d state=%s",
            event.event_type,
            event.step,
            event.state,
        )

        # Notify subscribers (outside lock to avoid deadlock)
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception as e:
                logger.warning("Event subscriber error: %s", e)

    def emit_simple(
        self,
        event_type: PanoramaEventType,
        step: int,
        state: PanoramaAgentState,
        payload: dict[str, Any] | None = None,
    ) -> PanoramaEvent:
        """Convenience: build and emit a PanoramaEvent in one call."""
        event = PanoramaEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            step=step,
            state=state,
            payload=payload or {},
        )
        self.emit(event)
        return event

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_events(self, since_step: int = 0) -> list[PanoramaEvent]:
        """Return all events with step >= since_step."""
        with self._lock:
            return [e for e in self._events if e.step >= since_step]

    def get_latest(self, n: int = 10) -> list[PanoramaEvent]:
        """Return the last N events."""
        with self._lock:
            items = list(self._events)
            return items[-n:] if len(items) >= n else items

    def get_events_by_type(
        self,
        event_type: PanoramaEventType,
        limit: int = 50,
    ) -> list[PanoramaEvent]:
        """Return recent events of a specific type."""
        with self._lock:
            matching = [e for e in self._events if e.event_type == event_type]
            return matching[-limit:]

    @property
    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def latest_step(self) -> int:
        with self._lock:
            return self._step_counter

    # ------------------------------------------------------------------
    # Subscriptions (for future WebSocket push)
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[PanoramaEvent], None]) -> None:
        """Register a callback invoked on every emit."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[PanoramaEvent], None]) -> None:
        """Remove a previously registered callback."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Hex-specific convenience methods
    # ------------------------------------------------------------------

    def emit_hex(
        self,
        event_type: PanoramaEventType,
        step: int,
        payload: dict[str, Any] | None = None,
    ) -> PanoramaEvent:
        """Emit a hex-scan event using scanning_image state."""
        return self.emit_simple(
            event_type=event_type,
            step=step,
            state=PanoramaAgentState.scanning_image,
            payload=payload or {},
        )

    def get_hex_events(self, limit: int = 50) -> list[PanoramaEvent]:
        """Return recent hex-related events."""
        hex_types = {
            PanoramaEventType.hex_scan_started,
            PanoramaEventType.hex_scan_pass,
            PanoramaEventType.hex_scan_complete,
            PanoramaEventType.focus_update,
            PanoramaEventType.user_label_confirmed,
            PanoramaEventType.user_label_rejected,
            PanoramaEventType.image_advanced,
        }
        with self._lock:
            matching = [e for e in self._events if e.event_type in hex_types]
            return matching[-limit:]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all events (for testing)."""
        with self._lock:
            self._events.clear()
            self._step_counter = 0

    def snapshot(self) -> list[dict[str, Any]]:
        """Return all events as serialisable dicts."""
        with self._lock:
            return [e.model_dump(mode="json") for e in self._events]
