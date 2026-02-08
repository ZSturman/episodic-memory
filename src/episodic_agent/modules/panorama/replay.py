"""JSONL event replay controller for post-hoc run analysis.

Loads a ``events.jsonl`` file produced during a live panorama run and
replays the events through the existing ``PanoramaEventBus`` and
``PanoramaAPIState``, allowing the dashboard to scrub through
historical data without a live agent.

Usage
-----
1.  ``POST /api/replay/load``  – load a JSONL file path
2.  ``POST /api/replay/control`` – play / pause / seek / step / set_speed
3.  ``GET  /api/replay/state``  – current cursor, speed, playing flag
4.  Events flow into the normal event bus → dashboard polls as usual.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from episodic_agent.schemas.panorama_events import PanoramaEvent

logger = logging.getLogger(__name__)


class ReplayController:
    """Controls playback of a JSONL event log.

    Parameters
    ----------
    event_bus : PanoramaEventBus
        Live event bus – replayed events are emitted here.
    state_writer : callable, optional
        Called with (event: PanoramaEvent) on each tick so the API
        state can be updated from event payload data.
    """

    def __init__(
        self,
        event_bus: Any = None,
        state_writer: Callable[[PanoramaEvent], None] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._state_writer = state_writer
        self._lock = threading.RLock()

        # Event store
        self._events: list[PanoramaEvent] = []
        self._file_path: str = ""

        # Playback state
        self._cursor: int = 0
        self._playing: bool = False
        self._speed: float = 1.0  # 1× = original timing
        self._loaded: bool = False

        # Background playback thread
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> int:
        """Load events from a JSONL file.

        Returns the number of events loaded.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If no valid events could be parsed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        events: list[PanoramaEvent] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    events.append(PanoramaEvent(**record))
                except Exception as exc:
                    logger.warning("Replay: skip line %d: %s", lineno, exc)

        if not events:
            raise ValueError(f"No valid events in {file_path}")

        with self._lock:
            self.stop()
            self._events = events
            self._file_path = str(path)
            self._cursor = 0
            self._playing = False
            self._loaded = True

        logger.info("Replay: loaded %d events from %s", len(events), file_path)
        return len(events)

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def play(self) -> None:
        """Start or resume playback."""
        with self._lock:
            if not self._loaded or self._playing:
                return
            self._playing = True
            self._stop_flag.clear()
            self._thread = threading.Thread(
                target=self._playback_loop,
                daemon=True,
                name="replay-controller",
            )
            self._thread.start()

    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            self._playing = False
            self._stop_flag.set()

    def stop(self) -> None:
        """Stop playback and reset cursor."""
        with self._lock:
            self._playing = False
            self._stop_flag.set()
            self._cursor = 0

    def step_forward(self) -> PanoramaEvent | None:
        """Advance one event and emit it.  Returns the event or None."""
        with self._lock:
            if not self._loaded or self._cursor >= len(self._events):
                return None
            event = self._events[self._cursor]
            self._cursor += 1
        self._emit(event)
        return event

    def step_back(self) -> None:
        """Move cursor back one event (does NOT undo side-effects)."""
        with self._lock:
            if self._cursor > 0:
                self._cursor -= 1

    def seek(self, position: int) -> None:
        """Jump to a specific event index (0-based).

        Emits all events from 0 → position so that state is consistent.
        """
        with self._lock:
            was_playing = self._playing
            if was_playing:
                self._playing = False
                self._stop_flag.set()

            position = max(0, min(position, len(self._events)))
            # Reset bus if we have one
            if self._event_bus and hasattr(self._event_bus, "clear"):
                self._event_bus.clear()
            self._cursor = 0

        # Replay from 0 → position
        for _ in range(position):
            self.step_forward()

        if was_playing:
            self.play()

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier (0.25 – 10×)."""
        self._speed = max(0.25, min(speed, 10.0))

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return current replay state as a JSON-serializable dict."""
        with self._lock:
            return {
                "loaded": self._loaded,
                "file": self._file_path,
                "total_events": len(self._events),
                "cursor": self._cursor,
                "playing": self._playing,
                "speed": self._speed,
            }

    def get_events_up_to_cursor(self) -> list[PanoramaEvent]:
        """Return all events from 0 up to current cursor."""
        with self._lock:
            return list(self._events[: self._cursor])

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total_events(self) -> int:
        return len(self._events)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _playback_loop(self) -> None:
        """Background thread that emits events at the configured speed."""
        base_interval = 0.05  # 50ms between events at 1×
        while not self._stop_flag.is_set():
            with self._lock:
                if self._cursor >= len(self._events):
                    self._playing = False
                    break
                event = self._events[self._cursor]
                self._cursor += 1

            self._emit(event)

            # Compute inter-event delay
            delay = base_interval / max(self._speed, 0.25)
            self._stop_flag.wait(delay)

    def _emit(self, event: PanoramaEvent) -> None:
        """Push an event through the bus and optional state writer."""
        if self._event_bus and hasattr(self._event_bus, "emit"):
            self._event_bus.emit(event)
        if self._state_writer:
            try:
                self._state_writer(event)
            except Exception as exc:
                logger.warning("Replay state_writer error: %s", exc)
