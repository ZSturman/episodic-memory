"""Agent control — thread-safe state for pause / step / advance.

Used by the CLI loop and the API server to coordinate agent execution.
The dashboard sends commands via API endpoints; the CLI loop checks
these flags between steps.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any


class UserResponse(str, Enum):
    """Possible user responses after image investigation."""

    CONFIRM = "confirm"              # Yes, this is the correct label
    REJECT = "reject"                # Wrong — provide a new label
    SAME_PLACE_DIFFERENT = "same_place_different"  # Same location, add variant
    NEW_LABEL = "new_label"          # Provide label for unknown location
    SKIP = "skip"                    # Skip this image
    PENDING = "pending"              # No response yet


class AgentControl:
    """Thread-safe control interface between API server and CLI loop.

    The CLI loop calls ``wait_if_paused()`` between steps and
    ``wait_for_user_response()`` when a label is needed.  The
    dashboard sets flags via the API server.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # starts un-paused (event is "set" = proceed)

        self._step_event = threading.Event()
        self._advance_event = threading.Event()

        # User response for label confirmation
        self._user_response = UserResponse.PENDING
        self._user_label: str = ""
        self._user_variant: str = ""
        self._user_response_event = threading.Event()

        # Focus profile override from dashboard
        self._focus_override: dict[str, Any] | None = None
        self._focus_override_event = threading.Event()

        # Auto-focus toggle
        self._auto_focus = True

        # State for UI
        self._paused = False
        self._awaiting_user = False

    # ----------------------------------------------------------------
    # Pause / resume
    # ----------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Pause the agent loop."""
        self._paused = True
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume the agent loop."""
        self._paused = False
        self._pause_event.set()

    def toggle_pause(self) -> bool:
        """Toggle pause state, returns new state."""
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused

    def wait_if_paused(self, timeout: float = 0.5) -> bool:
        """Block if paused.  Returns True if a step was requested.

        Called by the CLI loop between steps.  If paused, blocks until
        either resumed, a single step is requested, or timeout expires.
        """
        if not self._paused:
            return False

        # Wait for either un-pause or step request
        while self._paused:
            # Check for single step
            if self._step_event.is_set():
                self._step_event.clear()
                return True
            self._pause_event.wait(timeout=timeout)
            if self._pause_event.is_set():
                break

        return False

    def request_step(self) -> None:
        """Request a single step while paused."""
        self._step_event.set()

    # ----------------------------------------------------------------
    # Image advance
    # ----------------------------------------------------------------

    @property
    def awaiting_user(self) -> bool:
        return self._awaiting_user

    def request_advance(self) -> None:
        """Signal that we should advance to the next image."""
        self._advance_event.set()

    def check_advance(self) -> bool:
        """Check if advance was requested (and clear the flag)."""
        if self._advance_event.is_set():
            self._advance_event.clear()
            return True
        return False

    # ----------------------------------------------------------------
    # User label response
    # ----------------------------------------------------------------

    def submit_user_response(
        self,
        response: str,
        label: str = "",
        variant: str = "",
    ) -> None:
        """Submit a user response (from dashboard or CLI)."""
        with self._lock:
            self._user_response = UserResponse(response)
            self._user_label = label
            self._user_variant = variant
            self._awaiting_user = False
        self._user_response_event.set()

    def wait_for_user_response(self, timeout: float | None = None) -> dict[str, str]:
        """Block until a user response is received.

        Called by the CLI loop when a label/confirmation is needed.

        Returns
        -------
        dict with keys: response, label, variant
        """
        with self._lock:
            self._awaiting_user = True
            self._user_response = UserResponse.PENDING
            self._user_response_event.clear()

        self._user_response_event.wait(timeout=timeout)

        with self._lock:
            result = {
                "response": self._user_response.value,
                "label": self._user_label,
                "variant": self._user_variant,
            }
            self._awaiting_user = False
            return result

    def has_pending_response(self) -> bool:
        return self._user_response_event.is_set()

    # ----------------------------------------------------------------
    # Focus profile
    # ----------------------------------------------------------------

    @property
    def auto_focus(self) -> bool:
        return self._auto_focus

    def set_auto_focus(self, enabled: bool) -> None:
        self._auto_focus = enabled

    def set_focus_override(self, profile_dict: dict[str, Any]) -> None:
        """Set a focus profile override from the dashboard."""
        with self._lock:
            self._focus_override = profile_dict
        self._focus_override_event.set()

    def get_focus_override(self) -> dict[str, Any] | None:
        """Get and clear any pending focus override."""
        if not self._focus_override_event.is_set():
            return None
        self._focus_override_event.clear()
        with self._lock:
            result = self._focus_override
            self._focus_override = None
            return result

    # ----------------------------------------------------------------
    # Status for API
    # ----------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "awaiting_user": self._awaiting_user,
            "auto_focus": self._auto_focus,
            "user_response": self._user_response.value,
        }
