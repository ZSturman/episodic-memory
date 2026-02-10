"""Multi-pass adaptive hex scanner.

Manages iterative scanning of a single image:

1. **Pass 1 (coarse)**: Uniform detail level, identifies high-detail
   regions via interest scores.
2. **Pass 2+ (refinement)**: Backend-recommended ``FocusProfile``
   concentrates detail on interesting regions.  Converges when the
   interest distribution stabilises or ``max_passes`` is reached.

Each pass produces a ``HexScanResult``.  The final pass becomes the
authoritative scan used for location embedding / matching.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np

from episodic_agent.modules.panorama.hex_cell import FocusProfile, HexCoord
from episodic_agent.modules.panorama.hex_feature_extractor import (
    HexFeatureExtractor,
    HexScanResult,
)

logger = logging.getLogger(__name__)


class ScanState(str, Enum):
    """Scanner lifecycle state."""

    IDLE = "idle"
    SCANNING = "scanning"
    AWAITING_FOCUS = "awaiting_focus"
    CONVERGED = "converged"
    AWAITING_USER = "awaiting_user"


class HexScanner:
    """Adaptive multi-pass hex scanner for a single image.

    Parameters
    ----------
    num_columns : int
        Hex density (approximate columns across width).
    max_passes : int
        Maximum scan passes before forced convergence.
    convergence_threshold : float
        Interest-stability threshold.  When the change in normalised
        top-K interest scores between passes is below this, scanning
        is considered converged.
    auto_focus : bool
        If True, automatically compute and apply the recommended
        focus profile after each pass.  If False, wait for an
        externally provided profile (dashboard / backend).
    """

    def __init__(
        self,
        num_columns: int = 20,
        max_passes: int = 5,
        convergence_threshold: float = 0.02,
        auto_focus: bool = True,
    ) -> None:
        self.num_columns = num_columns
        self.max_passes = max_passes
        self.convergence_threshold = convergence_threshold
        self.auto_focus = auto_focus

        self._extractor = HexFeatureExtractor(num_columns=num_columns)
        self._current_image: np.ndarray | None = None
        self._current_profile: FocusProfile | None = None
        self._scan_history: list[HexScanResult] = []
        self._state = ScanState.IDLE
        self._pass_number = 0
        self._prev_interest_hash: float = 0.0

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def state(self) -> ScanState:
        return self._state

    @property
    def current_pass(self) -> int:
        return self._pass_number

    @property
    def scan_history(self) -> list[HexScanResult]:
        return list(self._scan_history)

    @property
    def latest_scan(self) -> HexScanResult | None:
        return self._scan_history[-1] if self._scan_history else None

    @property
    def current_profile(self) -> FocusProfile | None:
        return self._current_profile

    def is_converged(self) -> bool:
        return self._state == ScanState.CONVERGED

    # ----------------------------------------------------------------
    # Full scan (blocking)
    # ----------------------------------------------------------------

    def scan_image(
        self,
        img_rgb: np.ndarray,
        initial_profile: FocusProfile | None = None,
    ) -> HexScanResult:
        """Run all passes to convergence (blocking).

        Returns the final authoritative scan result.
        """
        self.begin_image(img_rgb, initial_profile)

        while not self.is_converged():
            self.scan_step()
            if self.auto_focus and not self.is_converged():
                # Auto-apply recommended focus
                if self.latest_scan is not None:
                    self._current_profile = self.latest_scan.recommend_focus(
                        self._current_profile
                    )

        return self.latest_scan  # type: ignore[return-value]

    # ----------------------------------------------------------------
    # Step-by-step scanning (for dashboard control)
    # ----------------------------------------------------------------

    def begin_image(
        self,
        img_rgb: np.ndarray,
        initial_profile: FocusProfile | None = None,
    ) -> None:
        """Prepare for a new image.  Call ``scan_step()`` to advance."""
        self._current_image = img_rgb
        self._current_profile = initial_profile
        self._scan_history.clear()
        self._pass_number = 0
        self._prev_interest_hash = 0.0
        self._state = ScanState.SCANNING

    def scan_step(self, profile_override: FocusProfile | None = None) -> HexScanResult:
        """Execute one scan pass and check convergence.

        Parameters
        ----------
        profile_override : FocusProfile or None
            If provided, use this profile instead of the current one.
            Useful for dashboard-driven manual focus.

        Returns
        -------
        The scan result for this pass.
        """
        if self._current_image is None:
            raise RuntimeError("No image loaded — call begin_image() first.")

        if self._state == ScanState.CONVERGED:
            # Already converged — return latest
            return self.latest_scan  # type: ignore[return-value]

        profile = profile_override or self._current_profile
        self._pass_number += 1
        self._state = ScanState.SCANNING

        result = self._extractor.extract(
            self._current_image,
            focus_profile=profile,
            scan_pass=self._pass_number,
        )
        self._scan_history.append(result)

        # Check convergence
        interest_hash = self._compute_interest_hash(result)
        delta = abs(interest_hash - self._prev_interest_hash)
        self._prev_interest_hash = interest_hash

        if self._pass_number >= self.max_passes:
            self._state = ScanState.CONVERGED
            logger.info(
                f"Scan converged (max passes {self.max_passes} reached)"
            )
        elif self._pass_number > 1 and delta < self.convergence_threshold:
            self._state = ScanState.CONVERGED
            logger.info(
                f"Scan converged at pass {self._pass_number} "
                f"(interest delta={delta:.4f} < {self.convergence_threshold})"
            )
        else:
            if self.auto_focus:
                # Auto-select next focus
                self._current_profile = result.recommend_focus(profile)
                self._state = ScanState.SCANNING
            else:
                self._state = ScanState.AWAITING_FOCUS

        return result

    def set_focus_profile(self, profile: FocusProfile) -> None:
        """Externally set the focus profile (from dashboard)."""
        self._current_profile = profile
        if self._state == ScanState.AWAITING_FOCUS:
            self._state = ScanState.SCANNING

    def force_converge(self) -> None:
        """Force scanner to convergence (skip remaining passes)."""
        self._state = ScanState.CONVERGED

    def reset(self) -> None:
        """Reset scanner state."""
        self._current_image = None
        self._current_profile = None
        self._scan_history.clear()
        self._pass_number = 0
        self._prev_interest_hash = 0.0
        self._state = ScanState.IDLE

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_interest_hash(result: HexScanResult) -> float:
        """Hash interest scores into a single scalar for convergence check.

        Uses L2 norm of the top-12 interest scores as a stable fingerprint.
        """
        active = [
            c for c in result.cells.values()
            if c.detail_level > 0 and c.interest_score > 0
        ]
        if not active:
            return 0.0

        active.sort(key=lambda c: c.interest_score, reverse=True)
        top_scores = [c.interest_score for c in active[:12]]
        return float(np.linalg.norm(top_scores))

    # ----------------------------------------------------------------
    # Status for API / UI
    # ----------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "current_pass": self._pass_number,
            "max_passes": self.max_passes,
            "auto_focus": self.auto_focus,
            "converged": self.is_converged(),
            "num_columns": self.num_columns,
            "history_length": len(self._scan_history),
            "profile": self._current_profile.to_dict() if self._current_profile else None,
        }
