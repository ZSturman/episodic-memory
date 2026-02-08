"""Saccade policy — controls where the agent looks next.

Inspired by the Thousand Brains Project Monty's hypothesis-driven
active-looking strategy.  Three modes:

  SCANNING     — systematic sweep across the image (default on arrival)
  INVESTIGATING — zoom into the grid cell with highest saliency
  CONFIRMING   — revisit the viewport direction where top-2 location
                 hypotheses differ most (diagnostically informative)

The policy receives the current image dimensions, the attention state,
and an optional hypothesis dict and returns the next viewport rectangle.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SaccadeMode(str, Enum):
    """Current looking-around mode."""

    SCANNING = "scanning"
    INVESTIGATING = "investigating"
    CONFIRMING = "confirming"


@dataclass
class Viewport:
    """A rectangular viewport region on the source image."""

    x: int
    y: int
    width: int
    height: int
    heading_deg: float = 0.0  # virtual heading corresponding to this crop

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "heading_deg": self.heading_deg,
        }


class SaccadePolicy:
    """Decides where to look next on the source image.

    The default *scanning* sweep divides the image into a configurable
    number of equally-spaced headings across the horizontal extent,
    with the viewport centered vertically.  Once the sweep is complete,
    `sweep_complete` is set to ``True``.

    If a saliency map is provided (16-element list from the 4×4 grid),
    the policy may switch to INVESTIGATING to focus on the most
    salient region.

    Parameters
    ----------
    viewport_width : int
        Width of each viewport crop (pixels).
    viewport_height : int
        Height of each viewport crop (pixels).
    headings : int
        Number of evenly-spaced horizontal stops per sweep.
    overlap : float
        Fraction of viewport that overlaps the previous stop (0–0.5).
    seed : int
        Random seed for any stochastic viewport jumps.
    """

    def __init__(
        self,
        viewport_width: int = 256,
        viewport_height: int = 256,
        headings: int = 8,
        overlap: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.vp_w = viewport_width
        self.vp_h = viewport_height
        self.headings = headings
        self.overlap = overlap
        self._rng = random.Random(seed)

        # Internal state
        self._mode = SaccadeMode.SCANNING
        self._heading_idx = 0
        self.sweep_complete = False
        self._viewports: list[Viewport] = []  # pre-computed for current image

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def mode(self) -> SaccadeMode:
        return self._mode

    def reset(self) -> None:
        """Reset to start of a new sweep."""
        self._heading_idx = 0
        self.sweep_complete = False
        self._mode = SaccadeMode.SCANNING
        self._viewports.clear()

    def prepare_for_image(self, img_width: int, img_height: int) -> None:
        """Pre-compute viewports for a given image size.

        Call this once when moving to a new source image.
        """
        self.reset()
        self._viewports = self._compute_scan_viewports(img_width, img_height)

    def next_viewport(
        self,
        img_width: int,
        img_height: int,
        saliency_map: list[float] | None = None,
        hypothesis: dict[str, Any] | None = None,
    ) -> Viewport | None:
        """Return the next viewport to examine.

        Returns ``None`` when the sweep is complete and no investigation
        target is available.
        """
        # Lazy init
        if not self._viewports:
            self.prepare_for_image(img_width, img_height)

        # --- handle modes -------------------------------------------------
        if self._mode == SaccadeMode.SCANNING:
            return self._next_scan()

        if self._mode == SaccadeMode.INVESTIGATING:
            return self._investigate(img_width, img_height, saliency_map)

        if self._mode == SaccadeMode.CONFIRMING:
            return self._confirm(img_width, img_height, hypothesis)

        return None

    def total_viewports(self) -> int:
        """Number of viewports in the current sweep."""
        return len(self._viewports)

    def current_index(self) -> int:
        return self._heading_idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_scan_viewports(
        self,
        img_width: int,
        img_height: int,
    ) -> list[Viewport]:
        """Evenly space ``self.headings`` viewports across the image."""
        vps: list[Viewport] = []
        vw = min(self.vp_w, img_width)
        vh = min(self.vp_h, img_height)

        usable_width = img_width - vw
        if usable_width <= 0 or self.headings <= 1:
            # Image narrower than viewport — single crop
            cx = max(0, (img_width - vw) // 2)
            cy = max(0, (img_height - vh) // 2)
            vps.append(Viewport(x=cx, y=cy, width=vw, height=vh, heading_deg=0.0))
            return vps

        step = usable_width / (self.headings - 1)
        for i in range(self.headings):
            x = int(round(i * step))
            y = max(0, (img_height - vh) // 2)
            heading = (i / (self.headings - 1)) * 360.0 if self.headings > 1 else 0.0
            vps.append(Viewport(x=x, y=y, width=vw, height=vh, heading_deg=heading))

        return vps

    def _next_scan(self) -> Viewport | None:
        if self._heading_idx >= len(self._viewports):
            self.sweep_complete = True
            return None
        vp = self._viewports[self._heading_idx]
        self._heading_idx += 1
        if self._heading_idx >= len(self._viewports):
            self.sweep_complete = True
        return vp

    def _investigate(
        self,
        img_width: int,
        img_height: int,
        saliency_map: list[float] | None,
    ) -> Viewport | None:
        """Centre viewport on the most salient grid cell."""
        if not saliency_map or len(saliency_map) < 16:
            self._mode = SaccadeMode.SCANNING
            return self._next_scan()

        max_idx = int(max(range(len(saliency_map)), key=lambda i: saliency_map[i]))
        row, col = divmod(max_idx, 4)
        cell_h = img_height // 4
        cell_w = img_width // 4
        cx = col * cell_w + cell_w // 2
        cy = row * cell_h + cell_h // 2
        vw = min(self.vp_w, img_width)
        vh = min(self.vp_h, img_height)
        x = max(0, min(cx - vw // 2, img_width - vw))
        y = max(0, min(cy - vh // 2, img_height - vh))
        heading = (cx / img_width) * 360.0
        # After one investigation step, drop back to scanning
        self._mode = SaccadeMode.SCANNING
        return Viewport(x=x, y=y, width=vw, height=vh, heading_deg=heading)

    def _confirm(
        self,
        img_width: int,
        img_height: int,
        hypothesis: dict[str, Any] | None,
    ) -> Viewport | None:
        """Revisit a diagnostic direction.

        If the hypothesis dict contains ``diagnostic_heading``, centre there.
        Otherwise fall back to a random viewport.
        """
        if hypothesis and "diagnostic_heading" in hypothesis:
            heading = float(hypothesis["diagnostic_heading"])
            frac = heading / 360.0
            vw = min(self.vp_w, img_width)
            vh = min(self.vp_h, img_height)
            x = max(0, min(int(frac * img_width) - vw // 2, img_width - vw))
            y = max(0, (img_height - vh) // 2)
            self._mode = SaccadeMode.SCANNING
            return Viewport(x=x, y=y, width=vw, height=vh, heading_deg=heading)

        # Fallback: pick a random viewport we haven't visited
        if self._viewports:
            vp = self._rng.choice(self._viewports)
            self._mode = SaccadeMode.SCANNING
            return vp
        return None
