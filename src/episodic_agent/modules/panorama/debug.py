"""Terminal debug output for panorama harness.

Provides coloured, structured terminal output showing:
  - Current image + viewport position
  - Hypothesis + confidence
  - Evidence summary (brightness, edges, colours)
  - Outbound sensor message payload (truncated)

Uses ``rich`` (included via typer[all]) for formatting.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try rich for pretty output; fall back to plain text
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


class TerminalDebugger:
    """Pretty-prints each orchestrator step to the terminal."""

    def __init__(self) -> None:
        self._console = Console() if _HAS_RICH else None
        self._step = 0

    def print_step(
        self,
        result: Any,
        sensor: Any | None = None,
        perception: Any | None = None,
    ) -> None:
        """Print a single step summary.

        Parameters
        ----------
        result : StepResult
            The orchestrator step result.
        sensor : PanoramaSensorProvider, optional
            Sensor for status info.
        perception : PanoramaPerception, optional
            Perception for hypothesis info.
        """
        self._step += 1

        # Build info dict
        source_name = "?"
        heading = 0.0
        vp_idx = 0
        vp_total = 0
        transition = False

        if sensor and hasattr(sensor, "get_status"):
            st = sensor.get_status()
            source_name = st.get("current_source", "?")
            if source_name:
                # Just the filename
                source_name = source_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        # Extract extras from the result (which wraps the step data)
        extras: dict = {}
        if hasattr(result, "extras") and isinstance(result.extras, dict):
            extras = result.extras
        # Also check the step result's attributes
        frame_id = getattr(result, "frame_id", None)

        # Try to get per-step info from perception
        hypothesis: dict = {}
        if perception and hasattr(perception, "hypothesis"):
            hypothesis = perception.hypothesis

        # Determine feature summary
        feature_summary: dict = {}
        if extras:
            feature_summary = extras.get("feature_summary", {})
            heading = extras.get("heading_deg", 0.0)
            vp_idx = extras.get("viewport_index", 0)
            vp_total = extras.get("total_viewports", 0)
            transition = extras.get("transition", False)

        if transition:
            self._print_transition(source_name)
            return

        if _HAS_RICH and self._console:
            self._print_rich(
                result, source_name, heading, vp_idx, vp_total,
                hypothesis, feature_summary, frame_id,
            )
        else:
            self._print_plain(
                result, source_name, heading, vp_idx, vp_total,
                hypothesis, feature_summary, frame_id,
            )

    # ------------------------------------------------------------------
    # Rich output
    # ------------------------------------------------------------------

    def _print_rich(
        self,
        result: Any,
        source: str,
        heading: float,
        vp_idx: int,
        vp_total: int,
        hypothesis: dict,
        features: dict,
        frame_id: int | None,
    ) -> None:
        assert self._console is not None

        # Progress bar
        if vp_total > 0:
            filled = int((vp_idx / vp_total) * 20)
            bar = "█" * filled + "░" * (20 - filled)
            progress = f"[{bar}] {vp_idx}/{vp_total}"
        else:
            progress = "[░░░░░░░░░░░░░░░░░░░░] ?/?"

        loc = getattr(result, "location_label", "?")
        conf = getattr(result, "location_confidence", 0.0)
        ep_count = getattr(result, "episode_count", 0)
        boundary = getattr(result, "boundary_triggered", False)
        boundary_mark = " [bold red]BOUNDARY[/]" if boundary else ""

        hyp_label = hypothesis.get("label") or "unknown"
        hyp_conf = hypothesis.get("confidence", 0.0)

        bright = features.get("global_brightness", 0.0)
        edge_d = features.get("global_edge_density", 0.0)
        colors = features.get("dominant_colors", [])
        sig = features.get("scene_signature", "")

        self._console.print(
            f"[dim][{self._step:04d}][/dim] "
            f"[cyan]{source}[/] "
            f"heading={heading:.0f}° "
            f"{progress}"
            f"  📍 {loc}({conf:.0%}) "
            f"📚 {ep_count}"
            f"{boundary_mark}"
        )
        evidence_parts = []
        if bright:
            evidence_parts.append(f"bright={bright:.2f}")
        if edge_d:
            evidence_parts.append(f"edges={edge_d:.3f}")
        if colors:
            color_strs = [f"({c[0]},{c[1]},{c[2]})" for c in colors[:3]]
            evidence_parts.append(f"colors={','.join(color_strs)}")
        if sig:
            evidence_parts.append(f"sig={sig}")
        if evidence_parts:
            self._console.print(
                f"         [dim]evidence: {' | '.join(evidence_parts)}[/]"
            )

    def _print_transition(self, source: str) -> None:
        if _HAS_RICH and self._console:
            self._console.print(
                f"[dim][{self._step:04d}][/dim] "
                f"[yellow]── transition → {source} ──[/]"
            )
        else:
            print(f"[{self._step:04d}] ── transition → {source} ──")

    # ------------------------------------------------------------------
    # Plain fallback
    # ------------------------------------------------------------------

    def _print_plain(
        self,
        result: Any,
        source: str,
        heading: float,
        vp_idx: int,
        vp_total: int,
        hypothesis: dict,
        features: dict,
        frame_id: int | None,
    ) -> None:
        loc = getattr(result, "location_label", "?")
        conf = getattr(result, "location_confidence", 0.0)
        ep_count = getattr(result, "episode_count", 0)
        boundary = getattr(result, "boundary_triggered", False)
        bmark = " [BOUNDARY]" if boundary else ""

        print(
            f"[{self._step:04d}] {source} "
            f"heading={heading:.0f}° vp={vp_idx}/{vp_total} "
            f"loc={loc}({conf:.0%}) ep={ep_count}{bmark}"
        )
