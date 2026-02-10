"""Per-hex data container for the hex grid scanner.

Each ``HexCell`` stores the extracted features for one hexagonal tile
of an image.  Features are computed at varying detail levels:

    Level 0 — skip (no data extracted)
    Level 1 — coarse: avg RGB, brightness only
    Level 2 — standard: + HSV, edge energy
    Level 3 — fine: + color histogram, edge direction histogram, dominant colors

Cells are the atomic unit sent to the backend for location fingerprinting.
They serialize cleanly to dicts for API transport and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from episodic_agent.modules.panorama.hex_grid import HexCoord


@dataclass
class HexCell:
    """Feature container for a single hex tile."""

    # -- identity --
    coord: HexCoord
    center_px: tuple[float, float]

    # -- detail level --
    detail_level: int = 2  # 0=skip, 1=coarse, 2=standard, 3=fine

    # -- level 1 features (coarse) --
    avg_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    brightness: float = 0.0

    # -- level 2 features (standard) --
    avg_hsv: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    edge_energy: float = 0.0

    # -- level 3 features (fine) --
    color_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(24, dtype=np.float32)
    )  # 8 bins × 3 channels
    edge_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )  # 8 direction bins
    dominant_colors: list[tuple[int, int, int]] = field(default_factory=list)

    # -- attention / interest --
    weight: float = 1.0           # focus weight [0, 1]
    interest_score: float = 0.0   # edge × (1 − weight) heuristic

    # -- pixel count (for weighted averaging) --
    pixel_count: int = 0

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for API / persistence."""
        d: dict[str, Any] = {
            "q": self.coord.q,
            "r": self.coord.r,
            "center_x": float(self.center_px[0]),
            "center_y": float(self.center_px[1]),
            "detail_level": self.detail_level,
            "avg_rgb": self.avg_rgb.tolist(),
            "brightness": round(self.brightness, 4),
            "edge_energy": round(self.edge_energy, 4) if self.detail_level >= 2 else 0.0,
            "weight": round(self.weight, 4),
            "interest_score": round(self.interest_score, 4),
            "pixel_count": self.pixel_count,
            "dominant_colors": [list(c) for c in self.dominant_colors] if self.detail_level >= 3 else [],
        }
        if self.detail_level >= 2:
            d["avg_hsv"] = self.avg_hsv.tolist()
        if self.detail_level >= 3:
            d["color_histogram"] = self.color_histogram.tolist()
            d["edge_histogram"] = self.edge_histogram.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HexCell":
        """Reconstruct from a serialized dict."""
        coord = HexCoord(d["q"], d["r"])
        cell = cls(
            coord=coord,
            center_px=tuple(d.get("center_px", (0.0, 0.0))),
            detail_level=d.get("detail_level", 2),
            avg_rgb=np.array(d.get("avg_rgb", [0, 0, 0]), dtype=np.float32),
            brightness=d.get("brightness", 0.0),
            weight=d.get("weight", 1.0),
            interest_score=d.get("interest_score", 0.0),
            pixel_count=d.get("pixel_count", 0),
        )
        if cell.detail_level >= 2:
            cell.avg_hsv = np.array(d.get("avg_hsv", [0, 0, 0]), dtype=np.float32)
            cell.edge_energy = d.get("edge_energy", 0.0)
        if cell.detail_level >= 3:
            cell.color_histogram = np.array(
                d.get("color_histogram", [0] * 24), dtype=np.float32
            )
            cell.edge_histogram = np.array(
                d.get("edge_histogram", [0] * 8), dtype=np.float32
            )
            cell.dominant_colors = [tuple(c) for c in d.get("dominant_colors", [])]
        return cell

    # ----------------------------------------------------------------
    # Compact label for debugging
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HexCell(q={self.coord.q}, r={self.coord.r}, "
            f"L{self.detail_level}, "
            f"bright={self.brightness:.2f}, "
            f"edge={self.edge_energy:.2f}, "
            f"w={self.weight:.2f})"
        )


# =====================================================================
# FocusProfile — controls what detail level each hex gets
# =====================================================================

@dataclass
class FocusProfile:
    """Describes where the scanner should concentrate detail.

    The scanner assigns detail levels based on hex distance from the
    focus center:

        distance ≤ fovea_radius   → level 3 (fine)
        distance ≤ mid_radius     → level 2 (standard)
        distance ≤ outer_radius   → level 1 (coarse)
        distance > outer_radius   → level 0 (skip)

    ``regions_of_interest`` can selectively promote distant hexes to
    level 3 regardless of distance.

    ``detail_overrides`` allows per-hex manual control from the
    dashboard.
    """

    center: HexCoord = field(default_factory=lambda: HexCoord(0, 0))
    fovea_radius: int = 3
    mid_radius: int = 7
    outer_radius: int = 15
    alpha: float = 0.35         # radial decay factor for weight
    regions_of_interest: list[HexCoord] = field(default_factory=list)
    detail_overrides: dict[tuple[int, int], int] = field(default_factory=dict)

    # Anisotropic stretch (>1 widens focus along that axis)
    stretch_q: float = 1.0
    stretch_r: float = 1.0

    def detail_level_for(self, h: HexCoord) -> int:
        """Compute the detail level for a hex based on distance."""
        # Check manual override first
        key = h.to_tuple()
        if key in self.detail_overrides:
            return self.detail_overrides[key]

        # Check regions of interest
        if h in self.regions_of_interest:
            return 3

        d = self._weighted_distance(h)

        if d <= self.fovea_radius:
            return 3
        elif d <= self.mid_radius:
            return 2
        elif d <= self.outer_radius:
            return 1
        else:
            return 0

    def weight_for(self, h: HexCoord) -> float:
        """Compute attention weight [0, 1] using radial decay."""
        d = self._weighted_distance(h)

        if d <= self.fovea_radius:
            return 1.0
        elif d <= self.mid_radius:
            t = (d - self.fovea_radius) / max(1e-6, self.mid_radius - self.fovea_radius)
            return 1.0 - 0.5 * t
        elif d <= self.outer_radius:
            return max(0.05, 0.5 * math.exp(-self.alpha * (d - self.mid_radius)))
        else:
            return 0.05

    def _weighted_distance(self, h: HexCoord) -> float:
        """Euclidean distance in hex-space with anisotropic stretch."""
        dq = (h.q - self.center.q) / max(1e-6, self.stretch_q)
        dr = (h.r - self.center.r) / max(1e-6, self.stretch_r)
        # Euclidean-ish hex distance: sqrt(dq² + dr² + dq·dr)
        return math.sqrt(dq * dq + dr * dr + dq * dr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": {"q": self.center.q, "r": self.center.r},
            "fovea_radius": self.fovea_radius,
            "mid_radius": self.mid_radius,
            "outer_radius": self.outer_radius,
            "alpha": self.alpha,
            "stretch_q": self.stretch_q,
            "stretch_r": self.stretch_r,
            "regions_of_interest": [
                {"q": h.q, "r": h.r} for h in self.regions_of_interest
            ],
            "detail_overrides": {
                f"{q},{r}": level
                for (q, r), level in self.detail_overrides.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FocusProfile":
        center_d = d.get("center", {"q": 0, "r": 0})
        rois = [
            HexCoord(h["q"], h["r"]) for h in d.get("regions_of_interest", [])
        ]
        overrides = {}
        for key_str, level in d.get("detail_overrides", {}).items():
            parts = key_str.split(",")
            if len(parts) == 2:
                overrides[(int(parts[0]), int(parts[1]))] = level
        return cls(
            center=HexCoord(center_d.get("q", 0), center_d.get("r", 0)),
            fovea_radius=d.get("fovea_radius", 3),
            mid_radius=d.get("mid_radius", 7),
            outer_radius=d.get("outer_radius", 15),
            alpha=d.get("alpha", 0.35),
            stretch_q=d.get("stretch_q", 1.0),
            stretch_r=d.get("stretch_r", 1.0),
            regions_of_interest=rois,
            detail_overrides=overrides,
        )


# Need math import for FocusProfile.weight_for
import math
