"""Hexagonal grid coordinate system and spatial utilities.

Implements axial coordinates (q, r) with cube coordinates (q, r, s)
computed on demand via ``s = -q - r``.  Uses pointy-top orientation.

Reference: https://www.redblobgames.com/grids/hexagons/

Key formulas (pointy-top):
    hex → pixel:  x = size * (√3*q + √3/2*r)
                  y = size * (3/2 * r)
    pixel → hex:  q = (√3/3 * x  -  1/3 * y) / size
                  r = (2/3 * y) / size
    distance:     max(|Δq|, |Δr|, |Δs|) where s = -q - r
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

# sqrt(3) constant
_SQRT3 = math.sqrt(3)

# The 6 axial direction vectors (pointy-top, starting E going CCW)
AXIAL_DIRECTIONS: list[tuple[int, int]] = [
    (+1, 0),   # 0: East
    (+1, -1),  # 1: NE
    (0, -1),   # 2: NW
    (-1, 0),   # 3: West
    (-1, +1),  # 4: SW
    (0, +1),   # 5: SE
]


# =====================================================================
# HexCoord — immutable axial coordinate
# =====================================================================

@dataclass(frozen=True, slots=True)
class HexCoord:
    """Axial hex coordinate (q, r).  ``s`` is computed as ``-q - r``."""

    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r

    # -- vector operations --

    def __add__(self, other: "HexCoord") -> "HexCoord":
        return HexCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other: "HexCoord") -> "HexCoord":
        return HexCoord(self.q - other.q, self.r - other.r)

    def scale(self, k: int) -> "HexCoord":
        return HexCoord(self.q * k, self.r * k)

    def __neg__(self) -> "HexCoord":
        return HexCoord(-self.q, -self.r)

    # -- distance --

    def distance_to(self, other: "HexCoord") -> int:
        """Hex grid distance (number of steps)."""
        dq = abs(self.q - other.q)
        dr = abs(self.r - other.r)
        ds = abs(self.s - other.s)
        return max(dq, dr, ds)

    # -- neighbours --

    def neighbor(self, direction: int) -> "HexCoord":
        """Return the neighbour in direction 0‑5."""
        dq, dr = AXIAL_DIRECTIONS[direction % 6]
        return HexCoord(self.q + dq, self.r + dr)

    def neighbors(self) -> list["HexCoord"]:
        """All 6 neighbours."""
        return [self.neighbor(d) for d in range(6)]

    # -- conversion helpers --

    def to_tuple(self) -> tuple[int, int]:
        return (self.q, self.r)

    @classmethod
    def from_tuple(cls, t: tuple[int, int]) -> "HexCoord":
        return cls(t[0], t[1])

    def __hash__(self) -> int:
        return hash((self.q, self.r))


# =====================================================================
# Pixel ↔ Hex conversion (pointy-top)
# =====================================================================

def hex_to_pixel(
    h: HexCoord,
    size: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """Convert axial hex coordinate to pixel (x, y)."""
    x = size * (_SQRT3 * h.q + _SQRT3 / 2 * h.r) + origin[0]
    y = size * (1.5 * h.r) + origin[1]
    return (x, y)


def pixel_to_hex(
    px: float,
    py: float,
    size: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> HexCoord:
    """Convert pixel (x, y) to the nearest axial hex coordinate."""
    x = (px - origin[0]) / size
    y = (py - origin[1]) / size
    fq = _SQRT3 / 3 * x - 1.0 / 3.0 * y
    fr = 2.0 / 3.0 * y
    return axial_round(fq, fr)


def axial_round(fq: float, fr: float) -> HexCoord:
    """Round fractional axial coordinates to the nearest hex (cube rounding)."""
    fs = -fq - fr
    q = round(fq)
    r = round(fr)
    s = round(fs)

    q_diff = abs(q - fq)
    r_diff = abs(r - fr)
    s_diff = abs(s - fs)

    if q_diff > r_diff and q_diff > s_diff:
        q = -r - s
    elif r_diff > s_diff:
        r = -q - s
    # else: s = -q - r (implicit, we only store q and r)

    return HexCoord(q, r)


# =====================================================================
# Polygon vertices (for rasterization / rendering)
# =====================================================================

def hex_polygon_vertices(
    h: HexCoord,
    size: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Return the 6 vertices of a pointy-top hex as (6, 2) array."""
    cx, cy = hex_to_pixel(h, size, origin)
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330], dtype=np.float64))
    xs = cx + size * np.cos(angles)
    ys = cy + size * np.sin(angles)
    return np.stack([xs, ys], axis=1)


# =====================================================================
# Ring and spiral traversal
# =====================================================================

def hex_ring(center: HexCoord, radius: int) -> list[HexCoord]:
    """All hexes exactly ``radius`` steps from ``center``.

    For radius == 0 returns [center].
    """
    if radius == 0:
        return [center]

    results: list[HexCoord] = []
    # Start at direction-4 corner scaled by radius
    dq, dr = AXIAL_DIRECTIONS[4]
    h = center + HexCoord(dq * radius, dr * radius)

    for direction in range(6):
        for _ in range(radius):
            results.append(h)
            h = h.neighbor(direction)

    return results


def hex_spiral(center: HexCoord, radius: int) -> list[HexCoord]:
    """All hexes within ``radius`` steps (filled disk), spiral order."""
    results = [center]
    for k in range(1, radius + 1):
        results.extend(hex_ring(center, k))
    return results


def hex_disk(center: HexCoord, radius: int) -> list[HexCoord]:
    """All hexes within ``radius`` steps (filled disk), row order.

    Uses the efficient double-loop from redblobgames.
    """
    results: list[HexCoord] = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            results.append(center + HexCoord(q, r))
    return results


# =====================================================================
# Grid builder — overlay hex grid on an image
# =====================================================================

def hex_size_from_image(image_width: int, num_columns: int = 20) -> float:
    """Compute hex radius so approximately ``num_columns`` hexes span the width.

    For pointy-top, horizontal spacing = √3 × size.
    """
    return image_width / (_SQRT3 * num_columns)


def build_hex_grid(
    width: int,
    height: int,
    hex_size: float,
    origin: tuple[float, float] | None = None,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Generate all hex center positions covering a ``width × height`` image.

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    hex_size : float
        Outer radius of each hexagon.
    origin : tuple or None
        Pixel position of hex (0, 0).  Default: offset so the grid
        starts near the top-left corner with padding.

    Returns
    -------
    dict mapping ``(q, r)`` to ``(pixel_x, pixel_y)``
    """
    if origin is None:
        # Place origin so hex (0,0) center is roughly at top-left with
        # minimal cropping.  We want the grid to cover the image.
        origin = (hex_size * _SQRT3 / 2, hex_size)

    dx = _SQRT3 * hex_size      # horizontal spacing
    dy = 1.5 * hex_size          # vertical spacing

    grid: dict[tuple[int, int], tuple[float, float]] = {}

    # Determine row range
    max_r = int(math.ceil(height / dy)) + 1
    min_r = -1  # slight padding

    for r in range(min_r, max_r + 1):
        # Pixel y for this row
        py = origin[1] + dy * r
        if py < -hex_size or py > height + hex_size:
            continue

        # Determine column range for this row
        # For pointy-top, q range shifts with r
        min_q = int(math.floor((-origin[0] - _SQRT3 / 2 * hex_size * r) / dx)) - 1
        max_q = int(math.ceil((width - origin[0] - _SQRT3 / 2 * hex_size * r) / dx)) + 1

        for q in range(min_q, max_q + 1):
            px = origin[0] + dx * q + (_SQRT3 / 2 * hex_size) * r
            if px < -hex_size or px > width + hex_size:
                continue
            grid[(q, r)] = (px, py)

    return grid


# =====================================================================
# Rasterization — create boolean mask for a hex within a bounding box
# =====================================================================

def rasterize_hex(
    h: HexCoord,
    size: float,
    img_width: int,
    img_height: int,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, int, int, int, int] | None:
    """Rasterize a hex into a boolean mask clipped to the image.

    Returns
    -------
    (mask, min_x, min_y, max_x, max_y) — mask covers the clipped bbox.
    ``None`` if the hex doesn't overlap the image.
    """
    poly = hex_polygon_vertices(h, size, origin)

    min_x = int(np.floor(poly[:, 0].min()))
    min_y = int(np.floor(poly[:, 1].min()))
    max_x = int(np.ceil(poly[:, 0].max()))
    max_y = int(np.ceil(poly[:, 1].max()))

    # Quick reject
    if max_x < 0 or max_y < 0 or min_x >= img_width or min_y >= img_height:
        return None

    # Clip to image
    min_x_c = max(0, min_x)
    min_y_c = max(0, min_y)
    max_x_c = min(img_width - 1, max_x)
    max_y_c = min(img_height - 1, max_y)

    bbox_w = max_x_c - min_x_c + 1
    bbox_h = max_y_c - min_y_c + 1
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    # Shift polygon to bbox-local coords
    poly_local = poly.copy()
    poly_local[:, 0] -= min_x_c
    poly_local[:, 1] -= min_y_c

    mask = _point_in_hex_mask(bbox_h, bbox_w, poly_local)
    if mask.sum() < 3:
        return None

    return (mask, min_x_c, min_y_c, max_x_c, max_y_c)


def _point_in_hex_mask(
    height: int, width: int, poly: np.ndarray
) -> np.ndarray:
    """Vectorized ray-casting test for a convex hex polygon."""
    x = np.arange(width, dtype=np.float32)[None, :].repeat(height, axis=0)
    y = np.arange(height, dtype=np.float32)[:, None].repeat(width, axis=1)

    inside = np.zeros((height, width), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= intersect
        j = i
    return inside


# =====================================================================
# Line drawing (hex Bresenham via cube lerp + round)
# =====================================================================

def hex_linedraw(a: HexCoord, b: HexCoord) -> list[HexCoord]:
    """Draw a line of hexes from ``a`` to ``b`` (inclusive)."""
    n = a.distance_to(b)
    if n == 0:
        return [a]

    results: list[HexCoord] = []
    # Nudge to avoid edge ambiguity
    a_q = a.q + 1e-6
    a_r = a.r + 2e-6
    b_q = float(b.q)
    b_r = float(b.r)

    for i in range(n + 1):
        t = i / n
        fq = a_q + (b_q - a_q) * t
        fr = a_r + (b_r - a_r) * t
        results.append(axial_round(fq, fr))

    return results
