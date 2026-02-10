"""Hex-grid feature extraction from images.

Replaces the old 4×4 rectangular grid extractor with a hexagonal grid
that supports variable detail levels per cell.  Each hex cell is
rasterized, pixels within the hex are aggregated into features, and
the results are packed into ``HexCell`` objects.

The extractor also builds a 128-dim scene embedding from aggregated
hex features, maintaining backward compatibility with the existing
``LocationResolverReal`` cosine-distance matching.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import cv2
import numpy as np

from episodic_agent.modules.panorama.hex_grid import (
    HexCoord,
    build_hex_grid,
    hex_polygon_vertices,
    hex_size_from_image,
    hex_to_pixel,
    rasterize_hex,
)
from episodic_agent.modules.panorama.hex_cell import FocusProfile, HexCell

logger = logging.getLogger(__name__)

# Embedding dimension — must match LocationResolverReal expectations
EMBEDDING_DIM = 128


class HexScanResult:
    """Container for a complete hex-grid scan of one image."""

    __slots__ = (
        "cells", "hex_size", "grid_origin", "num_columns",
        "image_width", "image_height", "scan_pass",
    )

    def __init__(
        self,
        cells: dict[tuple[int, int], HexCell],
        hex_size: float,
        grid_origin: tuple[float, float],
        num_columns: int,
        image_width: int,
        image_height: int,
        scan_pass: int = 1,
    ) -> None:
        self.cells = cells
        self.hex_size = hex_size
        self.grid_origin = grid_origin
        self.num_columns = num_columns
        self.image_width = image_width
        self.image_height = image_height
        self.scan_pass = scan_pass

    # ----------------------------------------------------------------
    # Embedding (128-dim, backward-compat with LocationResolverReal)
    # ----------------------------------------------------------------

    def to_embedding(self) -> list[float]:
        """Build a 128-dim embedding from aggregated hex features.

        Layout (matches old ExtractedVisualFeatures.to_embedding()):
            [0:64]   global color histogram (8 bins × 3 channels × ~2.67 repeats,
                     expanded from weighted average of per-hex histograms)
            [64:72]  global edge histogram (8 direction bins)
            [72:88]  brightness distribution (16 bins from hex brightnesses)
            [88:104] edge energy distribution (16 bins)
            [104:128] spatial feature summary (24 values)
        """
        active_cells = [
            c for c in self.cells.values() if c.detail_level > 0 and c.pixel_count > 0
        ]
        if not active_cells:
            return [0.0] * EMBEDDING_DIM

        embedding = [0.0] * EMBEDDING_DIM

        # --- [0:64] Color histogram ---
        # Aggregate color histograms from fine cells, or approximate from avg_rgb
        color_hist = np.zeros(24, dtype=np.float64)
        fine_cells = [c for c in active_cells if c.detail_level >= 3]
        if fine_cells:
            total_weight = 0.0
            for c in fine_cells:
                w = c.weight * c.pixel_count
                color_hist += c.color_histogram.astype(np.float64) * w
                total_weight += w
            if total_weight > 0:
                color_hist /= total_weight
        else:
            # Approximate from avg_rgb of all cells
            for c in active_cells:
                r_bin = min(7, int(c.avg_rgb[0] * 8))
                g_bin = min(7, int(c.avg_rgb[1] * 8))
                b_bin = min(7, int(c.avg_rgb[2] * 8))
                w = c.weight * c.pixel_count
                color_hist[r_bin] += w
                color_hist[8 + g_bin] += w
                color_hist[16 + b_bin] += w
            total = color_hist.sum()
            if total > 0:
                color_hist /= total

        # Expand 24-bin to 64 values by repeating and padding
        expanded = np.zeros(64, dtype=np.float64)
        for i in range(24):
            idx = int(i * 64 / 24)
            expanded[idx] += color_hist[i]
        # Normalize
        total = expanded.sum()
        if total > 0:
            expanded /= total
        embedding[0:64] = expanded.tolist()

        # --- [64:72] Edge histogram ---
        edge_hist = np.zeros(8, dtype=np.float64)
        fine_with_edges = [c for c in active_cells if c.detail_level >= 3]
        if fine_with_edges:
            total_weight = 0.0
            for c in fine_with_edges:
                w = c.weight * c.pixel_count
                edge_hist += c.edge_histogram.astype(np.float64) * w
                total_weight += w
            if total_weight > 0:
                edge_hist /= total_weight
        else:
            # Approximate: spread edge_energy across bins uniformly
            for c in active_cells:
                if c.detail_level >= 2:
                    energy_per_bin = c.edge_energy / 8.0
                    edge_hist += energy_per_bin * c.weight
            total = edge_hist.sum()
            if total > 0:
                edge_hist /= total
        embedding[64:72] = edge_hist.tolist()

        # --- [72:88] Brightness distribution ---
        brightness_hist = np.zeros(16, dtype=np.float64)
        for c in active_cells:
            b_bin = min(15, int(c.brightness * 16))
            brightness_hist[b_bin] += c.weight * c.pixel_count
        total = brightness_hist.sum()
        if total > 0:
            brightness_hist /= total
        embedding[72:88] = brightness_hist.tolist()

        # --- [88:104] Edge energy distribution ---
        edge_energy_hist = np.zeros(16, dtype=np.float64)
        energies = [c.edge_energy for c in active_cells if c.detail_level >= 2]
        if energies:
            max_e = max(energies) if max(energies) > 0 else 1.0
            for c in active_cells:
                if c.detail_level >= 2:
                    e_bin = min(15, int((c.edge_energy / max_e) * 16))
                    edge_energy_hist[e_bin] += c.weight * c.pixel_count
            total = edge_energy_hist.sum()
            if total > 0:
                edge_energy_hist /= total
        embedding[88:104] = edge_energy_hist.tolist()

        # --- [104:128] Spatial summary ---
        # Encode spatial distribution of brightness and edges across the image
        # Divide image into a 4x6 grid of spatial bins
        spatial = np.zeros(24, dtype=np.float64)
        for c in active_cells:
            px, py = c.center_px
            col = min(5, int(px / max(1, self.image_width) * 6))
            row = min(3, int(py / max(1, self.image_height) * 4))
            idx = row * 6 + col
            spatial[idx] += c.brightness * c.weight

        total = spatial.sum()
        if total > 0:
            spatial /= total
        embedding[104:128] = spatial.tolist()

        # Normalize full embedding to unit length
        norm = math.sqrt(sum(v * v for v in embedding))
        if norm > 1e-8:
            embedding = [v / norm for v in embedding]

        return embedding

    # ----------------------------------------------------------------
    # Reconstruction data (for dashboard rebuild)
    # ----------------------------------------------------------------

    def to_reconstruction_data(self) -> dict[str, Any]:
        """Minimal per-hex data for visual reconstruction."""
        cells_data: list[dict[str, Any]] = []
        for cell in self.cells.values():
            if cell.detail_level > 0 and cell.pixel_count > 0:
                cells_data.append({
                    "q": cell.coord.q,
                    "r": cell.coord.r,
                    "avg_rgb": cell.avg_rgb.tolist(),
                    "edge_energy": round(cell.edge_energy, 4),
                    "brightness": round(cell.brightness, 4),
                    "detail_level": cell.detail_level,
                })
        return {
            "hex_size": self.hex_size,
            "grid_origin": list(self.grid_origin),
            "num_columns": self.num_columns,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "cells": cells_data,
        }

    # ----------------------------------------------------------------
    # Interest analysis
    # ----------------------------------------------------------------

    def interest_map(self) -> dict[tuple[int, int], float]:
        """Per-hex interest scores."""
        return {
            key: cell.interest_score
            for key, cell in self.cells.items()
            if cell.detail_level > 0
        }

    def recommend_focus(self, current: FocusProfile | None = None) -> FocusProfile:
        """Suggest a new focus profile based on interest scores.

        Computes the weighted centroid of the top-K interest cells
        and returns a FocusProfile centered there.
        """
        active = [
            c for c in self.cells.values()
            if c.detail_level > 0 and c.interest_score > 0
        ]
        if not active:
            # Default to image center
            center_q = 0
            center_r = 0
            for c in self.cells.values():
                center_q += c.coord.q
                center_r += c.coord.r
            n = max(1, len(self.cells))
            return FocusProfile(
                center=HexCoord(round(center_q / n), round(center_r / n)),
            )

        # Top-K by interest
        k = min(12, len(active))
        active.sort(key=lambda c: c.interest_score, reverse=True)
        top_k = active[:k]

        scores = np.array([c.interest_score for c in top_k], dtype=np.float64)
        total = scores.sum()
        if total > 1e-8:
            weights = scores / total
        else:
            weights = np.ones(k) / k

        avg_q = sum(c.coord.q * w for c, w in zip(top_k, weights))
        avg_r = sum(c.coord.r * w for c, w in zip(top_k, weights))

        # Inherit radii from current profile or use defaults
        base = current or FocusProfile()
        return FocusProfile(
            center=HexCoord(round(avg_q), round(avg_r)),
            fovea_radius=base.fovea_radius,
            mid_radius=base.mid_radius,
            outer_radius=base.outer_radius,
            alpha=base.alpha,
            stretch_q=base.stretch_q,
            stretch_r=base.stretch_r,
        )

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_api_dict(self, converged: bool = False, focus_center: tuple[int, int] | None = None) -> dict[str, Any]:
        """Full scan result for the API.

        Parameters
        ----------
        converged : bool
            Whether the multi-pass scan has converged.
        focus_center : tuple or None
            The (q, r) of the focus center, if any.
        """
        fc_q, fc_r = focus_center if focus_center else (0, 0)
        return {
            "hex_size": self.hex_size,
            "grid_origin": list(self.grid_origin),
            "num_columns": self.num_columns,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "scan_pass": self.scan_pass,
            "total_cells": len(self.cells),
            "cell_count": len(self.cells),
            "active_cells": sum(
                1 for c in self.cells.values() if c.detail_level > 0
            ),
            "converged": converged,
            "focus_center_q": fc_q,
            "focus_center_r": fc_r,
            "cells": [c.to_dict() for c in self.cells.values() if c.detail_level > 0],
        }


# =====================================================================
# HexFeatureExtractor — the main extraction pipeline
# =====================================================================

class HexFeatureExtractor:
    """Extract visual features using a hexagonal grid overlay.

    Parameters
    ----------
    num_columns : int
        Approximate number of hex columns across the image width.
    """

    def __init__(self, num_columns: int = 20) -> None:
        self.num_columns = num_columns

    def extract(
        self,
        img_rgb: np.ndarray,
        focus_profile: FocusProfile | None = None,
        scan_pass: int = 1,
    ) -> HexScanResult:
        """Run feature extraction over the full image.

        Parameters
        ----------
        img_rgb : ndarray
            Image as (H, W, 3) uint8 RGB.
        focus_profile : FocusProfile or None
            Controls detail levels.  If None, all hexes get level 2.
        scan_pass : int
            Pass number (for multi-pass scanning).

        Returns
        -------
        HexScanResult
        """
        h, w = img_rgb.shape[:2]
        hex_size = hex_size_from_image(w, self.num_columns)
        origin = (hex_size * math.sqrt(3) / 2, hex_size)

        # Precompute image-wide data
        img_f = img_rgb.astype(np.float32) / 255.0
        gray = (
            0.2989 * img_f[..., 0]
            + 0.5870 * img_f[..., 1]
            + 0.1140 * img_f[..., 2]
        )

        # Edge maps (computed once)
        edges_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        edges_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edge_magnitude = edges_x + edges_y

        # HSV (for level ≥ 2)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0

        # Sobel for direction histogram (level 3)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_angle = np.arctan2(sobel_y, sobel_x)  # [-π, π]
        edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Build hex grid
        grid = build_hex_grid(w, h, hex_size, origin)

        # If no focus profile, build a default centered on image center
        if focus_profile is None:
            from episodic_agent.modules.panorama.hex_grid import pixel_to_hex
            center_hex = pixel_to_hex(w / 2, h / 2, hex_size, origin)
            focus_profile = FocusProfile(
                center=center_hex,
                fovea_radius=100,   # effectively unlimited: uniform level 2
                mid_radius=200,
                outer_radius=300,
            )

        cells: dict[tuple[int, int], HexCell] = {}

        for (q, r), (px, py) in grid.items():
            coord = HexCoord(q, r)
            detail = focus_profile.detail_level_for(coord)
            weight = focus_profile.weight_for(coord)

            if detail == 0:
                # Skip — create placeholder
                cells[(q, r)] = HexCell(
                    coord=coord,
                    center_px=(px, py),
                    detail_level=0,
                    weight=weight,
                )
                continue

            # Rasterize
            rast = rasterize_hex(coord, hex_size, w, h, origin)
            if rast is None:
                continue
            mask, x1, y1, x2, y2 = rast
            pixel_count = int(mask.sum())
            if pixel_count < 3:
                continue

            cell = HexCell(
                coord=coord,
                center_px=(px, py),
                detail_level=detail,
                weight=weight,
                pixel_count=pixel_count,
            )

            # -- Level 1: avg RGB + brightness --
            patch_rgb = img_f[y1:y2 + 1, x1:x2 + 1]
            cell.avg_rgb = patch_rgb[mask].mean(axis=0).astype(np.float32)
            patch_gray = gray[y1:y2 + 1, x1:x2 + 1]
            cell.brightness = float(patch_gray[mask].mean())

            if detail >= 2:
                # -- Level 2: HSV + edge energy --
                patch_hsv = img_hsv[y1:y2 + 1, x1:x2 + 1]
                cell.avg_hsv = patch_hsv[mask].mean(axis=0).astype(np.float32)
                patch_edges = edge_magnitude[y1:y2 + 1, x1:x2 + 1]
                cell.edge_energy = float(patch_edges[mask].mean())

            if detail >= 3:
                # -- Level 3: histograms --
                # Color histogram (8 bins × 3 channels)
                pixels = (patch_rgb[mask] * 255).astype(np.uint8)
                hist = np.zeros(24, dtype=np.float32)
                for ch in range(3):
                    h_ch, _ = np.histogram(pixels[:, ch], bins=8, range=(0, 256))
                    h_ch = h_ch.astype(np.float32)
                    total = h_ch.sum()
                    if total > 0:
                        h_ch /= total
                    hist[ch * 8:(ch + 1) * 8] = h_ch
                cell.color_histogram = hist

                # Edge direction histogram
                patch_angle = edge_angle[y1:y2 + 1, x1:x2 + 1]
                patch_mag = edge_mag[y1:y2 + 1, x1:x2 + 1]
                angles = patch_angle[mask]
                mags = patch_mag[mask]
                e_hist = np.zeros(8, dtype=np.float32)
                bins_idx = ((angles + np.pi) / (2 * np.pi) * 8).astype(int) % 8
                for bi in range(8):
                    e_hist[bi] = float(mags[bins_idx == bi].sum())
                total = e_hist.sum()
                if total > 0:
                    e_hist /= total
                cell.edge_histogram = e_hist

                # Dominant colors (top 3 from quantized)
                quant = (pixels // 32).astype(np.uint32)
                keys = quant[:, 0] * 64 + quant[:, 1] * 8 + quant[:, 2]
                unique, counts = np.unique(keys, return_counts=True)
                top_idx = np.argsort(-counts)[:3]
                dom = []
                for ti in top_idx:
                    k = int(unique[ti])
                    rr = ((k // 64) * 32 + 16)
                    gg = (((k % 64) // 8) * 32 + 16)
                    bb = ((k % 8) * 32 + 16)
                    dom.append((rr, gg, bb))
                cell.dominant_colors = dom

            cells[(q, r)] = cell

        # -- Compute interest scores --
        self._compute_interest(cells)

        return HexScanResult(
            cells=cells,
            hex_size=hex_size,
            grid_origin=origin,
            num_columns=self.num_columns,
            image_width=w,
            image_height=h,
            scan_pass=scan_pass,
        )

    @staticmethod
    def _compute_interest(cells: dict[tuple[int, int], HexCell]) -> None:
        """Compute interest scores: high edge energy in low-weight regions."""
        active = [c for c in cells.values() if c.detail_level >= 2 and c.pixel_count > 0]
        if not active:
            return

        energies = np.array([c.edge_energy for c in active], dtype=np.float32)
        e_min = energies.min()
        e_max = energies.max()
        denom = (e_max - e_min) if (e_max > e_min) else 1.0

        for c in active:
            norm_edge = (c.edge_energy - e_min) / denom
            c.interest_score = norm_edge * (1.0 - c.weight)

        # Normalize to [0, 1]
        scores = [c.interest_score for c in active]
        max_score = max(scores) if scores else 1.0
        if max_score > 1e-8:
            for c in active:
                c.interest_score /= max_score
