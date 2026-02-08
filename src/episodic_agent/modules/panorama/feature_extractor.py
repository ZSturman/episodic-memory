"""Feature extraction from panorama viewport crops.

ARCHITECTURAL INVARIANT: Raw images discarded after feature extraction.
Only derived features (color histograms, edge histograms, brightness,
texture) are persisted. No ML dependencies — uses OpenCV + numpy only.

Reuses the ExtractedVisualFeatures schema and 4×4 grid model from the
visual channel, ensuring compatibility with LocationResolverReal's
cosine-distance fingerprinting.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import cv2
import numpy as np

from episodic_agent.schemas.visual import (
    COLOR_HISTOGRAM_BINS,
    EDGE_DIRECTION_BINS,
    ExtractedVisualFeatures,
    GRID_COLS,
    GRID_ROWS,
)

logger = logging.getLogger(__name__)


class PanoramaFeatureExtractor:
    """Extract visual features from panorama viewport crops.

    Produces the same ExtractedVisualFeatures dataclass used by the
    Unity visual channel, so downstream modules (LocationResolverReal,
    entity matching) work identically.

    Features extracted per 4×4 grid cell:
      - Color histogram (8 bins × 3 channels, normalized)
      - Edge direction histogram (8 bins, magnitude-weighted)
      - Mean brightness (0–1)
      - Edge density (0–1)
    """

    def __init__(
        self,
        color_bins: int = COLOR_HISTOGRAM_BINS,
        edge_bins: int = EDGE_DIRECTION_BINS,
    ) -> None:
        self._color_bins = color_bins
        self._edge_bins = edge_bins
        self._frames_processed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_array(
        self,
        img_bgr: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> ExtractedVisualFeatures:
        """Extract features from a BGR numpy array (OpenCV convention).

        Args:
            img_bgr: Image as H×W×3 BGR uint8 array.
            frame_id: Monotonic frame identifier.
            timestamp: Unix timestamp.

        Returns:
            ExtractedVisualFeatures with per-cell and global features.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self._extract(img_rgb, frame_id, timestamp)

    def extract_from_rgb(
        self,
        img_rgb: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> ExtractedVisualFeatures:
        """Extract features from an RGB numpy array.

        Args:
            img_rgb: Image as H×W×3 RGB uint8 array.
            frame_id: Monotonic frame identifier.
            timestamp: Unix timestamp.

        Returns:
            ExtractedVisualFeatures with per-cell and global features.
        """
        return self._extract(img_rgb, frame_id, timestamp)

    def accumulate_panorama_embedding(
        self,
        heading_embeddings: list[list[float]],
    ) -> list[float]:
        """Combine per-heading embeddings into a single panoramic embedding.

        Uses element-wise mean so the result stays in the same 128-dim
        space that LocationResolverReal expects.

        Args:
            heading_embeddings: List of 128-dim embedding vectors.

        Returns:
            128-dim averaged panoramic embedding.
        """
        if not heading_embeddings:
            return [0.0] * 128

        arr = np.array(heading_embeddings, dtype=np.float64)
        mean = arr.mean(axis=0)
        return mean.tolist()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract(
        self,
        img_rgb: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> ExtractedVisualFeatures:
        """Core extraction from an RGB array."""
        features = ExtractedVisualFeatures(
            frame_id=frame_id,
            timestamp=timestamp,
        )

        height, width = img_rgb.shape[:2]
        cell_h = max(1, height // GRID_ROWS)
        cell_w = max(1, width // GRID_COLS)

        # Per-cell features
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                y1, y2 = row * cell_h, min((row + 1) * cell_h, height)
                x1, x2 = col * cell_w, min((col + 1) * cell_w, width)
                cell = img_rgb[y1:y2, x1:x2]

                features.cell_color_histograms.append(
                    self._color_histogram(cell)
                )
                features.cell_edge_histograms.append(
                    self._edge_histogram(cell)
                )
                features.cell_brightness.append(
                    float(np.mean(cell) / 255.0)
                )
                features.cell_motion.append(0.0)

        # Global features
        features.global_color_histogram = self._color_histogram(img_rgb)
        features.global_edge_histogram = self._edge_histogram(img_rgb)
        features.dominant_colors = self._dominant_colors(img_rgb)
        features.scene_signature = self._scene_signature(features)

        self._frames_processed += 1
        return features

    # -- histogram helpers ------------------------------------------------

    def _color_histogram(self, img_rgb: np.ndarray) -> list[float]:
        """Compute normalized RGB histogram."""
        hist: list[float] = []
        for ch in range(3):
            h = cv2.calcHist(
                [img_rgb], [ch], None,
                [self._color_bins], [0, 256],
            ).flatten().astype(np.float64)
            s = h.sum()
            if s > 0:
                h /= s
            hist.extend(h.tolist())
        return hist

    def _edge_histogram(self, img_rgb: np.ndarray) -> list[float]:
        """Compute edge direction histogram using Sobel gradients."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        direction = np.arctan2(gy, gx)  # -π to π
        direction_norm = (direction + np.pi) / (2 * np.pi)  # 0 to 1
        bins = np.clip(
            (direction_norm * self._edge_bins).astype(int),
            0, self._edge_bins - 1,
        )

        hist = np.zeros(self._edge_bins, dtype=np.float64)
        for i in range(self._edge_bins):
            hist[i] = magnitude[bins == i].sum()

        s = hist.sum()
        if s > 0:
            hist /= s
        return hist.tolist()

    def _dominant_colors(
        self,
        img_rgb: np.ndarray,
        k: int = 3,
    ) -> list[tuple[int, int, int]]:
        """Return top-k dominant colours by quantised frequency."""
        quantized = (img_rgb // 32) * 32 + 16
        pixels = quantized.reshape(-1, 3)
        unique, counts = np.unique(pixels, axis=0, return_counts=True)
        top = np.argsort(counts)[-k:][::-1]
        return [tuple(int(v) for v in unique[i]) for i in top]

    def _scene_signature(self, features: ExtractedVisualFeatures) -> str:
        """Quick-compare hash of key features."""
        parts: list[str] = []
        for b in features.cell_brightness[:16]:
            parts.append(str(int(b * 10)))
        for c in features.dominant_colors[:3]:
            parts.append(f"{c[0] // 32}{c[1] // 32}{c[2] // 32}")
        return hashlib.md5("_".join(parts).encode()).hexdigest()[:12]

    # -- stats ------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        return {
            "frames_processed": self._frames_processed,
            "color_bins": self._color_bins,
            "edge_bins": self._edge_bins,
        }
