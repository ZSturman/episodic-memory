"""Panorama perception module — converts viewport crops into Percepts.

Implements the PerceptionModule ABC.  Decodes the JPEG viewport from
``SensorFrame.raw_data``, runs feature extraction via
PanoramaFeatureExtractor, and produces a ``Percept`` with a 128-dim
``scene_embedding``.

Maintains a rolling buffer of per-heading embeddings for each source
image.  When the agent finishes sweeping one image, the accumulated
panoramic embedding (element-wise mean) is stored in
``percept.extras["panoramic_embedding"]`` on the final heading's
Percept so that the location resolver can fingerprint on the full view.

No hidden ground-truth: the embedding is derived entirely from
visual features (colour histograms, edge histograms, brightness).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from episodic_agent.core.interfaces import PerceptionModule
from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor
from episodic_agent.schemas.frames import Percept, SensorFrame
from episodic_agent.schemas.panorama_events import (
    PanoramaAgentState,
    PanoramaEvent,
    PanoramaEventType,
    PerceptionPayload,
)
from episodic_agent.schemas.visual import ExtractedVisualFeatures

if TYPE_CHECKING:
    from episodic_agent.modules.panorama.event_bus import PanoramaEventBus

logger = logging.getLogger(__name__)


class PanoramaPerception(PerceptionModule):
    """Perception module for panorama/video viewport crops.

    Extracts hand-crafted visual features and converts them into the
    128-dim scene embedding that LocationResolverReal consumes.

    Hypothesis tracking
    -------------------
    Maintains lightweight evidence counters so that the debug layer
    can display what the agent currently thinks about the scene.  The
    hypothesis itself is *not* used in inference — that stays in
    LocationResolverReal.  It exists only for human observability.
    """

    def __init__(self, event_bus: "PanoramaEventBus | None" = None) -> None:
        self._extractor = PanoramaFeatureExtractor()
        self._event_bus: PanoramaEventBus | None = event_bus
        self._investigation_sm: Any = None  # injected post-hoc

        # Rolling buffer of embeddings for one source image
        self._heading_embeddings: list[list[float]] = []
        self._current_source: str | None = None
        self._steps = 0

        # Lightweight hypothesis state (for debug / UI only)
        self.hypothesis: dict[str, Any] = {
            "label": None,
            "confidence": 0.0,
            "evidence_count": 0,
            "competing": [],
        }

        # Accumulated features for current image (for debug)
        self._accumulated_features: list[ExtractedVisualFeatures] = []

    # ------------------------------------------------------------------
    # PerceptionModule ABC
    # ------------------------------------------------------------------

    def process(self, frame: SensorFrame) -> Percept:
        """Convert a panorama SensorFrame into a Percept."""
        self._steps += 1

        # Handle transition marker frames (no image data)
        if frame.extras.get("transition"):
            return self._make_transition_percept(frame)

        # Decode viewport image
        img_rgb = self._decode_image(frame)
        if img_rgb is None:
            return self._make_empty_percept(frame)

        # Track source changes
        source_name = frame.extras.get("source_file", "")
        if source_name != self._current_source:
            self._flush_panoramic_embedding()
            self._current_source = source_name
            self._heading_embeddings.clear()
            self._accumulated_features.clear()

        # Extract features
        features = self._extractor.extract_from_rgb(
            img_rgb,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp.timestamp(),
        )
        self._accumulated_features.append(features)

        # Build per-viewport embedding
        embedding = features.to_embedding()
        self._heading_embeddings.append(embedding)

        # Check if this is the last heading for the image
        total_vp = frame.extras.get("total_viewports", 0)
        vp_idx = frame.extras.get("viewport_index", 0)
        is_last = (vp_idx >= total_vp) and total_vp > 0

        # Compute panoramic embedding on final heading
        panoramic: list[float] | None = None
        if is_last and self._heading_embeddings:
            panoramic = self._extractor.accumulate_panorama_embedding(
                self._heading_embeddings
            )

        percept = Percept(
            percept_id=str(uuid.uuid4()),
            source_frame_id=frame.frame_id,
            timestamp=datetime.now(),
            scene_embedding=panoramic if panoramic else embedding,
            candidates=[],  # no object detection in panorama mode
            confidence=self._compute_confidence(features),
            extras={
                "source_file": source_name,
                "heading_deg": frame.extras.get("heading_deg", 0.0),
                "viewport": frame.raw_data.get("viewport", {}),
                "viewport_embedding": embedding,
                "panoramic_embedding": panoramic,
                "is_last_heading": is_last,
                "feature_summary": {
                    "global_brightness": float(np.mean(features.cell_brightness))
                    if features.cell_brightness
                    else 0.0,
                    "global_edge_density": float(
                        np.mean(features.global_edge_histogram)
                    )
                    if features.global_edge_histogram
                    else 0.0,
                    "dominant_colors": [
                        list(c) for c in features.dominant_colors[:3]
                    ],
                    "scene_signature": features.scene_signature,
                },
                "camera_pose": frame.extras.get("camera_pose", {}),
                "hypothesis": dict(self.hypothesis),
            },
        )

        # Emit perception_update event
        self._emit_perception_update(
            percept, features, source_name, vp_idx, total_vp, is_last, embedding
        )

        return percept

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_image(self, frame: SensorFrame) -> np.ndarray | None:
        """Decode image bytes from the SensorFrame raw_data."""
        b64 = frame.raw_data.get("image_bytes_b64")
        if not b64:
            return None
        try:
            img_bytes = base64.b64decode(b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return np.array(pil_img)
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
            return None

    def _make_transition_percept(self, frame: SensorFrame) -> Percept:
        """Build a minimal Percept for a source transition."""
        panoramic = None
        if self._heading_embeddings:
            panoramic = self._extractor.accumulate_panorama_embedding(
                self._heading_embeddings
            )
        self._heading_embeddings.clear()
        self._accumulated_features.clear()
        self._current_source = frame.extras.get("source_file", "")

        return Percept(
            percept_id=str(uuid.uuid4()),
            source_frame_id=frame.frame_id,
            timestamp=datetime.now(),
            scene_embedding=panoramic if panoramic else [0.0] * 128,
            candidates=[],
            confidence=0.0,
            extras={
                "transition": True,
                "panoramic_embedding": panoramic,
                "camera_pose": frame.extras.get("camera_pose", {}),
            },
        )

    def _make_empty_percept(self, frame: SensorFrame) -> Percept:
        return Percept(
            percept_id=str(uuid.uuid4()),
            source_frame_id=frame.frame_id,
            timestamp=datetime.now(),
            scene_embedding=[0.0] * 128,
            candidates=[],
            confidence=0.0,
            extras={
                "camera_pose": frame.extras.get("camera_pose", {}),
            },
        )

    def _flush_panoramic_embedding(self) -> None:
        """Store the final panoramic embedding when switching sources."""
        self._heading_embeddings.clear()
        self._accumulated_features.clear()

    @staticmethod
    def _compute_confidence(features: ExtractedVisualFeatures) -> float:
        """Heuristic perception confidence based on feature quality."""
        score = 0.0
        # Non-trivial colour content
        if features.global_color_histogram:
            nonzero = sum(1 for v in features.global_color_histogram if v > 0.01)
            score += min(0.4, nonzero / 24 * 0.4)
        # Non-trivial edge content
        if features.global_edge_histogram:
            nonzero = sum(1 for v in features.global_edge_histogram if v > 0.01)
            score += min(0.3, nonzero / 8 * 0.3)
        # Brightness variance across cells → richer scene
        if features.cell_brightness:
            vals = np.array(features.cell_brightness)
            std = float(vals.std()) if len(vals) > 1 else 0.0
            score += min(0.3, std * 3.0)
        return min(1.0, score)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_accumulated_features(self) -> list[ExtractedVisualFeatures]:
        """Return accumulated features for current image (debug)."""
        return list(self._accumulated_features)

    def get_heading_count(self) -> int:
        return len(self._heading_embeddings)

    def get_feature_details(self) -> dict[str, Any]:
        """Return raw feature arrays for the current viewport (for UI).

        Unlike the summary stats in percept.extras, this exposes the
        full histogram arrays needed for detailed visualisation.
        """
        if not self._accumulated_features:
            return {}
        latest = self._accumulated_features[-1]
        # Compute per-cell edge density as the sum of each cell's edge
        # histogram (proxy for total edge energy in that cell).
        cell_edge_density = [
            float(sum(h)) for h in (latest.cell_edge_histograms or [])
        ]
        return {
            "color_histogram": list(latest.global_color_histogram or []),
            "edge_histogram": list(latest.global_edge_histogram or []),
            "cell_brightness": list(latest.cell_brightness or []),
            "cell_edge_density": cell_edge_density,
            "dominant_colors": [list(c) for c in (latest.dominant_colors or [])],
            "scene_signature": latest.scene_signature,
            "cell_count": len(latest.cell_brightness or []),
        }

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_perception_update(
        self,
        percept: Percept,
        features: ExtractedVisualFeatures,
        source_file: str,
        viewport_index: int,
        total_viewports: int,
        is_last_heading: bool,
        embedding: list[float],
    ) -> None:
        """Emit a structured perception_update event."""
        if not self._event_bus:
            return

        emb_norm = math.sqrt(sum(v * v for v in embedding)) if embedding else 0.0

        payload = PerceptionPayload(
            confidence=percept.confidence,
            feature_summary=percept.extras.get("feature_summary", {}),
            heading_index=viewport_index,
            total_headings=total_viewports,
            heading_deg=percept.extras.get("heading_deg", 0.0),
            is_panoramic_complete=is_last_heading,
            embedding_norm=emb_norm,
            source_file=source_file,
        )

        event = PanoramaEvent(
            event_type=PanoramaEventType.perception_update,
            timestamp=datetime.now(),
            step=self._steps,
            state=(
                self._investigation_sm.state
                if self._investigation_sm
                else PanoramaAgentState.investigating_unknown
            ),
            payload=payload.model_dump(),
        )
        self._event_bus.emit(event)
