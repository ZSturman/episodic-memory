"""Panorama sensor provider — loads images/video from a folder.

Implements the SensorProvider ABC.  Scans a directory for image files,
then advances a sliding viewport across each image to simulate an agent
"looking around".  Viewport positioning is delegated to SaccadePolicy.

Each call to ``get_frame()`` returns a single SensorFrame whose
``raw_data`` carries the JPEG-encoded viewport crop and metadata.
The ``extras`` dict carries a synthetic ``camera_pose`` so that
downstream modules (LocationResolverReal) have a forward vector.

Video files (.mp4, .avi, .mov) are supported as well: frames are
decoded at a configurable FPS and each decoded frame is treated like
a single-heading image.

Single-image mode
-----------------
When using the hex-grid scanning pipeline, call ``get_current_image()``
to retrieve the full-resolution image as an RGB numpy array and
``advance_to_next_image()`` to step to the next source file.  This
bypasses the viewport-sweep mechanism entirely.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from episodic_agent.core.interfaces import SensorProvider
from episodic_agent.modules.panorama.saccade import SaccadePolicy, Viewport
from episodic_agent.schemas.frames import SensorFrame

logger = logging.getLogger(__name__)

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class PanoramaSensorProvider(SensorProvider):
    """Sensor provider that reads images/video from a local folder.

    Parameters
    ----------
    image_dir : Path
        Directory containing image and/or video files.
    viewport_width, viewport_height : int
        Size of each viewport crop in pixels.
    headings_per_image : int
        Number of evenly-spaced horizontal stops per image sweep.
    video_fps : float
        Target frame-rate for video decode (frames per second).
    seed : int
        Random seed for saccade policy.
    """

    def __init__(
        self,
        image_dir: Path,
        viewport_width: int = 256,
        viewport_height: int = 256,
        headings_per_image: int = 8,
        video_fps: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.vp_w = viewport_width
        self.vp_h = viewport_height
        self.headings_per_image = headings_per_image
        self.video_fps = video_fps

        # Discover source files
        self._sources: list[Path] = self._discover_sources()
        if not self._sources:
            logger.warning(f"No images or videos found in {self.image_dir}")

        # Saccade controller
        self._saccade = SaccadePolicy(
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            headings=headings_per_image,
            seed=seed,
        )

        # Iteration state
        self._source_idx = 0          # index into self._sources
        self._frame_id = 0            # monotonic counter
        self._current_image: np.ndarray | None = None
        self._current_source: Path | None = None

        # Video state
        self._video_cap: cv2.VideoCapture | None = None
        self._video_frame_interval: float = 0.0
        self._video_next_ts: float = 0.0

        # Overall done flag
        self._exhausted = False

        # Preload first source
        self._load_next_source()

    # ------------------------------------------------------------------
    # SensorProvider ABC
    # ------------------------------------------------------------------

    def get_frame(self) -> SensorFrame:
        """Return the next SensorFrame with a viewport crop."""
        if self._exhausted:
            raise StopIteration("All sources exhausted")

        # If image mode: get next viewport from saccade policy
        if self._current_image is not None:
            vp = self._saccade.next_viewport(
                self._current_image.shape[1],
                self._current_image.shape[0],
            )

            if vp is None:
                # Sweep complete — move to next source
                return self._advance_source()

            crop = self._crop_viewport(self._current_image, vp)
            frame = self._build_frame(crop, vp, is_transition=False)
            return frame

        # If video mode: decode next frame
        if self._video_cap is not None:
            return self._next_video_frame()

        # Fallback — advance
        return self._advance_source()

    def has_frames(self) -> bool:
        return not self._exhausted

    def reset(self) -> None:
        """Reset to beginning of the source list."""
        self._close_video()
        self._source_idx = 0
        self._frame_id = 0
        self._current_image = None
        self._current_source = None
        self._exhausted = False
        self._saccade.reset()
        if self._sources:
            self._load_next_source()

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _discover_sources(self) -> list[Path]:
        """Find all image/video files sorted alphabetically."""
        if not self.image_dir.is_dir():
            return []
        files: list[Path] = []
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                files.append(p)
        return files

    def _load_next_source(self) -> None:
        """Load the current source file (image or video)."""
        self._close_video()
        self._current_image = None
        self._current_source = None

        if self._source_idx >= len(self._sources):
            self._exhausted = True
            return

        path = self._sources[self._source_idx]
        self._current_source = path

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to load image: {path}")
                self._source_idx += 1
                self._load_next_source()
                return
            self._current_image = img
            self._saccade.prepare_for_image(img.shape[1], img.shape[0])
            logger.info(
                f"Loaded image {path.name} "
                f"({img.shape[1]}×{img.shape[0]}, "
                f"{self._saccade.total_viewports()} viewports)"
            )

        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.warning(f"Failed to open video: {path}")
                self._source_idx += 1
                self._load_next_source()
                return
            self._video_cap = cap
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._video_frame_interval = native_fps / max(0.1, self.video_fps)
            self._video_next_ts = 0.0
            logger.info(
                f"Opened video {path.name} "
                f"(native {native_fps:.1f} fps, sampling at {self.video_fps} fps)"
            )
        else:
            self._source_idx += 1
            self._load_next_source()

    def _advance_source(self) -> SensorFrame:
        """Move to the next source file and emit a transition frame."""
        self._source_idx += 1
        self._load_next_source()

        # Transition marker frame
        self._frame_id += 1
        return SensorFrame(
            frame_id=self._frame_id,
            timestamp=datetime.now(),
            sensor_type="panorama",
            raw_data={},
            extras={
                "transition": True,
                "source_file": str(self._current_source) if self._current_source else "",
                "camera_pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "forward": [1.0, 0.0, 0.0],
                },
            },
        )

    def _close_video(self) -> None:
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------

    def _next_video_frame(self) -> SensorFrame:
        """Decode the next sampled frame from the video."""
        assert self._video_cap is not None

        while True:
            ret, frame = self._video_cap.read()
            if not ret:
                return self._advance_source()

            current_pos = self._video_cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_pos >= self._video_next_ts:
                self._video_next_ts = current_pos + self._video_frame_interval
                break

        # Treat the whole decoded frame as a single viewport
        h, w = frame.shape[:2]
        vp = Viewport(x=0, y=0, width=w, height=h, heading_deg=0.0)
        return self._build_frame(frame, vp, is_transition=False)

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    def _crop_viewport(self, img: np.ndarray, vp: Viewport) -> np.ndarray:
        """Extract the viewport rectangle from the image."""
        h, w = img.shape[:2]
        x1 = max(0, min(vp.x, w - 1))
        y1 = max(0, min(vp.y, h - 1))
        x2 = min(x1 + vp.width, w)
        y2 = min(y1 + vp.height, h)
        return img[y1:y2, x1:x2]

    def _build_frame(
        self,
        crop_bgr: np.ndarray,
        vp: Viewport,
        is_transition: bool,
    ) -> SensorFrame:
        """Package a viewport crop into a SensorFrame."""
        self._frame_id += 1

        # Encode crop as JPEG bytes
        _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buf.tobytes()

        # Forward vector from heading
        rad = math.radians(vp.heading_deg)
        forward = [math.cos(rad), 0.0, math.sin(rad)]

        source_name = self._current_source.name if self._current_source else "unknown"

        return SensorFrame(
            frame_id=self._frame_id,
            timestamp=datetime.now(),
            sensor_type="panorama",
            raw_data={
                "image_bytes_b64": base64.b64encode(image_bytes).decode("ascii"),
                "image_width": crop_bgr.shape[1],
                "image_height": crop_bgr.shape[0],
                "viewport": vp.as_dict(),
                "image_file": source_name,
            },
            extras={
                "transition": is_transition,
                "source_file": source_name,
                "source_index": self._source_idx,
                "heading_deg": vp.heading_deg,
                "viewport_index": self._saccade.current_index(),
                "total_viewports": self._saccade.total_viewports(),
                "camera_pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "forward": forward,
                },
            },
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def current_source_name(self) -> str | None:
        return self._current_source.name if self._current_source else None

    @property
    def current_source_index(self) -> int:
        return self._source_idx

    def get_status(self) -> dict[str, Any]:
        return {
            "image_dir": str(self.image_dir),
            "source_count": len(self._sources),
            "source_index": self._source_idx,
            "frame_id": self._frame_id,
            "exhausted": self._exhausted,
            "current_source": str(self._current_source) if self._current_source else None,
            "saccade_mode": self._saccade.mode.value if hasattr(self, '_saccade') else "hex",
        }

    # ------------------------------------------------------------------
    # Single-image mode (hex-grid pipeline)
    # ------------------------------------------------------------------

    def get_current_image(self) -> np.ndarray | None:
        """Return the current full-resolution image as RGB.

        Returns ``None`` when all sources are exhausted or between
        sources.  The image is converted from BGR (OpenCV default) to
        RGB for consistency with the hex feature extractor.
        """
        if self._current_image is None:
            return None
        return cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)

    def get_current_image_path(self) -> Path | None:
        """Return the filesystem path to the current source file."""
        return self._current_source

    def advance_to_next_image(self) -> bool:
        """Advance to the next image source.

        Returns ``True`` if a new image was loaded, ``False`` if no
        more sources remain.  Video sources are skipped (use
        ``get_frame()`` for those).
        """
        self._source_idx += 1
        while self._source_idx < len(self._sources):
            path = self._sources[self._source_idx]
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                self._load_next_source()
                if self._current_image is not None:
                    return True
            else:
                self._source_idx += 1
        self._exhausted = True
        return False

    def peek_remaining(self) -> int:
        """Return the number of remaining sources (including current)."""
        if self._exhausted:
            return 0
        return len(self._sources) - self._source_idx

    def stop(self) -> None:
        """Clean up resources."""
        self._close_video()
