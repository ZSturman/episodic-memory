"""Visual Stream Client for bandwidth-efficient visual sensing.

ARCHITECTURAL INVARIANT: Bandwidth-Efficient Sensing
- Visual data is summarized via 4×4 grid, not raw-streamed
- High-res on demand only via focus requests
- Raw images auto-evicted from ring buffer after extraction
- Features persisted, raw pixels discarded

This module provides:
1. VisualRingBuffer - Fixed-size buffer for recent high-res frames
2. VisualFeatureExtractor - Extract histograms/edges from images
3. VisualStreamClient - WebSocket client for visual channel
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from episodic_agent.schemas.visual import (
    AttentionMode,
    ExtractedVisualFeatures,
    FocusRequest,
    FocusRequestType,
    FocusResponse,
    GridCell,
    RingBufferEntry,
    VisualAttentionState,
    VisualSummaryFrame,
    COLOR_HISTOGRAM_BINS,
    EDGE_DIRECTION_BINS,
    GRID_CELL_COUNT,
    GRID_COLS,
    GRID_ROWS,
    RING_BUFFER_MAX_FRAMES,
    RING_BUFFER_SIZE_MB,
    VISUAL_CHANNEL_PORT,
    VISUAL_SUMMARY_RATE_HZ,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Visual Ring Buffer
# =============================================================================

class VisualRingBuffer:
    """Fixed-size ring buffer for high-resolution frames.
    
    ARCHITECTURAL INVARIANT: Fixed memory budget.
    Buffer auto-evicts oldest frames to stay within ~50MB.
    
    Features:
    - O(1) add/remove operations
    - Size-based eviction (not just count-based)
    - Frame lookup by frame_id
    - Memory usage tracking
    """
    
    def __init__(
        self,
        max_size_mb: int = RING_BUFFER_SIZE_MB,
        max_frames: int = RING_BUFFER_MAX_FRAMES,
    ) -> None:
        """Initialize the ring buffer.
        
        Args:
            max_size_mb: Maximum buffer size in megabytes.
            max_frames: Maximum number of frames to store.
        """
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_frames = max_frames
        
        # Storage
        self._buffer: deque[RingBufferEntry] = deque(maxlen=max_frames)
        self._index: dict[int, RingBufferEntry] = {}  # frame_id → entry
        
        # Stats
        self._current_size_bytes = 0
        self._frames_evicted = 0
        self._frames_added = 0
    
    def add(self, entry: RingBufferEntry) -> bool:
        """Add a frame to the buffer.
        
        May evict oldest frames if size limit exceeded.
        
        Args:
            entry: Frame entry to add.
            
        Returns:
            True if added successfully.
        """
        # Evict if we're over size limit
        while (
            self._current_size_bytes + entry.size_bytes > self._max_size_bytes
            and self._buffer
        ):
            self._evict_oldest()
        
        # Check count limit (handled by deque maxlen, but we need to update index)
        if len(self._buffer) >= self._max_frames:
            self._evict_oldest()
        
        # Add to buffer
        self._buffer.append(entry)
        self._index[entry.frame_id] = entry
        self._current_size_bytes += entry.size_bytes
        self._frames_added += 1
        
        logger.debug(
            f"[RING] Added frame {entry.frame_id} "
            f"({entry.size_bytes / 1024:.1f} KB, "
            f"total: {self._current_size_bytes / 1024 / 1024:.1f} MB)"
        )
        
        return True
    
    def get(self, frame_id: int) -> RingBufferEntry | None:
        """Get a frame by ID.
        
        Args:
            frame_id: Frame ID to retrieve.
            
        Returns:
            RingBufferEntry if found, None otherwise.
        """
        return self._index.get(frame_id)
    
    def get_latest(self) -> RingBufferEntry | None:
        """Get the most recent frame."""
        if self._buffer:
            return self._buffer[-1]
        return None
    
    def has_frame(self, frame_id: int) -> bool:
        """Check if frame is in buffer."""
        return frame_id in self._index
    
    def _evict_oldest(self) -> None:
        """Evict the oldest frame from buffer."""
        if self._buffer:
            oldest = self._buffer.popleft()
            del self._index[oldest.frame_id]
            self._current_size_bytes -= oldest.size_bytes
            self._frames_evicted += 1
            
            logger.debug(f"[RING] Evicted frame {oldest.frame_id}")
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        self._buffer.clear()
        self._index.clear()
        self._current_size_bytes = 0
    
    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        return {
            "current_frames": len(self._buffer),
            "max_frames": self._max_frames,
            "current_size_mb": self._current_size_bytes / 1024 / 1024,
            "max_size_mb": self._max_size_bytes / 1024 / 1024,
            "utilization_pct": (
                self._current_size_bytes / self._max_size_bytes * 100
                if self._max_size_bytes > 0
                else 0
            ),
            "frames_added": self._frames_added,
            "frames_evicted": self._frames_evicted,
        }
    
    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return len(self._buffer)
    
    def __iter__(self):
        """Iterate over frames (newest to oldest)."""
        return reversed(self._buffer)


# =============================================================================
# Visual Feature Extractor
# =============================================================================

class VisualFeatureExtractor:
    """Extract features from visual data.
    
    ARCHITECTURAL INVARIANT: Raw images discarded after extraction.
    Only derived features are persisted.
    
    Features extracted:
    - Color histograms (RGB, 8 bins per channel)
    - Edge direction histograms (8 directions)
    - Brightness statistics
    - Motion estimation (frame difference)
    """
    
    def __init__(
        self,
        color_bins: int = COLOR_HISTOGRAM_BINS,
        edge_bins: int = EDGE_DIRECTION_BINS,
    ) -> None:
        """Initialize the feature extractor.
        
        Args:
            color_bins: Number of bins per color channel.
            edge_bins: Number of edge direction bins.
        """
        self._color_bins = color_bins
        self._edge_bins = edge_bins
        
        # Previous frame for motion estimation
        self._prev_summary: VisualSummaryFrame | None = None
        
        # Stats
        self._frames_processed = 0
    
    def extract_from_summary(
        self,
        summary: VisualSummaryFrame,
        compute_motion: bool = True,
    ) -> ExtractedVisualFeatures:
        """Extract features from a visual summary frame.
        
        Args:
            summary: The visual summary frame.
            compute_motion: Whether to compute motion from previous frame.
            
        Returns:
            Extracted visual features.
        """
        features = ExtractedVisualFeatures(
            frame_id=summary.frame_id,
            timestamp=summary.timestamp,
        )
        
        # Extract per-cell features
        for cell in summary.cells:
            features.cell_color_histograms.append(cell.color_histogram.copy())
            features.cell_edge_histograms.append(cell.edge_directions.copy())
            features.cell_brightness.append(cell.mean_brightness)
            
            # Compute motion if we have previous frame
            if compute_motion and self._prev_summary:
                prev_cell = self._prev_summary.get_cell(cell.row, cell.col)
                if prev_cell:
                    motion = self._compute_cell_motion(cell, prev_cell)
                    features.cell_motion.append(motion)
                else:
                    features.cell_motion.append(0.0)
            else:
                features.cell_motion.append(cell.motion_magnitude)
        
        # Global features
        features.global_color_histogram = self._aggregate_color_histograms(
            features.cell_color_histograms
        )
        features.global_edge_histogram = self._aggregate_edge_histograms(
            features.cell_edge_histograms
        )
        features.dominant_colors = list(summary.dominant_colors)
        
        # Compute scene signature (hash of features)
        features.scene_signature = self._compute_scene_signature(features)
        
        # Update for next frame
        self._prev_summary = summary
        self._frames_processed += 1
        
        return features
    
    def extract_from_raw_image(
        self,
        image_data: bytes,
        frame_id: int,
        timestamp: float,
    ) -> ExtractedVisualFeatures | None:
        """Extract features from raw image data.
        
        ARCHITECTURAL INVARIANT: Image is NOT stored after extraction.
        
        Args:
            image_data: Raw image bytes (JPEG/PNG).
            frame_id: Frame identifier.
            timestamp: Frame timestamp.
            
        Returns:
            Extracted features, or None if extraction failed.
        """
        try:
            # Import here to avoid hard dependency on image processing libs
            import io
            
            try:
                from PIL import Image
                import numpy as np
            except ImportError:
                logger.warning("[VISUAL] PIL/numpy not available, skipping raw extraction")
                return None
            
            # Decode image
            img = Image.open(io.BytesIO(image_data))
            img_array = np.array(img.convert("RGB"))
            
            # Extract global features
            features = ExtractedVisualFeatures(
                frame_id=frame_id,
                timestamp=timestamp,
            )
            
            # Color histogram
            features.global_color_histogram = self._compute_color_histogram(img_array)
            
            # Edge histogram
            features.global_edge_histogram = self._compute_edge_histogram(img_array)
            
            # Dominant colors
            features.dominant_colors = self._compute_dominant_colors(img_array)
            
            # Grid cell features
            height, width = img_array.shape[:2]
            cell_h, cell_w = height // GRID_ROWS, width // GRID_COLS
            
            for row in range(GRID_ROWS):
                for col in range(GRID_COLS):
                    y1, y2 = row * cell_h, (row + 1) * cell_h
                    x1, x2 = col * cell_w, (col + 1) * cell_w
                    cell_img = img_array[y1:y2, x1:x2]
                    
                    features.cell_color_histograms.append(
                        self._compute_color_histogram(cell_img)
                    )
                    features.cell_edge_histograms.append(
                        self._compute_edge_histogram(cell_img)
                    )
                    features.cell_brightness.append(
                        float(np.mean(cell_img) / 255.0)
                    )
                    features.cell_motion.append(0.0)  # No motion for single frame
            
            # Scene signature
            features.scene_signature = self._compute_scene_signature(features)
            
            self._frames_processed += 1
            
            logger.debug(f"[VISUAL] Extracted features from raw frame {frame_id}")
            
            return features
            
        except Exception as e:
            logger.error(f"[VISUAL] Failed to extract from raw image: {e}")
            return None
    
    def _compute_color_histogram(self, img_array: Any) -> list[float]:
        """Compute RGB color histogram."""
        try:
            import numpy as np
            
            histograms = []
            for channel in range(3):  # R, G, B
                hist, _ = np.histogram(
                    img_array[:, :, channel],
                    bins=self._color_bins,
                    range=(0, 256),
                )
                # Normalize
                hist = hist.astype(float)
                if hist.sum() > 0:
                    hist = hist / hist.sum()
                histograms.extend(hist.tolist())
            
            return histograms
            
        except Exception:
            return [0.0] * (self._color_bins * 3)
    
    def _compute_edge_histogram(self, img_array: Any) -> list[float]:
        """Compute edge direction histogram."""
        try:
            import numpy as np
            
            # Convert to grayscale
            gray = np.mean(img_array, axis=2)
            
            # Simple Sobel-like edge detection
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
            
            # Edge magnitude and direction
            magnitude = np.sqrt(gx**2 + gy**2)
            direction = np.arctan2(gy, gx)  # -π to π
            
            # Bin directions (0 to 2π mapped to bins)
            direction_normalized = (direction + np.pi) / (2 * np.pi)  # 0 to 1
            bin_indices = (direction_normalized * self._edge_bins).astype(int)
            bin_indices = np.clip(bin_indices, 0, self._edge_bins - 1)
            
            # Weight by magnitude
            hist = np.zeros(self._edge_bins)
            for i in range(self._edge_bins):
                hist[i] = np.sum(magnitude[bin_indices == i])
            
            # Normalize
            if hist.sum() > 0:
                hist = hist / hist.sum()
            
            return hist.tolist()
            
        except Exception:
            return [0.0] * self._edge_bins
    
    def _compute_dominant_colors(
        self,
        img_array: Any,
        k: int = 3,
    ) -> list[tuple[int, int, int]]:
        """Compute dominant colors using simple histogram peaks."""
        try:
            import numpy as np
            
            # Quantize colors
            quantized = (img_array // 32) * 32 + 16  # 8 levels per channel
            
            # Flatten to RGB tuples
            pixels = quantized.reshape(-1, 3)
            
            # Count unique colors
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Get top k
            top_indices = np.argsort(counts)[-k:][::-1]
            
            return [tuple(map(int, unique[i])) for i in top_indices]
            
        except Exception:
            return [(128, 128, 128)]  # Gray as fallback
    
    def _compute_cell_motion(
        self,
        current: GridCell,
        previous: GridCell,
    ) -> float:
        """Compute motion between two cells."""
        if not current.color_histogram or not previous.color_histogram:
            return 0.0
        
        # Histogram difference as proxy for motion
        total_diff = 0.0
        min_len = min(len(current.color_histogram), len(previous.color_histogram))
        
        for i in range(min_len):
            total_diff += abs(current.color_histogram[i] - previous.color_histogram[i])
        
        # Normalize to 0-1 range
        return min(1.0, total_diff / 2.0)
    
    def _aggregate_color_histograms(
        self,
        cell_histograms: list[list[float]],
    ) -> list[float]:
        """Aggregate cell histograms into global histogram."""
        if not cell_histograms:
            return [0.0] * (self._color_bins * 3)
        
        # Average across cells
        hist_len = len(cell_histograms[0])
        result = [0.0] * hist_len
        
        for hist in cell_histograms:
            for i in range(min(hist_len, len(hist))):
                result[i] += hist[i]
        
        # Normalize
        total = sum(result)
        if total > 0:
            result = [v / total for v in result]
        
        return result
    
    def _aggregate_edge_histograms(
        self,
        cell_histograms: list[list[float]],
    ) -> list[float]:
        """Aggregate cell edge histograms."""
        if not cell_histograms:
            return [0.0] * self._edge_bins
        
        # Average across cells
        result = [0.0] * self._edge_bins
        
        for hist in cell_histograms:
            for i in range(min(self._edge_bins, len(hist))):
                result[i] += hist[i]
        
        # Normalize
        total = sum(result)
        if total > 0:
            result = [v / total for v in result]
        
        return result
    
    def _compute_scene_signature(
        self,
        features: ExtractedVisualFeatures,
    ) -> str:
        """Compute a hash signature for quick scene comparison."""
        # Quantize key features to create stable signature
        sig_parts = []
        
        # Add global brightness pattern
        for b in features.cell_brightness[:16]:
            sig_parts.append(str(int(b * 10)))
        
        # Add dominant color info
        for color in features.dominant_colors[:3]:
            sig_parts.append(f"{color[0]//32}{color[1]//32}{color[2]//32}")
        
        sig_str = "_".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()[:12]
    
    def get_statistics(self) -> dict[str, Any]:
        """Get extractor statistics."""
        return {
            "frames_processed": self._frames_processed,
            "color_bins": self._color_bins,
            "edge_bins": self._edge_bins,
        }


# =============================================================================
# Visual Attention Manager
# =============================================================================

class VisualAttentionManager:
    """Manages visual attention and focus requests.
    
    Determines where to focus attention based on:
    - Motion detection (salient changes)
    - Edge density (interesting regions)
    - User-guided focus (manual attention)
    - Entity tracking (following objects)
    """
    
    def __init__(
        self,
        motion_weight: float = 0.4,
        edge_weight: float = 0.2,
        brightness_change_weight: float = 0.2,
        recency_weight: float = 0.2,
    ) -> None:
        """Initialize attention manager.
        
        Args:
            motion_weight: Weight for motion in saliency.
            edge_weight: Weight for edge density.
            brightness_change_weight: Weight for brightness changes.
            recency_weight: Weight for recently attended regions.
        """
        self._motion_weight = motion_weight
        self._edge_weight = edge_weight
        self._brightness_weight = brightness_change_weight
        self._recency_weight = recency_weight
        
        # State
        self._state = VisualAttentionState()
        self._prev_brightness: list[float] = [0.0] * GRID_CELL_COUNT
        
        # Pending focus requests
        self._pending_requests: list[FocusRequest] = []
        
        # Stats
        self._focus_requests_made = 0
    
    def update_from_summary(
        self,
        summary: VisualSummaryFrame,
    ) -> VisualAttentionState:
        """Update attention state from a visual summary.
        
        Args:
            summary: The visual summary frame.
            
        Returns:
            Updated attention state.
        """
        # Compute saliency for each cell
        for cell in summary.cells:
            idx = cell.row * GRID_COLS + cell.col
            
            # Compute saliency components
            motion_sal = cell.motion_magnitude * self._motion_weight
            edge_sal = cell.edge_density * self._edge_weight
            
            # Brightness change
            brightness_change = abs(
                cell.mean_brightness - self._prev_brightness[idx]
            ) if idx < len(self._prev_brightness) else 0.0
            brightness_sal = brightness_change * self._brightness_weight
            
            # Update saliency map
            total_sal = motion_sal + edge_sal + brightness_sal
            self._state.update_saliency(idx, total_sal)
            
            # Store cell attention weight
            cell.attention_weight = self._state.saliency_map[idx]
            
            # Update previous brightness
            if idx < len(self._prev_brightness):
                self._prev_brightness[idx] = cell.mean_brightness
        
        # Add to attention history
        most_salient = self._state.get_most_salient()
        self._state.attention_history.append(
            (most_salient[0], most_salient[1], summary.timestamp)
        )
        
        # Trim history
        if len(self._state.attention_history) > 100:
            self._state.attention_history = self._state.attention_history[-100:]
        
        return self._state
    
    def should_investigate(
        self,
        threshold: float = 0.5,
    ) -> tuple[bool, int, int] | tuple[bool, None, None]:
        """Check if any cell warrants investigation.
        
        Args:
            threshold: Saliency threshold for investigation.
            
        Returns:
            Tuple of (should_investigate, row, col) or (False, None, None).
        """
        max_sal = max(self._state.saliency_map)
        
        if max_sal >= threshold:
            row, col = self._state.get_most_salient()
            return True, row, col
        
        return False, None, None
    
    def create_focus_request(
        self,
        row: int,
        col: int,
        reason: str = "saliency",
        priority: int = 5,
        duration: float = 1.0,
    ) -> FocusRequest:
        """Create a focus request for a cell.
        
        Args:
            row: Target row.
            col: Target column.
            reason: Reason for focus.
            priority: Request priority.
            duration: Focus duration.
            
        Returns:
            FocusRequest object.
        """
        request = FocusRequest(
            request_id=f"focus_{uuid.uuid4().hex[:8]}",
            request_type=FocusRequestType.REGION,
            target_row=row,
            target_col=col,
            duration_seconds=duration,
            priority=priority,
        )
        
        self._pending_requests.append(request)
        self._focus_requests_made += 1
        
        # Update state
        self._state.mode = AttentionMode.INVESTIGATING
        self._state.focus_cell = (row, col)
        self._state.investigating_since = time.time()
        self._state.investigation_reason = reason
        
        logger.info(
            f"[ATTENTION] Focus request: cell ({row}, {col}), reason: {reason}"
        )
        
        return request
    
    def create_entity_track_request(
        self,
        entity_guid: str,
        duration: float = 5.0,
    ) -> FocusRequest:
        """Create a request to track an entity.
        
        Args:
            entity_guid: Entity GUID to track.
            duration: Tracking duration.
            
        Returns:
            FocusRequest object.
        """
        request = FocusRequest(
            request_id=f"track_{uuid.uuid4().hex[:8]}",
            request_type=FocusRequestType.TRACK_ENTITY,
            entity_guid=entity_guid,
            duration_seconds=duration,
            priority=7,  # Tracking gets higher priority
        )
        
        self._pending_requests.append(request)
        self._focus_requests_made += 1
        
        # Update state
        self._state.mode = AttentionMode.TRACKING
        self._state.focus_entity_guid = entity_guid
        
        logger.info(f"[ATTENTION] Track request: entity {entity_guid}")
        
        return request
    
    def get_pending_requests(self) -> list[FocusRequest]:
        """Get and clear pending focus requests."""
        requests = self._pending_requests.copy()
        self._pending_requests.clear()
        return requests
    
    def reset_investigation(self) -> None:
        """Reset investigation state."""
        self._state.mode = AttentionMode.PASSIVE
        self._state.focus_cell = None
        self._state.focus_entity_guid = None
        self._state.investigating_since = None
        self._state.investigation_reason = ""
    
    @property
    def state(self) -> VisualAttentionState:
        """Get current attention state."""
        return self._state
    
    def get_statistics(self) -> dict[str, Any]:
        """Get attention manager statistics."""
        return {
            "mode": self._state.mode.value,
            "focus_cell": self._state.focus_cell,
            "focus_entity": self._state.focus_entity_guid,
            "focus_requests_made": self._focus_requests_made,
            "pending_requests": len(self._pending_requests),
            "max_saliency": max(self._state.saliency_map) if self._state.saliency_map else 0,
        }


# =============================================================================
# Visual Stream Client
# =============================================================================

class VisualStreamClient:
    """WebSocket client for visual summary channel.
    
    Connects to Unity visual channel on port 8767 to receive:
    - 4×4 grid summaries at ~5Hz
    - High-res crops on demand
    
    ARCHITECTURAL INVARIANT: Python extracts features, discards raw images.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = VISUAL_CHANNEL_PORT,
        on_summary: Callable[[VisualSummaryFrame], None] | None = None,
        on_focus_response: Callable[[FocusResponse], None] | None = None,
        buffer_size_mb: int = RING_BUFFER_SIZE_MB,
    ) -> None:
        """Initialize the visual stream client.
        
        Args:
            host: WebSocket server host.
            port: WebSocket server port.
            on_summary: Callback for summary frames.
            on_focus_response: Callback for focus responses.
            buffer_size_mb: Ring buffer size in MB.
        """
        self._host = host
        self._port = port
        self._on_summary = on_summary
        self._on_focus_response = on_focus_response
        
        # Components
        self._ring_buffer = VisualRingBuffer(max_size_mb=buffer_size_mb)
        self._feature_extractor = VisualFeatureExtractor()
        self._attention_manager = VisualAttentionManager()
        
        # Connection state
        self._connected = False
        self._websocket = None
        self._receive_task = None
        
        # Stats
        self._summaries_received = 0
        self._focus_responses_received = 0
        self._connection_errors = 0
    
    @property
    def url(self) -> str:
        """Get WebSocket URL."""
        return f"ws://{self._host}:{self._port}"
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    @property
    def ring_buffer(self) -> VisualRingBuffer:
        """Get the ring buffer."""
        return self._ring_buffer
    
    @property
    def feature_extractor(self) -> VisualFeatureExtractor:
        """Get the feature extractor."""
        return self._feature_extractor
    
    @property
    def attention_manager(self) -> VisualAttentionManager:
        """Get the attention manager."""
        return self._attention_manager
    
    async def connect(self) -> bool:
        """Connect to the visual channel.
        
        Returns:
            True if connection successful.
        """
        try:
            import websockets
            
            self._websocket = await websockets.connect(self.url)
            self._connected = True
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            logger.info(f"[VISUAL] Connected to {self.url}")
            return True
            
        except ImportError:
            logger.error("[VISUAL] websockets library not installed")
            return False
        except Exception as e:
            logger.error(f"[VISUAL] Connection failed: {e}")
            self._connection_errors += 1
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the visual channel."""
        self._connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        logger.info("[VISUAL] Disconnected")
    
    async def _receive_loop(self) -> None:
        """Main receive loop for WebSocket messages."""
        try:
            while self._connected and self._websocket:
                try:
                    message = await self._websocket.recv()
                    await self._handle_message(message)
                except Exception as e:
                    if self._connected:
                        logger.error(f"[VISUAL] Receive error: {e}")
                        self._connection_errors += 1
                    break
        except asyncio.CancelledError:
            pass
    
    async def _handle_message(self, message: str | bytes) -> None:
        """Handle incoming WebSocket message.
        
        Args:
            message: Raw message from WebSocket.
        """
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            
            data = json.loads(message)
            msg_type = data.get("type", "summary")
            
            if msg_type == "summary":
                await self._handle_summary(data)
            elif msg_type == "focus_response":
                await self._handle_focus_response(data)
            else:
                logger.warning(f"[VISUAL] Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"[VISUAL] JSON decode error: {e}")
        except Exception as e:
            logger.error(f"[VISUAL] Message handling error: {e}")
    
    async def _handle_summary(self, data: dict[str, Any]) -> None:
        """Handle visual summary message."""
        try:
            # Parse cells
            cells = []
            for cell_data in data.get("cells", []):
                cell = GridCell(
                    row=cell_data.get("row", 0),
                    col=cell_data.get("col", 0),
                    color_histogram=cell_data.get("color_histogram", []),
                    edge_directions=cell_data.get("edge_directions", []),
                    mean_brightness=cell_data.get("mean_brightness", 0.0),
                    edge_density=cell_data.get("edge_density", 0.0),
                    motion_magnitude=cell_data.get("motion_magnitude", 0.0),
                )
                cells.append(cell)
            
            # Create summary
            summary = VisualSummaryFrame(
                frame_id=data.get("frame_id", 0),
                timestamp=data.get("timestamp", time.time()),
                cells=cells,
                global_brightness=data.get("global_brightness", 0.0),
                global_edge_density=data.get("global_edge_density", 0.0),
                global_motion=data.get("global_motion", 0.0),
                dominant_colors=[
                    tuple(c) for c in data.get("dominant_colors", [])
                ],
                high_res_available=data.get("high_res_available", False),
            )
            
            # Update attention
            self._attention_manager.update_from_summary(summary)
            
            # Extract features (raw image discarded after this)
            self._feature_extractor.extract_from_summary(summary)
            
            self._summaries_received += 1
            
            # Callback
            if self._on_summary:
                self._on_summary(summary)
            
            logger.debug(f"[VISUAL] Summary received: frame {summary.frame_id}")
            
        except Exception as e:
            logger.error(f"[VISUAL] Summary handling error: {e}")
    
    async def _handle_focus_response(self, data: dict[str, Any]) -> None:
        """Handle focus response message."""
        try:
            response = FocusResponse(
                request_id=data.get("request_id", ""),
                success=data.get("success", False),
                error_message=data.get("error_message"),
                image_data=data.get("image_data"),
                image_format=data.get("image_format", "jpeg"),
                image_width=data.get("image_width", 0),
                image_height=data.get("image_height", 0),
                source_frame_id=data.get("source_frame_id"),
                crop_x=data.get("crop_x", 0),
                crop_y=data.get("crop_y", 0),
                crop_width=data.get("crop_width", 0),
                crop_height=data.get("crop_height", 0),
            )
            
            # Store high-res in ring buffer if available
            if response.success and response.image_data:
                image_bytes = base64.b64decode(response.image_data)
                entry = RingBufferEntry(
                    frame_id=response.source_frame_id or 0,
                    timestamp=response.timestamp,
                    image_data=image_bytes,
                    image_format=response.image_format,
                    image_width=response.image_width,
                    image_height=response.image_height,
                )
                self._ring_buffer.add(entry)
            
            self._focus_responses_received += 1
            
            # Reset investigation state
            self._attention_manager.reset_investigation()
            
            # Callback
            if self._on_focus_response:
                self._on_focus_response(response)
            
            logger.debug(f"[VISUAL] Focus response: {response.request_id}")
            
        except Exception as e:
            logger.error(f"[VISUAL] Focus response handling error: {e}")
    
    async def send_focus_request(self, request: FocusRequest) -> bool:
        """Send a focus request to the visual channel.
        
        Args:
            request: The focus request to send.
            
        Returns:
            True if sent successfully.
        """
        if not self._connected or not self._websocket:
            logger.warning("[VISUAL] Cannot send focus request: not connected")
            return False
        
        try:
            message = {
                "type": "focus_request",
                "request_id": request.request_id,
                "request_type": request.request_type.value,
                "target_row": request.target_row,
                "target_col": request.target_col,
                "frame_id": request.frame_id,
                "entity_guid": request.entity_guid,
                "duration_seconds": request.duration_seconds,
                "priority": request.priority,
            }
            
            await self._websocket.send(json.dumps(message))
            
            logger.debug(f"[VISUAL] Focus request sent: {request.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"[VISUAL] Failed to send focus request: {e}")
            return False
    
    def focus_region(
        self,
        row: int,
        col: int,
        duration: float = 1.0,
    ) -> FocusRequest:
        """Request focus on a grid region.
        
        Args:
            row: Target row (0-3).
            col: Target column (0-3).
            duration: Focus duration in seconds.
            
        Returns:
            The created focus request.
        """
        return self._attention_manager.create_focus_request(
            row=row,
            col=col,
            reason="manual",
            duration=duration,
        )
    
    def escalate_full(self, duration: float = 2.0) -> FocusRequest:
        """Request full high-res frame.
        
        Args:
            duration: Duration to capture high-res.
            
        Returns:
            The created focus request.
        """
        request = FocusRequest(
            request_id=f"full_{uuid.uuid4().hex[:8]}",
            request_type=FocusRequestType.FULL_FRAME,
            duration_seconds=duration,
            priority=8,  # High priority for full frame
        )
        
        self._attention_manager._pending_requests.append(request)
        
        logger.info("[VISUAL] Full frame escalation requested")
        
        return request
    
    def get_statistics(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "connected": self._connected,
            "url": self.url,
            "summaries_received": self._summaries_received,
            "focus_responses_received": self._focus_responses_received,
            "connection_errors": self._connection_errors,
            "ring_buffer": self._ring_buffer.get_statistics(),
            "feature_extractor": self._feature_extractor.get_statistics(),
            "attention": self._attention_manager.get_statistics(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_visual_client(
    host: str = "localhost",
    port: int = VISUAL_CHANNEL_PORT,
    buffer_size_mb: int = RING_BUFFER_SIZE_MB,
) -> VisualStreamClient:
    """Create a configured visual stream client.
    
    Args:
        host: WebSocket host.
        port: WebSocket port.
        buffer_size_mb: Ring buffer size.
        
    Returns:
        Configured VisualStreamClient.
    """
    return VisualStreamClient(
        host=host,
        port=port,
        buffer_size_mb=buffer_size_mb,
    )
