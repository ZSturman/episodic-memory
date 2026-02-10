"""Visual summary channel data contracts.

ARCHITECTURAL INVARIANT: Bandwidth-Efficient Sensing
- Visual data is summarized, not raw-streamed
- High-res on demand only
- Raw images not persisted in production mode
- Python extracts features and discards raw images

This module defines the protocol for the visual summary channel (port 8767)
which provides:
1. 4×4 FOV grid summaries (low bandwidth baseline)
2. Focus region requests for high-res crops
3. Fixed ring buffer for recent high-res frames
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Protocol Constants
# =============================================================================

# Visual channel port (separate from main sensor channel)
VISUAL_CHANNEL_PORT: int = 8767

# Grid configuration (4×4 = 16 cells covering FOV)
GRID_ROWS: int = 4
GRID_COLS: int = 4
GRID_CELL_COUNT: int = GRID_ROWS * GRID_COLS

# Ring buffer size (~50MB with typical frames)
RING_BUFFER_SIZE_MB: int = 50
RING_BUFFER_MAX_FRAMES: int = 100  # Limit by count as well

# Histogram configuration
COLOR_HISTOGRAM_BINS: int = 8  # Per channel (8×8×8 = 512 total bins)
EDGE_DIRECTION_BINS: int = 8   # 8 directions (0°, 45°, 90°, ..., 315°)

# Summary frame rate (lower than main sensor rate)
VISUAL_SUMMARY_RATE_HZ: float = 5.0


# =============================================================================
# Grid Cell Schemas
# =============================================================================

class GridCell(BaseModel):
    """Summary of a single cell in the 4×4 FOV grid.
    
    ARCHITECTURAL INVARIANT: Contains only extracted features, never raw pixels.
    This enables bandwidth-efficient sensing suitable for real-world deployment.
    """
    
    row: int = Field(..., ge=0, lt=GRID_ROWS, description="Row index (0-3)")
    col: int = Field(..., ge=0, lt=GRID_COLS, description="Column index (0-3)")
    
    # Color histogram: flattened [R×8, G×8, B×8] = 512 values normalized 0-1
    color_histogram: list[float] = Field(
        default_factory=list,
        description="Normalized RGB color histogram (512 bins)",
    )
    
    # Edge direction histogram: 8 bins for dominant edge orientations
    edge_directions: list[float] = Field(
        default_factory=list,
        description="Normalized edge direction histogram (8 bins)",
    )
    
    # Summary statistics
    mean_brightness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean brightness of cell (0-1)",
    )
    edge_density: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Edge density (edges/total pixels)",
    )
    
    # Motion indicator (change from previous frame)
    motion_magnitude: float = Field(
        default=0.0,
        ge=0.0,
        description="Motion magnitude in this cell",
    )
    
    # Attention weight (computed by Python, used for focus decisions)
    attention_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Computed attention weight for this cell",
    )
    
    def get_cell_index(self) -> int:
        """Get flattened cell index (0-15)."""
        return self.row * GRID_COLS + self.col


class VisualSummaryFrame(BaseModel):
    """Complete 4×4 grid summary for a single frame.
    
    Streamed at ~5Hz on port 8767, providing bandwidth-efficient
    visual context without raw image data.
    """
    
    frame_id: int = Field(..., description="Matches main sensor frame_id")
    timestamp: float = Field(..., description="Unix timestamp")
    
    # Grid cells (always 16 for 4×4)
    cells: list[GridCell] = Field(
        default_factory=list,
        description="16 grid cells covering FOV",
    )
    
    # Global frame statistics
    global_brightness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall frame brightness",
    )
    global_edge_density: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall edge density",
    )
    global_motion: float = Field(
        default=0.0,
        ge=0.0,
        description="Total motion in frame",
    )
    
    # Dominant colors (top 3)
    dominant_colors: list[tuple[int, int, int]] = Field(
        default_factory=list,
        description="Top 3 dominant RGB colors",
    )
    
    # High-res available flag (in ring buffer)
    high_res_available: bool = Field(
        default=False,
        description="Whether high-res is in ring buffer",
    )
    
    def get_cell(self, row: int, col: int) -> GridCell | None:
        """Get cell by row/col position."""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None
    
    def get_most_salient_cell(self) -> GridCell | None:
        """Get the cell with highest attention weight."""
        if not self.cells:
            return None
        return max(self.cells, key=lambda c: c.attention_weight)


# =============================================================================
# Focus Request/Response
# =============================================================================

class FocusRequestType(str, Enum):
    """Types of focus requests."""
    
    REGION = "region"          # Focus on specific grid cell
    FULL_FRAME = "full_frame"  # Request full high-res frame
    TRACK_ENTITY = "track"     # Track entity (auto-focus)


class FocusRequest(BaseModel):
    """Request for high-resolution visual data.
    
    ARCHITECTURAL INVARIANT: High-res on demand only.
    System requests detailed visual data only when needed for
    recognition or disambiguation.
    """
    
    request_id: str = Field(..., description="Unique request identifier")
    request_type: FocusRequestType = Field(..., description="Type of focus request")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    # For REGION requests
    target_row: int | None = Field(
        default=None,
        ge=0,
        lt=GRID_ROWS,
        description="Target row for region focus",
    )
    target_col: int | None = Field(
        default=None,
        ge=0,
        lt=GRID_COLS,
        description="Target column for region focus",
    )
    
    # For FULL_FRAME requests
    frame_id: int | None = Field(
        default=None,
        description="Specific frame to retrieve (from ring buffer)",
    )
    
    # For TRACK requests
    entity_guid: str | None = Field(
        default=None,
        description="Entity GUID to track",
    )
    
    # Duration (how long to maintain focus)
    duration_seconds: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Duration to maintain focus",
    )
    
    # Priority
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Request priority (10=highest)",
    )


class FocusResponse(BaseModel):
    """Response to a focus request.
    
    Contains high-resolution crop or full frame data.
    """
    
    request_id: str = Field(..., description="Matches FocusRequest")
    success: bool = Field(..., description="Whether request was fulfilled")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    # Error info if not successful
    error_message: str | None = Field(default=None)
    
    # Image data (base64 encoded JPEG/PNG)
    image_data: str | None = Field(
        default=None,
        description="Base64 encoded image data",
    )
    image_format: str = Field(
        default="jpeg",
        description="Image format (jpeg/png)",
    )
    image_width: int = Field(default=0, ge=0)
    image_height: int = Field(default=0, ge=0)
    
    # Source frame info
    source_frame_id: int | None = Field(default=None)
    
    # Region info (if crop)
    crop_x: int = Field(default=0, ge=0)
    crop_y: int = Field(default=0, ge=0)
    crop_width: int = Field(default=0, ge=0)
    crop_height: int = Field(default=0, ge=0)


# =============================================================================
# Ring Buffer Entry
# =============================================================================

@dataclass
class RingBufferEntry:
    """Entry in the high-resolution frame ring buffer.
    
    ARCHITECTURAL INVARIANT: Fixed memory budget.
    Buffer auto-evicts oldest frames to stay within ~50MB.
    """
    
    frame_id: int
    timestamp: float
    
    # Image data (raw bytes)
    image_data: bytes
    image_format: str = "jpeg"
    image_width: int = 0
    image_height: int = 0
    
    # Approximate size in bytes
    size_bytes: int = 0
    
    # Associated summary (for quick lookup)
    summary: VisualSummaryFrame | None = None
    
    def __post_init__(self) -> None:
        """Compute size if not set."""
        if self.size_bytes == 0:
            self.size_bytes = len(self.image_data)


# =============================================================================
# Visual Feature Extraction Results
# =============================================================================

@dataclass
class ExtractedVisualFeatures:
    """Features extracted from visual data.
    
    ARCHITECTURAL INVARIANT: Raw images discarded after feature extraction.
    Only these derived features are persisted.
    """
    
    frame_id: int
    timestamp: float
    
    # Per-cell features (flattened)
    cell_color_histograms: list[list[float]] = field(default_factory=list)
    cell_edge_histograms: list[list[float]] = field(default_factory=list)
    cell_brightness: list[float] = field(default_factory=list)
    cell_motion: list[float] = field(default_factory=list)
    
    # Global features
    global_color_histogram: list[float] = field(default_factory=list)
    global_edge_histogram: list[float] = field(default_factory=list)
    dominant_colors: list[tuple[int, int, int]] = field(default_factory=list)
    
    # Scene classification hints (emergent, not predefined)
    scene_signature: str = ""  # Hash of features for quick comparison
    
    def to_embedding(self) -> list[float]:
        """Convert to flat embedding vector for similarity search."""
        embedding = []
        
        # Add global histogram
        embedding.extend(self.global_color_histogram[:64])  # Truncate for size
        embedding.extend(self.global_edge_histogram[:8])
        
        # Add cell brightness pattern
        embedding.extend(self.cell_brightness)
        
        # Pad to fixed size
        target_size = 128
        if len(embedding) < target_size:
            embedding.extend([0.0] * (target_size - len(embedding)))
        
        return embedding[:target_size]


# =============================================================================
# Visual Attention State
# =============================================================================

class AttentionMode(str, Enum):
    """Current attention mode."""
    
    PASSIVE = "passive"      # Just monitoring
    INVESTIGATING = "investigating"  # Focused on anomaly
    TRACKING = "tracking"    # Following entity
    SCANNING = "scanning"    # Systematic sweep


@dataclass
class VisualAttentionState:
    """Current state of visual attention system.
    
    Tracks where attention is focused and why.
    """
    
    mode: AttentionMode = AttentionMode.PASSIVE
    
    # Focus target (if any)
    focus_cell: tuple[int, int] | None = None
    focus_entity_guid: str | None = None
    
    # Attention history (last N focus points)
    attention_history: list[tuple[int, int, float]] = field(default_factory=list)
    
    # Saliency map (16 values for 4×4 grid)
    saliency_map: list[float] = field(default_factory=lambda: [0.0] * GRID_CELL_COUNT)
    
    # Investigation state
    investigating_since: float | None = None
    investigation_reason: str = ""
    
    def update_saliency(self, cell_idx: int, weight: float) -> None:
        """Update saliency for a cell with decay."""
        if 0 <= cell_idx < GRID_CELL_COUNT:
            # Exponential decay on existing + new value
            self.saliency_map[cell_idx] = self.saliency_map[cell_idx] * 0.9 + weight * 0.1
    
    def get_most_salient(self) -> tuple[int, int]:
        """Get row, col of most salient cell."""
        max_idx = max(range(len(self.saliency_map)), key=lambda i: self.saliency_map[i])
        return max_idx // GRID_COLS, max_idx % GRID_COLS


# =============================================================================
# Hex-grid visual data schemas
# =============================================================================

class ReconstructionLayer(str, Enum):
    """Toggleable layers for location reconstruction visualization."""

    COLOR = "color"
    EDGES = "edges"
    TEXTURE = "texture"
    INTEREST = "interest"
    DETAIL_LEVEL = "detail_level"


class HexCellData(BaseModel):
    """API-facing data for a single hex cell."""

    q: int = Field(..., description="Axial coordinate q")
    r: int = Field(..., description="Axial coordinate r")
    center_x: float = Field(0.0, description="Pixel center X")
    center_y: float = Field(0.0, description="Pixel center Y")
    detail_level: int = Field(0, ge=0, le=3)
    weight: float = Field(0.0, ge=0.0, le=1.0)
    interest_score: float = Field(0.0, ge=0.0)

    # Features (populated depending on detail level)
    avg_rgb: list[float] = Field(default_factory=list)
    brightness: float = 0.0
    edge_energy: float = 0.0
    dominant_colors: list[list[float]] = Field(default_factory=list)

    model_config = {"frozen": True}


class HexScanData(BaseModel):
    """API-facing summary of a hex scan result."""

    cells: list[HexCellData] = Field(default_factory=list)
    hex_size: float = 0.0
    image_width: int = 0
    image_height: int = 0
    scan_pass: int = 0
    converged: bool = False
    total_cells: int = 0

    # Focus profile snapshot
    focus_center_q: int = 0
    focus_center_r: int = 0
    focus_fovea_radius: int = 1
    focus_mid_radius: int = 3
    focus_outer_radius: int = 6


class HexReconstructionData(BaseModel):
    """Data needed to reconstruct a location's visual appearance."""

    cells: list[HexCellData] = Field(default_factory=list)
    hex_size: float = 0.0
    image_width: int = 0
    image_height: int = 0
    location_id: str = ""
    parent_label: str = ""
    variant_label: str = ""
    observation_count: int = 0
    layers: list[str] = Field(
        default_factory=lambda: ["color", "edges"],
        description="Available reconstruction layers",
    )
