"""Tests for Phase 5: Visual Summary Channel.

Tests the bandwidth-efficient visual sensing system:
- VisualRingBuffer (fixed-size frame storage)
- VisualFeatureExtractor (histogram/edge extraction)
- VisualAttentionManager (saliency-based attention)
- VisualStreamClient (WebSocket client)
- Visual schemas (GridCell, VisualSummaryFrame, etc.)

ARCHITECTURAL INVARIANT: Bandwidth-Efficient Sensing
- Visual data summarized via 4×4 grid, not raw-streamed
- High-res on demand only
- Raw images discarded after feature extraction
"""

import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

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
)
from episodic_agent.modules.sensor_gateway.visual_client import (
    VisualAttentionManager,
    VisualFeatureExtractor,
    VisualRingBuffer,
    VisualStreamClient,
    create_visual_client,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_grid_cell() -> GridCell:
    """Create a sample grid cell."""
    return GridCell(
        row=1,
        col=2,
        color_histogram=[0.1] * (COLOR_HISTOGRAM_BINS * 3),  # 24 values
        edge_directions=[0.125] * EDGE_DIRECTION_BINS,  # 8 values
        mean_brightness=0.6,
        edge_density=0.4,
        motion_magnitude=0.1,
    )


@pytest.fixture
def sample_visual_summary() -> VisualSummaryFrame:
    """Create a sample visual summary frame."""
    cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cells.append(GridCell(
                row=row,
                col=col,
                color_histogram=[0.1] * (COLOR_HISTOGRAM_BINS * 3),
                edge_directions=[0.125] * EDGE_DIRECTION_BINS,
                mean_brightness=0.5 + row * 0.1,
                edge_density=0.3 + col * 0.05,
                motion_magnitude=0.0,
            ))
    
    return VisualSummaryFrame(
        frame_id=42,
        timestamp=time.time(),
        cells=cells,
        global_brightness=0.55,
        global_edge_density=0.35,
        global_motion=0.0,
        dominant_colors=[(128, 64, 32), (200, 180, 160), (50, 50, 80)],
        high_res_available=True,
    )


@pytest.fixture
def sample_ring_buffer_entry() -> RingBufferEntry:
    """Create a sample ring buffer entry."""
    image_data = b"fake_jpeg_data_" * 1000  # ~15KB
    return RingBufferEntry(
        frame_id=100,
        timestamp=time.time(),
        image_data=image_data,
        image_format="jpeg",
        image_width=640,
        image_height=480,
    )


@pytest.fixture
def ring_buffer() -> VisualRingBuffer:
    """Create a ring buffer with small limits for testing."""
    return VisualRingBuffer(max_size_mb=1, max_frames=10)


@pytest.fixture
def feature_extractor() -> VisualFeatureExtractor:
    """Create a feature extractor."""
    return VisualFeatureExtractor()


@pytest.fixture
def attention_manager() -> VisualAttentionManager:
    """Create an attention manager."""
    return VisualAttentionManager()


# =============================================================================
# Test: GridCell Schema
# =============================================================================

class TestGridCell:
    """Tests for GridCell schema."""
    
    def test_grid_cell_creation(self, sample_grid_cell: GridCell):
        """GridCell should be created with valid values."""
        assert sample_grid_cell.row == 1
        assert sample_grid_cell.col == 2
        assert len(sample_grid_cell.color_histogram) == COLOR_HISTOGRAM_BINS * 3
        assert len(sample_grid_cell.edge_directions) == EDGE_DIRECTION_BINS
    
    def test_grid_cell_index(self, sample_grid_cell: GridCell):
        """get_cell_index should compute flattened index."""
        # row=1, col=2 → 1*4 + 2 = 6
        assert sample_grid_cell.get_cell_index() == 6
    
    def test_grid_cell_validation(self):
        """GridCell should validate row/col bounds."""
        with pytest.raises(ValueError):
            GridCell(row=5, col=0)  # row > 3
        
        with pytest.raises(ValueError):
            GridCell(row=0, col=5)  # col > 3
    
    def test_grid_cell_defaults(self):
        """GridCell should have sensible defaults."""
        cell = GridCell(row=0, col=0)
        
        assert cell.color_histogram == []
        assert cell.edge_directions == []
        assert cell.mean_brightness == 0.0
        assert cell.edge_density == 0.0
        assert cell.attention_weight == 0.0


# =============================================================================
# Test: VisualSummaryFrame Schema
# =============================================================================

class TestVisualSummaryFrame:
    """Tests for VisualSummaryFrame schema."""
    
    def test_summary_creation(self, sample_visual_summary: VisualSummaryFrame):
        """Summary should be created with all cells."""
        assert sample_visual_summary.frame_id == 42
        assert len(sample_visual_summary.cells) == GRID_CELL_COUNT
        assert len(sample_visual_summary.dominant_colors) == 3
    
    def test_get_cell(self, sample_visual_summary: VisualSummaryFrame):
        """get_cell should return correct cell by position."""
        cell = sample_visual_summary.get_cell(2, 3)
        
        assert cell is not None
        assert cell.row == 2
        assert cell.col == 3
    
    def test_get_cell_not_found(self, sample_visual_summary: VisualSummaryFrame):
        """get_cell should return None for invalid position."""
        cell = sample_visual_summary.get_cell(10, 10)
        assert cell is None
    
    def test_get_most_salient_cell(self, sample_visual_summary: VisualSummaryFrame):
        """get_most_salient_cell should return cell with highest attention."""
        # Set one cell with high attention
        sample_visual_summary.cells[5].attention_weight = 0.9
        
        most_salient = sample_visual_summary.get_most_salient_cell()
        
        assert most_salient is not None
        assert most_salient.attention_weight == 0.9


# =============================================================================
# Test: FocusRequest Schema
# =============================================================================

class TestFocusRequest:
    """Tests for FocusRequest schema."""
    
    def test_region_focus_request(self):
        """Region focus request should have row/col."""
        request = FocusRequest(
            request_id="focus_001",
            request_type=FocusRequestType.REGION,
            target_row=2,
            target_col=3,
            duration_seconds=1.5,
        )
        
        assert request.request_type == FocusRequestType.REGION
        assert request.target_row == 2
        assert request.target_col == 3
        assert request.duration_seconds == 1.5
    
    def test_full_frame_request(self):
        """Full frame request should have frame_id."""
        request = FocusRequest(
            request_id="full_001",
            request_type=FocusRequestType.FULL_FRAME,
            frame_id=42,
        )
        
        assert request.request_type == FocusRequestType.FULL_FRAME
        assert request.frame_id == 42
    
    def test_track_entity_request(self):
        """Track entity request should have entity_guid."""
        request = FocusRequest(
            request_id="track_001",
            request_type=FocusRequestType.TRACK_ENTITY,
            entity_guid="ent_lamp_001",
            duration_seconds=5.0,
        )
        
        assert request.request_type == FocusRequestType.TRACK_ENTITY
        assert request.entity_guid == "ent_lamp_001"
    
    def test_focus_request_priority(self):
        """Focus request should respect priority bounds."""
        request = FocusRequest(
            request_id="focus_002",
            request_type=FocusRequestType.REGION,
            target_row=0,
            target_col=0,
            priority=10,  # Max priority
        )
        
        assert request.priority == 10
        
        with pytest.raises(ValueError):
            FocusRequest(
                request_id="focus_003",
                request_type=FocusRequestType.REGION,
                target_row=0,
                target_col=0,
                priority=11,  # Over max
            )


# =============================================================================
# Test: VisualRingBuffer
# =============================================================================

class TestVisualRingBuffer:
    """Tests for VisualRingBuffer."""
    
    def test_buffer_add_and_get(
        self, ring_buffer: VisualRingBuffer, sample_ring_buffer_entry: RingBufferEntry
    ):
        """Buffer should store and retrieve frames."""
        ring_buffer.add(sample_ring_buffer_entry)
        
        retrieved = ring_buffer.get(100)
        
        assert retrieved is not None
        assert retrieved.frame_id == 100
        assert retrieved.image_data == sample_ring_buffer_entry.image_data
    
    def test_buffer_evicts_oldest(self, ring_buffer: VisualRingBuffer):
        """Buffer should evict oldest frames when full."""
        # Add more frames than max_frames
        for i in range(15):
            entry = RingBufferEntry(
                frame_id=i,
                timestamp=time.time(),
                image_data=b"x" * 1000,
            )
            ring_buffer.add(entry)
        
        # Should only have 10 frames (max_frames)
        assert len(ring_buffer) == 10
        
        # Oldest frames should be evicted
        assert ring_buffer.get(0) is None
        assert ring_buffer.get(4) is None
        assert ring_buffer.get(5) is not None  # Frame 5 should still exist
    
    def test_buffer_respects_size_limit(self):
        """Buffer should evict when size limit exceeded."""
        # Very small buffer: 0.01 MB = ~10KB
        small_buffer = VisualRingBuffer(max_size_mb=0.01, max_frames=100)
        
        # Add frames that exceed size limit
        for i in range(5):
            entry = RingBufferEntry(
                frame_id=i,
                timestamp=time.time(),
                image_data=b"x" * 5000,  # 5KB each
            )
            small_buffer.add(entry)
        
        # Should have evicted some frames
        assert len(small_buffer) < 5
    
    def test_buffer_get_latest(self, ring_buffer: VisualRingBuffer):
        """get_latest should return most recent frame."""
        for i in range(5):
            entry = RingBufferEntry(
                frame_id=i,
                timestamp=time.time(),
                image_data=b"x" * 100,
            )
            ring_buffer.add(entry)
        
        latest = ring_buffer.get_latest()
        
        assert latest is not None
        assert latest.frame_id == 4
    
    def test_buffer_has_frame(self, ring_buffer: VisualRingBuffer, sample_ring_buffer_entry):
        """has_frame should check if frame exists."""
        assert not ring_buffer.has_frame(100)
        
        ring_buffer.add(sample_ring_buffer_entry)
        
        assert ring_buffer.has_frame(100)
    
    def test_buffer_clear(self, ring_buffer: VisualRingBuffer, sample_ring_buffer_entry):
        """clear should remove all frames."""
        ring_buffer.add(sample_ring_buffer_entry)
        assert len(ring_buffer) == 1
        
        ring_buffer.clear()
        
        assert len(ring_buffer) == 0
        assert ring_buffer.get(100) is None
    
    def test_buffer_statistics(self, ring_buffer: VisualRingBuffer, sample_ring_buffer_entry):
        """get_statistics should return buffer stats."""
        ring_buffer.add(sample_ring_buffer_entry)
        
        stats = ring_buffer.get_statistics()
        
        assert stats["current_frames"] == 1
        assert stats["frames_added"] == 1
        assert stats["frames_evicted"] == 0
        assert stats["utilization_pct"] > 0


# =============================================================================
# Test: VisualFeatureExtractor
# =============================================================================

class TestVisualFeatureExtractor:
    """Tests for VisualFeatureExtractor."""
    
    def test_extract_from_summary(
        self, feature_extractor: VisualFeatureExtractor, sample_visual_summary: VisualSummaryFrame
    ):
        """Should extract features from summary frame."""
        features = feature_extractor.extract_from_summary(sample_visual_summary)
        
        assert features.frame_id == sample_visual_summary.frame_id
        assert len(features.cell_brightness) == GRID_CELL_COUNT
        assert len(features.cell_color_histograms) == GRID_CELL_COUNT
        assert features.scene_signature != ""
    
    def test_extract_computes_motion(
        self, feature_extractor: VisualFeatureExtractor, sample_visual_summary: VisualSummaryFrame
    ):
        """Should compute motion between consecutive frames."""
        # First frame
        features1 = feature_extractor.extract_from_summary(sample_visual_summary)
        
        # Modify summary for second frame
        sample_visual_summary.frame_id = 43
        for cell in sample_visual_summary.cells:
            cell.mean_brightness += 0.2  # Brightness change
        
        # Second frame
        features2 = feature_extractor.extract_from_summary(sample_visual_summary)
        
        # Should have computed motion
        assert len(features2.cell_motion) == GRID_CELL_COUNT
    
    def test_feature_embedding(self, feature_extractor: VisualFeatureExtractor, sample_visual_summary):
        """to_embedding should return fixed-size vector."""
        features = feature_extractor.extract_from_summary(sample_visual_summary)
        
        embedding = features.to_embedding()
        
        assert len(embedding) == 128  # Fixed target size
        assert all(isinstance(v, float) for v in embedding)
    
    def test_scene_signature_deterministic(
        self, feature_extractor: VisualFeatureExtractor, sample_visual_summary
    ):
        """Scene signature should be deterministic for same input."""
        features1 = feature_extractor.extract_from_summary(sample_visual_summary)
        
        # Reset extractor state
        feature_extractor._prev_summary = None
        
        features2 = feature_extractor.extract_from_summary(sample_visual_summary)
        
        assert features1.scene_signature == features2.scene_signature
    
    def test_extractor_statistics(self, feature_extractor: VisualFeatureExtractor, sample_visual_summary):
        """get_statistics should track processing."""
        feature_extractor.extract_from_summary(sample_visual_summary)
        
        stats = feature_extractor.get_statistics()
        
        assert stats["frames_processed"] == 1


# =============================================================================
# Test: VisualAttentionManager
# =============================================================================

class TestVisualAttentionManager:
    """Tests for VisualAttentionManager."""
    
    def test_update_from_summary(
        self, attention_manager: VisualAttentionManager, sample_visual_summary: VisualSummaryFrame
    ):
        """Should update attention state from summary."""
        # Add motion to one cell
        sample_visual_summary.cells[5].motion_magnitude = 0.8
        
        state = attention_manager.update_from_summary(sample_visual_summary)
        
        assert state.mode == AttentionMode.PASSIVE
        assert len(state.saliency_map) == GRID_CELL_COUNT
        # Cell 5 should have higher saliency
        assert state.saliency_map[5] > 0
    
    def test_should_investigate(
        self, attention_manager: VisualAttentionManager, sample_visual_summary: VisualSummaryFrame
    ):
        """should_investigate should detect salient regions."""
        # No motion → no investigation
        should, row, col = attention_manager.should_investigate(threshold=0.5)
        assert not should
        
        # Add high motion
        sample_visual_summary.cells[10].motion_magnitude = 1.0
        attention_manager.update_from_summary(sample_visual_summary)
        
        should, row, col = attention_manager.should_investigate(threshold=0.01)
        assert should
        assert row is not None
        assert col is not None
    
    def test_create_focus_request(self, attention_manager: VisualAttentionManager):
        """Should create focus requests for cells."""
        request = attention_manager.create_focus_request(
            row=2, col=3, reason="motion", priority=7
        )
        
        assert request.request_type == FocusRequestType.REGION
        assert request.target_row == 2
        assert request.target_col == 3
        assert request.priority == 7
        
        # State should update
        assert attention_manager.state.mode == AttentionMode.INVESTIGATING
        assert attention_manager.state.focus_cell == (2, 3)
    
    def test_create_entity_track_request(self, attention_manager: VisualAttentionManager):
        """Should create track requests for entities."""
        request = attention_manager.create_entity_track_request(
            entity_guid="ent_lamp_001",
            duration=5.0,
        )
        
        assert request.request_type == FocusRequestType.TRACK_ENTITY
        assert request.entity_guid == "ent_lamp_001"
        
        # State should update
        assert attention_manager.state.mode == AttentionMode.TRACKING
        assert attention_manager.state.focus_entity_guid == "ent_lamp_001"
    
    def test_get_pending_requests(self, attention_manager: VisualAttentionManager):
        """get_pending_requests should return and clear queue."""
        attention_manager.create_focus_request(row=0, col=0, reason="test")
        attention_manager.create_focus_request(row=1, col=1, reason="test")
        
        requests = attention_manager.get_pending_requests()
        
        assert len(requests) == 2
        
        # Queue should be cleared
        assert len(attention_manager.get_pending_requests()) == 0
    
    def test_reset_investigation(self, attention_manager: VisualAttentionManager):
        """reset_investigation should clear focus state."""
        attention_manager.create_focus_request(row=1, col=2, reason="test")
        assert attention_manager.state.mode == AttentionMode.INVESTIGATING
        
        attention_manager.reset_investigation()
        
        assert attention_manager.state.mode == AttentionMode.PASSIVE
        assert attention_manager.state.focus_cell is None
    
    def test_attention_history_tracking(
        self, attention_manager: VisualAttentionManager, sample_visual_summary: VisualSummaryFrame
    ):
        """Should track attention history."""
        for i in range(5):
            sample_visual_summary.frame_id = i
            attention_manager.update_from_summary(sample_visual_summary)
        
        assert len(attention_manager.state.attention_history) == 5


# =============================================================================
# Test: VisualStreamClient
# =============================================================================

class TestVisualStreamClient:
    """Tests for VisualStreamClient."""
    
    def test_client_creation(self):
        """Client should be created with defaults."""
        client = VisualStreamClient()
        
        assert client._host == "localhost"
        assert client._port == VISUAL_CHANNEL_PORT
        assert not client.is_connected
    
    def test_client_url(self):
        """Client should generate correct WebSocket URL."""
        client = VisualStreamClient(host="192.168.1.100", port=9000)
        
        assert client.url == "ws://192.168.1.100:9000"
    
    def test_focus_region(self):
        """focus_region should create request."""
        client = VisualStreamClient()
        
        request = client.focus_region(row=1, col=2, duration=2.0)
        
        assert request.request_type == FocusRequestType.REGION
        assert request.target_row == 1
        assert request.target_col == 2
    
    def test_escalate_full(self):
        """escalate_full should create full frame request."""
        client = VisualStreamClient()
        
        request = client.escalate_full(duration=3.0)
        
        assert request.request_type == FocusRequestType.FULL_FRAME
        assert request.duration_seconds == 3.0
        assert request.priority == 8  # High priority
    
    def test_client_statistics(self):
        """get_statistics should return client stats."""
        client = VisualStreamClient()
        
        stats = client.get_statistics()
        
        assert "connected" in stats
        assert "ring_buffer" in stats
        assert "feature_extractor" in stats
        assert "attention" in stats
        assert stats["connected"] is False
    
    def test_create_visual_client_helper(self):
        """Helper should create configured client."""
        client = create_visual_client(
            host="test.local",
            port=9999,
            buffer_size_mb=25,
        )
        
        assert client._host == "test.local"
        assert client._port == 9999


# =============================================================================
# Test: ExtractedVisualFeatures
# =============================================================================

class TestExtractedVisualFeatures:
    """Tests for ExtractedVisualFeatures dataclass."""
    
    def test_features_creation(self):
        """Features should be created with defaults."""
        features = ExtractedVisualFeatures(
            frame_id=1,
            timestamp=time.time(),
        )
        
        assert features.frame_id == 1
        assert features.cell_color_histograms == []
        assert features.scene_signature == ""
    
    def test_features_embedding_size(self):
        """Embedding should have fixed size."""
        features = ExtractedVisualFeatures(
            frame_id=1,
            timestamp=time.time(),
            cell_brightness=[0.5] * GRID_CELL_COUNT,
            global_color_histogram=[0.1] * 64,
            global_edge_histogram=[0.125] * 8,
        )
        
        embedding = features.to_embedding()
        
        assert len(embedding) == 128


# =============================================================================
# Test: Architectural Invariants
# =============================================================================

class TestArchitecturalInvariants:
    """Tests for bandwidth-efficient sensing invariants."""
    
    def test_summary_contains_no_raw_pixels(self, sample_visual_summary: VisualSummaryFrame):
        """Summary should contain features, not raw pixels."""
        # Check that no cell has raw image data
        for cell in sample_visual_summary.cells:
            # Should only have histograms and stats
            assert hasattr(cell, "color_histogram")
            assert hasattr(cell, "edge_directions")
            assert hasattr(cell, "mean_brightness")
            # Should NOT have raw pixel data
            assert not hasattr(cell, "pixels")
            assert not hasattr(cell, "raw_image")
    
    def test_extracted_features_discards_image(
        self, feature_extractor: VisualFeatureExtractor, sample_visual_summary: VisualSummaryFrame
    ):
        """Feature extraction should not store raw images."""
        features = feature_extractor.extract_from_summary(sample_visual_summary)
        
        # Features should not contain raw image data
        assert not hasattr(features, "raw_image")
        assert not hasattr(features, "image_bytes")
        
        # Should only have derived features
        assert hasattr(features, "cell_color_histograms")
        assert hasattr(features, "scene_signature")
    
    def test_ring_buffer_respects_memory_budget(self):
        """Ring buffer should enforce memory limit."""
        # 1MB limit
        buffer = VisualRingBuffer(max_size_mb=1, max_frames=1000)
        
        # Try to add 2MB of data
        for i in range(20):
            entry = RingBufferEntry(
                frame_id=i,
                timestamp=time.time(),
                image_data=b"x" * 100_000,  # 100KB each
            )
            buffer.add(entry)
        
        stats = buffer.get_statistics()
        
        # Should stay under 1MB
        assert stats["current_size_mb"] <= 1.0
    
    def test_focus_is_on_demand(self, attention_manager: VisualAttentionManager):
        """High-res focus should only be requested explicitly."""
        # Initially no pending requests
        requests = attention_manager.get_pending_requests()
        assert len(requests) == 0
        
        # Request must be explicit
        attention_manager.create_focus_request(row=0, col=0, reason="test")
        
        requests = attention_manager.get_pending_requests()
        assert len(requests) == 1


# =============================================================================
# Test: VisualAttentionState
# =============================================================================

class TestVisualAttentionState:
    """Tests for VisualAttentionState."""
    
    def test_state_defaults(self):
        """State should have sensible defaults."""
        state = VisualAttentionState()
        
        assert state.mode == AttentionMode.PASSIVE
        assert state.focus_cell is None
        assert len(state.saliency_map) == GRID_CELL_COUNT
    
    def test_update_saliency(self):
        """update_saliency should accumulate with decay."""
        state = VisualAttentionState()
        
        # Initial saliency
        state.update_saliency(5, 1.0)
        first_value = state.saliency_map[5]
        
        # Second update (should decay + add)
        state.update_saliency(5, 1.0)
        second_value = state.saliency_map[5]
        
        assert second_value > first_value  # Accumulated
        assert second_value < 2.0  # But with decay
    
    def test_get_most_salient(self):
        """get_most_salient should return correct cell."""
        state = VisualAttentionState()
        
        # Set cell 10 (row=2, col=2) as most salient
        state.saliency_map[10] = 1.0
        
        row, col = state.get_most_salient()
        
        assert row == 2
        assert col == 2


# =============================================================================
# Test: Constants
# =============================================================================

class TestConstants:
    """Tests for visual channel constants."""
    
    def test_grid_dimensions(self):
        """Grid should be 4×4."""
        assert GRID_ROWS == 4
        assert GRID_COLS == 4
        assert GRID_CELL_COUNT == 16
    
    def test_visual_port(self):
        """Visual port should be 8767."""
        assert VISUAL_CHANNEL_PORT == 8767
    
    def test_buffer_defaults(self):
        """Buffer defaults should be sensible."""
        assert RING_BUFFER_SIZE_MB == 50
        assert RING_BUFFER_MAX_FRAMES == 100
    
    def test_histogram_bins(self):
        """Histogram bins should be configured."""
        assert COLOR_HISTOGRAM_BINS == 8
        assert EDGE_DIRECTION_BINS == 8
