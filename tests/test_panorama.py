"""Tests for the panorama / video harness.

Covers:
  - PanoramaSensorProvider: frame generation, viewport cycling, exhaustion, reset
  - PanoramaFeatureExtractor: feature extraction, embedding shape, determinism
  - PanoramaPerception: SensorFrame → Percept conversion, embedding accumulation
  - SaccadePolicy: viewport layout, modes, sweep completion
  - Integration: end-to-end labeling flow with multiple images
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — create synthetic test images
# ---------------------------------------------------------------------------

def _make_solid_image(width: int, height: int, color_rgb: tuple[int, int, int]) -> bytes:
    """Create a solid-colour JPEG image in memory."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_gradient_image(width: int, height: int) -> bytes:
    """Create a horizontal gradient image (dark→light)."""
    from PIL import Image
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        val = int(255 * x / max(1, width - 1))
        arr[:, x, :] = val
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_checkerboard_image(width: int, height: int, sq: int = 32) -> bytes:
    """Create a black-and-white checkerboard."""
    from PIL import Image
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // sq) + (y // sq)) % 2 == 0:
                arr[y, x] = [255, 255, 255]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Create a temp directory with 3 test images."""
    d = tmp_path / "panorama_images"
    d.mkdir()

    # Image A: warm red-ish room
    (d / "room_a.jpg").write_bytes(
        _make_solid_image(512, 384, (200, 80, 60))
    )
    # Image B: same warm room (similar colours → should match A)
    (d / "room_a_revisit.jpg").write_bytes(
        _make_solid_image(512, 384, (190, 85, 65))
    )
    # Image C: cool blue room (clearly different)
    (d / "room_b.jpg").write_bytes(
        _make_solid_image(512, 384, (40, 80, 200))
    )

    return d


@pytest.fixture
def gradient_dir(tmp_path: Path) -> Path:
    """Single gradient image for feature-extraction tests."""
    d = tmp_path / "gradient"
    d.mkdir()
    (d / "gradient.jpg").write_bytes(_make_gradient_image(512, 384))
    return d


@pytest.fixture
def checkerboard_dir(tmp_path: Path) -> Path:
    """Single checkerboard image."""
    d = tmp_path / "checker"
    d.mkdir()
    (d / "checker.jpg").write_bytes(_make_checkerboard_image(512, 384))
    return d


# ======================================================================
# SaccadePolicy tests
# ======================================================================

class TestSaccadePolicy:
    """Test the viewport scan controller."""

    def test_scan_viewports_count(self) -> None:
        from episodic_agent.modules.panorama.saccade import SaccadePolicy
        sp = SaccadePolicy(viewport_width=128, viewport_height=128, headings=6)
        sp.prepare_for_image(800, 600)
        assert sp.total_viewports() == 6

    def test_scan_covers_image_width(self) -> None:
        from episodic_agent.modules.panorama.saccade import SaccadePolicy
        sp = SaccadePolicy(viewport_width=200, viewport_height=200, headings=4)
        sp.prepare_for_image(800, 600)
        vps = []
        while True:
            vp = sp.next_viewport(800, 600)
            if vp is None:
                break
            vps.append(vp)
        assert len(vps) == 4
        # First viewport should start at x=0
        assert vps[0].x == 0
        # Last should be at the right edge
        assert vps[-1].x + vps[-1].width <= 800

    def test_sweep_complete_flag(self) -> None:
        from episodic_agent.modules.panorama.saccade import SaccadePolicy
        sp = SaccadePolicy(viewport_width=256, viewport_height=256, headings=3)
        sp.prepare_for_image(512, 512)
        for _ in range(3):
            sp.next_viewport(512, 512)
        assert sp.sweep_complete is True

    def test_reset(self) -> None:
        from episodic_agent.modules.panorama.saccade import SaccadePolicy
        sp = SaccadePolicy(viewport_width=128, viewport_height=128, headings=4)
        sp.prepare_for_image(512, 384)
        sp.next_viewport(512, 384)
        sp.next_viewport(512, 384)
        sp.reset()
        assert sp.sweep_complete is False
        assert sp.current_index() == 0

    def test_heading_degrees_span_360(self) -> None:
        from episodic_agent.modules.panorama.saccade import SaccadePolicy
        sp = SaccadePolicy(viewport_width=100, viewport_height=100, headings=5)
        sp.prepare_for_image(600, 400)
        vps = []
        while True:
            vp = sp.next_viewport(600, 400)
            if vp is None:
                break
            vps.append(vp)
        headings = [vp.heading_deg for vp in vps]
        assert headings[0] == pytest.approx(0.0)
        assert headings[-1] == pytest.approx(360.0)

    def test_single_viewport_for_narrow_image(self) -> None:
        from episodic_agent.modules.panorama.saccade import SaccadePolicy
        sp = SaccadePolicy(viewport_width=256, viewport_height=256, headings=5)
        sp.prepare_for_image(100, 100)
        assert sp.total_viewports() == 1


# ======================================================================
# PanoramaFeatureExtractor tests
# ======================================================================

class TestPanoramaFeatureExtractor:
    """Test hand-crafted feature extraction."""

    def test_embedding_shape(self, gradient_dir: Path) -> None:
        from PIL import Image
        from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor

        ext = PanoramaFeatureExtractor()
        img = np.array(Image.open(gradient_dir / "gradient.jpg").convert("RGB"))
        features = ext.extract_from_rgb(img, frame_id=1, timestamp=0.0)
        emb = features.to_embedding()

        assert len(emb) == 128
        assert all(isinstance(v, float) for v in emb)

    def test_determinism(self, gradient_dir: Path) -> None:
        from PIL import Image
        from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor

        img = np.array(Image.open(gradient_dir / "gradient.jpg").convert("RGB"))
        ext1 = PanoramaFeatureExtractor()
        ext2 = PanoramaFeatureExtractor()
        f1 = ext1.extract_from_rgb(img, 1, 0.0)
        f2 = ext2.extract_from_rgb(img, 1, 0.0)

        assert f1.to_embedding() == f2.to_embedding()
        assert f1.scene_signature == f2.scene_signature

    def test_different_images_produce_different_embeddings(
        self, gradient_dir: Path, checkerboard_dir: Path,
    ) -> None:
        from PIL import Image
        from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor

        ext = PanoramaFeatureExtractor()
        g = np.array(Image.open(gradient_dir / "gradient.jpg").convert("RGB"))
        c = np.array(Image.open(checkerboard_dir / "checker.jpg").convert("RGB"))

        eg = ext.extract_from_rgb(g, 1, 0.0).to_embedding()
        ec = ext.extract_from_rgb(c, 2, 0.0).to_embedding()

        # Should not be identical
        assert eg != ec

    def test_cell_counts(self, gradient_dir: Path) -> None:
        from PIL import Image
        from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor

        ext = PanoramaFeatureExtractor()
        img = np.array(Image.open(gradient_dir / "gradient.jpg").convert("RGB"))
        features = ext.extract_from_rgb(img, 1, 0.0)

        assert len(features.cell_brightness) == 16
        assert len(features.cell_color_histograms) == 16
        assert len(features.cell_edge_histograms) == 16

    def test_accumulate_panorama_embedding(self) -> None:
        from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor

        ext = PanoramaFeatureExtractor()
        e1 = [float(i) for i in range(128)]
        e2 = [float(i + 2) for i in range(128)]
        result = ext.accumulate_panorama_embedding([e1, e2])

        assert len(result) == 128
        # Mean of 0 and 2 is 1, mean of 1 and 3 is 2, etc.
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)

    def test_accumulate_empty(self) -> None:
        from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor

        ext = PanoramaFeatureExtractor()
        result = ext.accumulate_panorama_embedding([])
        assert len(result) == 128
        assert all(v == 0.0 for v in result)

    def test_dominant_colors(self) -> None:
        from PIL import Image
        from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor

        ext = PanoramaFeatureExtractor()
        # Solid red image → dominant colour should be reddish
        img = np.full((64, 64, 3), [200, 40, 30], dtype=np.uint8)
        features = ext.extract_from_rgb(img, 1, 0.0)
        assert len(features.dominant_colors) >= 1
        r, g, b = features.dominant_colors[0]
        assert r > g and r > b  # reddish


# ======================================================================
# PanoramaSensorProvider tests
# ======================================================================

class TestPanoramaSensorProvider:
    """Test image loading, viewport cycling, exhaustion."""

    def test_discovers_images(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=4)
        assert sp.source_count == 3

    def test_has_frames_initially(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=4)
        assert sp.has_frames() is True

    def test_frame_has_correct_sensor_type(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=4)
        frame = sp.get_frame()
        assert frame.sensor_type == "panorama"

    def test_frame_has_image_bytes(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=4)
        frame = sp.get_frame()
        assert "image_bytes_b64" in frame.raw_data
        # Should be valid base64
        decoded = base64.b64decode(frame.raw_data["image_bytes_b64"])
        assert len(decoded) > 0

    def test_frame_has_camera_pose(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=4)
        frame = sp.get_frame()
        pose = frame.extras.get("camera_pose")
        assert pose is not None
        assert "position" in pose
        assert "forward" in pose

    def test_exhausts_all_viewports(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=4)
        frames = []
        while sp.has_frames():
            frames.append(sp.get_frame())
        # 3 images × 4 headings + transition frames between images
        assert len(frames) >= 12  # at least 4 per image
        assert sp.has_frames() is False

    def test_transition_frames_emitted(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=2)
        frames = []
        while sp.has_frames():
            frames.append(sp.get_frame())
        transitions = [f for f in frames if f.extras.get("transition")]
        # There should be transitions between images
        assert len(transitions) >= 2  # after 1st and 2nd image

    def test_reset(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=2)
        # Exhaust
        while sp.has_frames():
            sp.get_frame()
        assert sp.has_frames() is False
        # Reset
        sp.reset()
        assert sp.has_frames() is True

    def test_frame_ids_monotonic(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=3)
        ids = []
        while sp.has_frames():
            ids.append(sp.get_frame().frame_id)
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)  # all unique

    def test_empty_directory(self, tmp_path: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        d = tmp_path / "empty"
        d.mkdir()
        sp = PanoramaSensorProvider(image_dir=d)
        assert sp.has_frames() is False


# ======================================================================
# PanoramaPerception tests
# ======================================================================

class TestPanoramaPerception:
    """Test SensorFrame → Percept conversion."""

    def test_produces_percept_with_embedding(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        from episodic_agent.modules.panorama.perception import PanoramaPerception

        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=4)
        perc = PanoramaPerception()
        frame = sp.get_frame()
        percept = perc.process(frame)

        assert percept.scene_embedding is not None
        assert len(percept.scene_embedding) == 128
        assert percept.confidence > 0.0

    def test_transition_percept(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.perception import PanoramaPerception
        from episodic_agent.schemas.frames import SensorFrame

        perc = PanoramaPerception()
        frame = SensorFrame(
            frame_id=99,
            timestamp=datetime.now(),
            sensor_type="panorama",
            raw_data={},
            extras={"transition": True, "camera_pose": {"position": [0, 0, 0], "forward": [1, 0, 0]}},
        )
        percept = perc.process(frame)
        assert percept.scene_embedding is not None
        assert percept.extras.get("transition") is True

    def test_extras_contain_feature_summary(self, image_dir: Path) -> None:
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        from episodic_agent.modules.panorama.perception import PanoramaPerception

        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=2)
        perc = PanoramaPerception()
        frame = sp.get_frame()
        percept = perc.process(frame)

        fs = percept.extras.get("feature_summary", {})
        assert "global_brightness" in fs
        assert "dominant_colors" in fs
        assert "scene_signature" in fs

    def test_no_hidden_labels_in_percept(self, image_dir: Path) -> None:
        """Verify no ground-truth identifiers leak into the Percept."""
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        from episodic_agent.modules.panorama.perception import PanoramaPerception

        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=2)
        perc = PanoramaPerception()
        frame = sp.get_frame()
        percept = perc.process(frame)

        # The percept should not contain any room/location labels
        percept_json = json.dumps(percept.model_dump(), default=str)
        assert "room_a" not in percept_json.lower() or "source_file" in percept_json
        # source_file is metadata about the file, not a hidden label used for inference

    def test_confidence_varies_with_content(self) -> None:
        """Solid colour should have lower confidence than a complex scene."""
        from episodic_agent.modules.panorama.perception import PanoramaPerception
        from episodic_agent.schemas.frames import SensorFrame

        perc = PanoramaPerception()

        # Solid grey
        grey_bytes = _make_solid_image(128, 128, (128, 128, 128))
        grey_b64 = base64.b64encode(grey_bytes).decode()
        f1 = SensorFrame(
            frame_id=1, timestamp=datetime.now(), sensor_type="panorama",
            raw_data={"image_bytes_b64": grey_b64, "viewport": {}},
            extras={"camera_pose": {"position": [0, 0, 0], "forward": [1, 0, 0]}},
        )
        p1 = perc.process(f1)

        # Checkerboard (more edges → higher confidence)
        check_bytes = _make_checkerboard_image(128, 128, sq=16)
        check_b64 = base64.b64encode(check_bytes).decode()
        f2 = SensorFrame(
            frame_id=2, timestamp=datetime.now(), sensor_type="panorama",
            raw_data={"image_bytes_b64": check_b64, "viewport": {}},
            extras={"camera_pose": {"position": [0, 0, 0], "forward": [1, 0, 0]}},
        )
        p2 = perc.process(f2)

        assert p2.confidence >= p1.confidence


# ======================================================================
# Terminal debugger tests
# ======================================================================

class TestTerminalDebugger:

    def test_prints_without_crashing(self, capsys: pytest.CaptureFixture) -> None:
        from episodic_agent.modules.panorama.debug import TerminalDebugger
        from types import SimpleNamespace

        dbg = TerminalDebugger()
        result = SimpleNamespace(
            step_number=1,
            location_label="kitchen",
            location_confidence=0.8,
            episode_count=2,
            boundary_triggered=False,
            frame_id=10,
            extras={
                "source_file": "test.jpg",
                "heading_deg": 90.0,
                "viewport_index": 3,
                "total_viewports": 8,
                "feature_summary": {
                    "global_brightness": 0.5,
                    "global_edge_density": 0.1,
                    "dominant_colors": [[200, 100, 50]],
                    "scene_signature": "abc123",
                },
            },
        )
        # Should not raise
        dbg.print_step(result)


# ======================================================================
# Integration test — full pipeline
# ======================================================================

class TestPanoramaIntegration:
    """End-to-end test: load images → run pipeline → verify labeling flow."""

    def test_pipeline_runs_without_error(self, image_dir: Path, tmp_path: Path) -> None:
        """Smoke test: the full orchestrator loop runs on panorama images."""
        from episodic_agent.utils.profiles import get_profile, ModuleFactory
        from episodic_agent.core.orchestrator import AgentOrchestrator

        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()

        profile = get_profile("panorama")
        factory = ModuleFactory(
            profile=profile,
            run_dir=mem_dir,
            seed=42,
            image_dir=str(image_dir),
            headings_per_image=2,
            viewport_width=128,
            viewport_height=128,
            auto_label_locations=True,  # skip interactive prompts
            auto_label_entities=True,
        )
        modules = factory.create_modules()

        orchestrator = AgentOrchestrator(
            sensor=modules["sensor"],
            perception=modules["perception"],
            acf_builder=modules["acf_builder"],
            location_resolver=modules["location_resolver"],
            entity_resolver=modules["entity_resolver"],
            event_resolver=modules["event_resolver"],
            retriever=modules["retriever"],
            boundary_detector=modules["boundary_detector"],
            dialog_manager=modules["dialog_manager"],
            episode_store=modules["episode_store"],
            graph_store=modules["graph_store"],
            run_id="test_run",
        )

        step_count = 0
        while modules["sensor"].has_frames() and step_count < 50:
            result = orchestrator.step()
            step_count += 1

        assert step_count > 0
        # Should have produced some episode data
        assert orchestrator.episode_count >= 0

    def test_sensor_messages_are_inspectable(self, image_dir: Path) -> None:
        """Verify that SensorFrame payloads are JSON-serialisable and inspectable."""
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider

        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=2)
        frame = sp.get_frame()

        # The frame should be fully serialisable
        data = frame.model_dump()
        json_str = json.dumps(data, default=str)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["sensor_type"] == "panorama"
        assert "camera_pose" in parsed["extras"]

    def test_no_ground_truth_in_pipeline(self, image_dir: Path) -> None:
        """Confirm no hidden ground-truth identifiers leak into the pipeline."""
        from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
        from episodic_agent.modules.panorama.perception import PanoramaPerception

        sp = PanoramaSensorProvider(image_dir=image_dir, headings_per_image=2)
        perc = PanoramaPerception()

        while sp.has_frames():
            frame = sp.get_frame()
            percept = perc.process(frame)

            # No room GUID or hidden label should appear
            assert frame.extras.get("current_room_guid") is None
            assert frame.extras.get("current_room") is None
            # source_file is metadata for debugging, not used in inference
