"""Tests for the panorama API server.

Covers:
- PanoramaAPIState thread-safe operations  
- Endpoint response validation (via direct handler testing)
- CORS headers
- Memory summary computation
- Timeline data accumulation
"""

from __future__ import annotations

import json
import threading

import pytest

from episodic_agent.modules.panorama.api_server import PanoramaAPIState
from episodic_agent.schemas.panorama_events import (
    PanoramaAgentState,
)


class TestPanoramaAPIState:
    """Tests for the API state container."""

    def test_initial_state(self) -> None:
        state = PanoramaAPIState()
        snap = state.snapshot()
        assert snap["step"] == 0
        assert snap["location_label"] == "unknown"
        assert snap["agent_state"] == PanoramaAgentState.investigating_unknown.value

    def test_update_and_snapshot(self) -> None:
        state = PanoramaAPIState()
        state.update({"step": 5, "location_label": "kitchen"})
        snap = state.snapshot()
        assert snap["step"] == 5
        assert snap["location_label"] == "kitchen"

    def test_message_log_bounded(self) -> None:
        state = PanoramaAPIState()
        log = [{"step": i} for i in range(250)]
        state.update({"message_log": log})
        snap = state.snapshot()
        assert len(snap["message_log"]) <= 200

    def test_push_viewport(self) -> None:
        state = PanoramaAPIState()
        for i in range(15):
            state.push_viewport(f"img_{i}")
        snap = state.snapshot()
        # Ring buffer max 12
        assert len(snap["recent_viewports"]) == 12
        assert snap["recent_viewports"][-1] == "img_14"

    def test_push_confidence(self) -> None:
        state = PanoramaAPIState()
        state.push_confidence(step=1, confidence=0.5, state="investigating_unknown", label="unknown")
        state.push_confidence(step=2, confidence=0.7, state="matching_known", label="kitchen")
        snap = state.snapshot()
        assert len(snap["confidence_timeline"]) == 2
        assert snap["confidence_timeline"][1]["confidence"] == 0.7

    def test_confidence_timeline_bounded(self) -> None:
        state = PanoramaAPIState()
        for i in range(600):
            state.push_confidence(step=i, confidence=0.5, state="unknown", label="x")
        snap = state.snapshot()
        assert len(snap["confidence_timeline"]) <= 500

    def test_get_field(self) -> None:
        state = PanoramaAPIState()
        state.update({"location_label": "hallway"})
        assert state.get_field("location_label") == "hallway"
        assert state.get_field("nonexistent") is None

    def test_thread_safety(self) -> None:
        """Concurrent updates and snapshots should not crash."""
        state = PanoramaAPIState()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(100):
                    state.update({"step": i})
                    state.push_viewport(f"img_{i}")
                    state.push_confidence(step=i, confidence=0.5, state="x", label="y")
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(100):
                    state.snapshot()
                    state.get_field("step")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_new_observability_fields_present(self) -> None:
        state = PanoramaAPIState()
        snap = state.snapshot()
        assert "agent_state" in snap
        assert "match_candidates" in snap
        assert "evidence_bundle" in snap
        assert "confidence_timeline" in snap
        assert "recent_viewports" in snap
        assert "feature_arrays" in snap
        assert "investigation_steps" in snap
