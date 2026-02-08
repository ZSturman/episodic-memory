"""JSON API server for the panorama observability dashboard.

Serves a structured REST API on a configurable port (default 8780)
that the Next.js dashboard consumes.  Built on stdlib ``http.server``
— no Flask/FastAPI dependency.

Endpoints
---------
GET /api/state          Full agent state snapshot (backward-compatible)
GET /api/events         Event stream (query: since_step)
GET /api/matches        Current ranked match evaluation
GET /api/evidence       Current evidence bundle (during investigation)
GET /api/memories       All location memory summaries
GET /api/memories/:id   Detailed memory card for one location
GET /api/features       Current viewport raw feature arrays
GET /api/timeline       Confidence + state time series
GET /api/config         Current observability config
POST /api/label         Submit a label response from the dashboard

All other paths serve the legacy inline HTML dashboard as fallback.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import numpy as np

from episodic_agent.schemas.panorama_events import (
    MatchCandidate,
    MatchEvaluation,
    MemoryCard,
    MemorySummary,
    PanoramaAgentState,
    PanoramaEventType,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Enriched shared state for the API
# =====================================================================

class PanoramaAPIState:
    """Thread-safe container for the full agent state.

    Extends the original _SharedState with investigation, match,
    memory, and feature data.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            # Legacy fields (backward-compatible with debug_server)
            "step": 0,
            "source_file": "",
            "heading_deg": 0.0,
            "viewport_index": 0,
            "total_viewports": 0,
            "location_label": "unknown",
            "location_confidence": 0.0,
            "episode_count": 0,
            "boundary_triggered": False,
            "hypothesis": {},
            "feature_summary": {},
            "evidence_history": [],
            "message_log": [],
            "viewport_image_b64": None,

            # New observability fields
            "agent_state": PanoramaAgentState.investigating_unknown.value,
            "match_candidates": [],
            "evidence_bundle": None,
            "confidence_timeline": [],
            "recent_viewports": [],
            "feature_arrays": {},
            "investigation_steps": 0,
        }
        # Recent viewports ring buffer (separate for efficiency)
        self._recent_viewports: deque[str] = deque(maxlen=12)
        # Confidence timeline
        self._confidence_timeline: deque[dict[str, Any]] = deque(maxlen=500)

    def update(self, patch: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(patch)
            # Keep message log bounded
            if len(self._data.get("message_log", [])) > 200:
                self._data["message_log"] = self._data["message_log"][-200:]
            if len(self._data.get("evidence_history", [])) > 500:
                self._data["evidence_history"] = self._data["evidence_history"][-500:]

    def push_viewport(self, image_b64: str) -> None:
        """Push a viewport image into the ring buffer."""
        with self._lock:
            self._recent_viewports.append(image_b64)

    def push_confidence(self, step: int, confidence: float, state: str, label: str) -> None:
        """Push a confidence data point for the timeline."""
        with self._lock:
            self._confidence_timeline.append({
                "step": step,
                "confidence": confidence,
                "state": state,
                "label": label,
                "timestamp": datetime.now().isoformat(),
            })

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._data)
            data["recent_viewports"] = list(self._recent_viewports)
            data["confidence_timeline"] = list(self._confidence_timeline)
            return data

    def get_field(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)


# =====================================================================
# Legacy HTML dashboard (kept for backward-compat / fallback)
# =====================================================================

_FALLBACK_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>Panorama API</title>
<style>
body{font-family:system-ui;background:#1e1e2e;color:#cdd6f4;padding:40px;max-width:600px;margin:0 auto}
h1{color:#89b4fa}a{color:#89b4fa}code{background:#313244;padding:2px 6px;border-radius:4px}
.ep{margin:6px 0}
</style></head><body>
<h1>&#9881; Panorama Observability API</h1>
<p>This server provides JSON endpoints for the panorama dashboard.</p>
<h2>Endpoints</h2>
<div class="ep"><code>GET /api/state</code> — Full agent state</div>
<div class="ep"><code>GET /api/events?since_step=N</code> — Event stream</div>
<div class="ep"><code>GET /api/matches</code> — Match candidates</div>
<div class="ep"><code>GET /api/evidence</code> — Evidence bundle</div>
<div class="ep"><code>GET /api/memories</code> — Memory summaries</div>
<div class="ep"><code>GET /api/features</code> — Raw feature arrays</div>
<div class="ep"><code>GET /api/timeline?last=N</code> — Confidence timeline</div>
<div class="ep"><code>GET /api/config</code> — Agent config</div>
<div class="ep"><code>POST /api/label</code> — Submit label</div>
<hr>
<p>Connect the Next.js dashboard at <code>http://localhost:3000</code></p>
</body></html>"""


# =====================================================================
# HTTP Request Handler
# =====================================================================

class _APIHandler(BaseHTTPRequestHandler):
    """Handles API requests for the panorama dashboard."""

    # Injected by PanoramaAPIServer.start()
    _state: PanoramaAPIState
    _event_bus: Any  # PanoramaEventBus (avoid import cycle)
    _location_resolver: Any  # LocationResolverReal
    _perception: Any  # PanoramaPerception
    _config: dict[str, Any]
    _label_callback: Any  # Callable[[str], None] | None
    _replay_controller: Any  # ReplayController | None

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default stderr

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        query = self._parse_query()

        if path == "/api/state":
            self._handle_state()
        elif path == "/api/events":
            self._handle_events(query)
        elif path == "/api/matches":
            self._handle_matches()
        elif path == "/api/evidence":
            self._handle_evidence()
        elif path == "/api/memories":
            self._handle_memories()
        elif path.startswith("/api/memories/"):
            location_id = path.split("/api/memories/")[1]
            self._handle_memory_detail(location_id)
        elif path == "/api/features":
            self._handle_features()
        elif path == "/api/timeline":
            self._handle_timeline(query)
        elif path == "/api/config":
            self._handle_config()
        elif path == "/api/graph/topology":
            self._handle_graph_topology()
        elif path == "/api/similarity/matrix":
            self._handle_similarity_matrix()
        elif path.startswith("/api/memories/") and path.endswith("/variance"):
            lid = path.replace("/api/memories/", "").replace("/variance", "")
            self._handle_embedding_variance(lid)
        elif path.startswith("/api/memories/") and path.endswith("/embedding"):
            lid = path.replace("/api/memories/", "").replace("/embedding", "")
            self._handle_full_embedding(lid)
        elif path == "/api/replay/state":
            self._handle_replay_state()
        else:
            self._send_html(_FALLBACK_HTML)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path == "/api/label":
            self._handle_label_post()
        elif path == "/api/replay/load":
            self._handle_replay_load()
        elif path == "/api/replay/control":
            self._handle_replay_control()
        else:
            self._send_json({"error": "Not found"}, 404)

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _handle_state(self) -> None:
        self._send_json(self._state.snapshot())

    def _handle_events(self, query: dict[str, str]) -> None:
        since_step = int(query.get("since_step", "0"))
        bus = self._event_bus
        if bus:
            events = bus.get_events(since_step=since_step)
            self._send_json({
                "events": [e.model_dump(mode="json") for e in events],
                "total": bus.event_count,
                "latest_step": bus.latest_step,
            })
        else:
            self._send_json({"events": [], "total": 0, "latest_step": 0})

    def _handle_matches(self) -> None:
        state = self._state.snapshot()
        self._send_json({
            "candidates": state.get("match_candidates", []),
            "agent_state": state.get("agent_state", "investigating_unknown"),
            "current_confidence": state.get("location_confidence", 0.0),
            "current_label": state.get("location_label", "unknown"),
        })

    def _handle_evidence(self) -> None:
        state = self._state.snapshot()
        evidence = state.get("evidence_bundle")
        self._send_json({
            "evidence_bundle": evidence,
            "agent_state": state.get("agent_state", "investigating_unknown"),
            "investigation_steps": state.get("investigation_steps", 0),
        })

    def _handle_memories(self) -> None:
        resolver = self._location_resolver
        if not resolver or not hasattr(resolver, "get_all_fingerprints"):
            self._send_json({"memories": [], "count": 0})
            return

        fingerprints = resolver.get_all_fingerprints()
        state = self._state.snapshot()

        # Build a lookup from current match candidates for confidence_vs_current
        candidate_conf: dict[str, float] = {}
        for mc in state.get("match_candidates", []):
            if isinstance(mc, dict):
                candidate_conf[mc.get("location_id", "")] = mc.get("confidence", 0.0)
            elif hasattr(mc, "location_id"):
                candidate_conf[mc.location_id] = mc.confidence

        summaries = []
        for lid, fp in fingerprints.items():
            node = resolver.get_location_node(lid)
            label = node.label if node else f"location_{lid[:8]}"
            emb_norm = math.sqrt(sum(v * v for v in fp.centroid_embedding)) if fp.centroid_embedding else 0.0

            variance = 0.0
            if hasattr(resolver, "get_embedding_variance"):
                variance = resolver.get_embedding_variance(lid)

            match_hist = []
            if hasattr(resolver, "get_match_history"):
                raw_hist = resolver.get_match_history(lid)
                match_hist = [conf for _, conf in raw_hist[-20:]]

            first_step = resolver.get_first_seen_step(lid) if hasattr(resolver, "get_first_seen_step") else 0
            last_step = resolver.get_last_seen_step(lid) if hasattr(resolver, "get_last_seen_step") else 0

            aliases = list(node.labels) if node and node.labels else []

            summaries.append(MemorySummary(
                location_id=lid,
                label=label,
                observation_count=fp.observation_count,
                embedding_centroid_norm=emb_norm,
                variance=variance,
                stability_score=1.0 / (1.0 + variance) if variance > 0 else 1.0,
                first_seen_step=first_step,
                last_seen_step=last_step,
                confidence_vs_current=candidate_conf.get(lid, 0.0),
                match_history=match_hist,
                aliases=aliases,
                entity_cooccurrence=dict(fp.entity_cooccurrence_counts),
            ).model_dump())

        self._send_json({"memories": summaries, "count": len(summaries)})

    def _handle_memory_detail(self, location_id: str) -> None:
        resolver = self._location_resolver
        if not resolver or not hasattr(resolver, "get_location_fingerprint"):
            self._send_json({"error": "not found"}, 404)
            return

        fp = resolver.get_location_fingerprint(location_id)
        if not fp:
            self._send_json({"error": "not found"}, 404)
            return

        node = resolver.get_location_node(location_id)
        label = node.label if node else f"location_{location_id[:8]}"

        variance = 0.0
        if hasattr(resolver, "get_embedding_variance"):
            variance = resolver.get_embedding_variance(location_id)

        match_hist: list[dict[str, Any]] = []
        if hasattr(resolver, "get_match_history"):
            raw = resolver.get_match_history(location_id)
            match_hist = [{"step": s, "confidence": c} for s, c in raw]

        aliases = list(node.labels) if node and node.labels else []

        first_step = resolver.get_first_seen_step(location_id) if hasattr(resolver, "get_first_seen_step") else 0
        last_step = resolver.get_last_seen_step(location_id) if hasattr(resolver, "get_last_seen_step") else 0
        agg_features = resolver.get_aggregated_features(location_id) if hasattr(resolver, "get_aggregated_features") else {}

        card = MemoryCard(
            location_id=location_id,
            label=label,
            embedding_centroid=fp.centroid_embedding[:16],  # truncated for network
            aggregated_features=agg_features,
            variance=variance,
            stability_score=1.0 / (1.0 + variance) if variance > 0 else 1.0,
            observation_count=fp.observation_count,
            first_seen_step=first_step,
            last_seen_step=last_step,
            match_confidence_history=match_hist,
            co_occurring_entities=list(fp.entity_cooccurrence_counts.keys()),
            aliases=aliases,
            transition_positions=[list(p) for p in fp.transition_positions],
        )
        data = card.model_dump()

        # Phase 4 enrichments — additional spatial/entity data
        data["approximate_center"] = list(fp.approximate_center) if hasattr(fp, "approximate_center") and fp.approximate_center else None
        data["approximate_radius"] = fp.approximate_radius if hasattr(fp, "approximate_radius") else None
        data["entity_guids_seen"] = list(fp.entity_guids_seen) if hasattr(fp, "entity_guids_seen") and fp.entity_guids_seen else []
        data["entity_cooccurrence_counts"] = dict(fp.entity_cooccurrence_counts) if fp.entity_cooccurrence_counts else {}
        data["embedding_dimensions"] = len(fp.centroid_embedding) if fp.centroid_embedding else 0
        data["first_visited"] = fp.first_visited.isoformat() if hasattr(fp, "first_visited") and fp.first_visited else None
        data["last_visited"] = fp.last_visited.isoformat() if hasattr(fp, "last_visited") and fp.last_visited else None

        self._send_json(data)

    def _handle_features(self) -> None:
        perception = self._perception
        if perception and hasattr(perception, "get_feature_details"):
            details = perception.get_feature_details()
            self._send_json(details)
        else:
            self._send_json({})

    def _handle_timeline(self, query: dict[str, str]) -> None:
        last_n = int(query.get("last", "100"))
        state = self._state.snapshot()
        timeline = state.get("confidence_timeline", [])
        self._send_json({
            "timeline": timeline[-last_n:],
            "total_steps": state.get("step", 0),
        })

    def _handle_config(self) -> None:
        self._send_json(self._config)

    def _handle_label_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        label = body.get("label", "").strip()
        if not label:
            self._send_json({"error": "label required"}, 400)
            return

        if self._label_callback:
            try:
                self._label_callback(label)
                self._send_json({"status": "ok", "label": label})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "no label handler registered"}, 501)

    # ------------------------------------------------------------------
    # Phase 4: Graph topology
    # ------------------------------------------------------------------

    def _handle_graph_topology(self) -> None:
        """Return location graph as nodes + edges for visualization."""
        resolver = self._location_resolver
        if not resolver or not hasattr(resolver, "get_all_fingerprints"):
            self._send_json({"nodes": [], "edges": []})
            return

        fingerprints = resolver.get_all_fingerprints()
        graph_store = getattr(resolver, "_graph_store", None)

        nodes = []
        for lid, fp in fingerprints.items():
            node = resolver.get_location_node(lid)
            label = node.label if node else f"location_{lid[:8]}"
            activation = node.activation if node else 0.0
            obs_count = fp.observation_count

            # Approximate spatial position (if available)
            center = None
            if hasattr(fp, "approximate_center") and fp.approximate_center:
                center = list(fp.approximate_center)

            nodes.append({
                "location_id": lid,
                "label": label,
                "observation_count": obs_count,
                "activation": activation,
                "position": center,
            })

        edges = []
        if graph_store:
            seen_edges: set[str] = set()
            for lid in fingerprints:
                if hasattr(graph_store, "get_edges"):
                    for edge in graph_store.get_edges(lid):
                        if edge.edge_id not in seen_edges:
                            seen_edges.add(edge.edge_id)
                            # Only include edges between known locations
                            src = edge.source_node_id
                            tgt = edge.target_node_id
                            if src in fingerprints or tgt in fingerprints:
                                edges.append({
                                    "edge_id": edge.edge_id,
                                    "source": src,
                                    "target": tgt,
                                    "edge_type": edge.edge_type,
                                    "weight": edge.weight,
                                    "confidence": edge.confidence,
                                })

        self._send_json({"nodes": nodes, "edges": edges})

    # ------------------------------------------------------------------
    # Phase 4: Similarity matrix
    # ------------------------------------------------------------------

    def _handle_similarity_matrix(self) -> None:
        """Return pairwise cosine similarity matrix across all locations."""
        resolver = self._location_resolver
        if not resolver or not hasattr(resolver, "get_all_fingerprints"):
            self._send_json({"locations": [], "labels": [], "matrix": []})
            return

        fingerprints = resolver.get_all_fingerprints()
        location_ids = list(fingerprints.keys())
        labels = []
        embeddings = []
        for lid in location_ids:
            fp = fingerprints[lid]
            node = resolver.get_location_node(lid)
            labels.append(node.label if node else f"location_{lid[:8]}")
            embeddings.append(fp.centroid_embedding)

        n = len(embeddings)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if embeddings[i] and embeddings[j]:
                    a = np.array(embeddings[i], dtype=np.float64)
                    b = np.array(embeddings[j], dtype=np.float64)
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a > 0 and norm_b > 0:
                        sim = float(np.dot(a, b) / (norm_a * norm_b))
                    else:
                        sim = 0.0
                else:
                    sim = 0.0
                matrix[i][j] = round(sim, 4)
                matrix[j][i] = round(sim, 4)

        self._send_json({
            "locations": location_ids,
            "labels": labels,
            "matrix": matrix,
        })

    # ------------------------------------------------------------------
    # Phase 4: Per-dimension embedding variance
    # ------------------------------------------------------------------

    def _handle_embedding_variance(self, location_id: str) -> None:
        """Return per-dimension variance vector for a location."""
        resolver = self._location_resolver
        if not resolver:
            self._send_json({"error": "not found"}, 404)
            return

        fp = resolver.get_location_fingerprint(location_id) if hasattr(
            resolver, "get_location_fingerprint"
        ) else None
        if not fp:
            self._send_json({"error": "not found"}, 404)
            return

        sum_sq = getattr(resolver, "_embedding_sum_sq", {}).get(location_id)
        if sum_sq and fp.centroid_embedding and fp.observation_count > 1:
            n = fp.observation_count
            centroid = np.array(fp.centroid_embedding, dtype=np.float64)
            sq = np.array(sum_sq, dtype=np.float64)
            var_per_dim = (sq / n) - (centroid ** 2)
            var_per_dim = np.maximum(var_per_dim, 0.0)  # numerical safety
            self._send_json({
                "location_id": location_id,
                "dimensions": len(var_per_dim),
                "variance_per_dim": [round(float(v), 6) for v in var_per_dim],
                "total_variance": round(float(np.sum(var_per_dim)), 6),
                "observation_count": n,
            })
        else:
            dims = len(fp.centroid_embedding) if fp.centroid_embedding else 0
            self._send_json({
                "location_id": location_id,
                "dimensions": dims,
                "variance_per_dim": [0.0] * dims,
                "total_variance": 0.0,
                "observation_count": fp.observation_count,
            })

    # ------------------------------------------------------------------
    # Phase 4: Full (untruncated) embedding
    # ------------------------------------------------------------------

    def _handle_full_embedding(self, location_id: str) -> None:
        """Return the full centroid embedding for a location."""
        resolver = self._location_resolver
        if not resolver:
            self._send_json({"error": "not found"}, 404)
            return

        fp = resolver.get_location_fingerprint(location_id) if hasattr(
            resolver, "get_location_fingerprint"
        ) else None
        if not fp:
            self._send_json({"error": "not found"}, 404)
            return

        emb = fp.centroid_embedding or []
        norm = math.sqrt(sum(v * v for v in emb)) if emb else 0.0
        self._send_json({
            "location_id": location_id,
            "embedding": [round(float(v), 6) for v in emb],
            "dimensions": len(emb),
            "norm": round(norm, 6),
        })

    # ------------------------------------------------------------------
    # Phase 4: Replay
    # ------------------------------------------------------------------

    def _handle_replay_state(self) -> None:
        """GET /api/replay/state — current replay controller state."""
        rc = self._replay_controller
        if rc:
            self._send_json(rc.get_state())
        else:
            self._send_json({
                "loaded": False, "file": "", "total_events": 0,
                "cursor": 0, "playing": False, "speed": 1.0,
            })

    def _handle_replay_load(self) -> None:
        """POST /api/replay/load — load a JSONL file for replay."""
        rc = self._replay_controller
        if not rc:
            self._send_json({"error": "replay not available"}, 501)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        file_path = body.get("file", "").strip()
        if not file_path:
            self._send_json({"error": "file path required"}, 400)
            return

        try:
            count = rc.load(file_path)
            self._send_json({"status": "ok", "events_loaded": count, "file": file_path})
        except FileNotFoundError:
            self._send_json({"error": f"file not found: {file_path}"}, 404)
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_replay_control(self) -> None:
        """POST /api/replay/control — play/pause/seek/step/speed."""
        rc = self._replay_controller
        if not rc:
            self._send_json({"error": "replay not available"}, 501)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode()) if length else {}
        except Exception:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        action = body.get("action", "")
        if action == "play":
            rc.play()
        elif action == "pause":
            rc.pause()
        elif action == "stop":
            rc.stop()
        elif action == "step_forward":
            event = rc.step_forward()
            self._send_json({
                "status": "ok",
                "action": "step_forward",
                **rc.get_state(),
                "event": event.model_dump(mode="json") if event else None,
            })
            return
        elif action == "step_back":
            rc.step_back()
        elif action == "seek":
            position = int(body.get("position", 0))
            rc.seek(position)
        elif action == "set_speed":
            speed = float(body.get("speed", 1.0))
            rc.set_speed(speed)
        else:
            self._send_json({"error": f"unknown action: {action}"}, 400)
            return

        self._send_json({"status": "ok", "action": action, **rc.get_state()})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_query(self) -> dict[str, str]:
        """Parse query string from self.path."""
        result: dict[str, str] = {}
        if "?" in self.path:
            qs = self.path.split("?", 1)[1]
            for pair in qs.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    result[k] = v
        return result


# =====================================================================
# Server wrapper
# =====================================================================

class PanoramaAPIServer:
    """Manages the HTTP API server in a daemon thread.

    Parameters
    ----------
    port : int
        Port to listen on (default 8780).
    event_bus : PanoramaEventBus, optional
        Event bus for streaming events.
    location_resolver : LocationResolverReal, optional
        For memory introspection endpoints.
    perception : PanoramaPerception, optional
        For feature detail endpoints.
    config : dict, optional
        Agent configuration for the /api/config endpoint.
    label_callback : callable, optional
        Called with (label: str) when a label is submitted via POST.
    """

    def __init__(
        self,
        port: int = 8780,
        event_bus: Any = None,
        location_resolver: Any = None,
        perception: Any = None,
        config: dict[str, Any] | None = None,
        label_callback: Any = None,
        replay_controller: Any = None,
    ) -> None:
        self.port = port
        self.state = PanoramaAPIState()
        self._event_bus = event_bus
        self._location_resolver = location_resolver
        self._perception = perception
        self._config = config or {}
        self._label_callback = label_callback
        self._replay_controller = replay_controller
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Inject dependencies into handler class
        _APIHandler._state = self.state
        _APIHandler._event_bus = self._event_bus
        _APIHandler._location_resolver = self._location_resolver
        _APIHandler._perception = self._perception
        _APIHandler._config = self._config
        _APIHandler._label_callback = self._label_callback
        _APIHandler._replay_controller = self._replay_controller

        self._server = HTTPServer(("", self.port), _APIHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="panorama-api-server",
        )
        self._thread.start()
        logger.info(f"Panorama API server started on http://localhost:{self.port}")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        logger.info("Panorama API server stopped")

    def update_state(
        self,
        result: Any,
        sensor: Any | None = None,
        perception: Any | None = None,
        investigation_sm: Any | None = None,
        match_candidates: list[Any] | None = None,
    ) -> None:
        """Push the latest agent state into the shared buffer.

        Parameters
        ----------
        result : StepResult
            Output from orchestrator.step().
        sensor, perception : optional
            Module refs for extra data.
        investigation_sm : optional
            Investigation state machine.
        match_candidates : optional
            Pre-computed match candidates (avoids duplicate computation).
        """
        patch: dict[str, Any] = {
            "step": getattr(result, "step_number", 0),
            "location_label": getattr(result, "location_label", "unknown"),
            "location_confidence": getattr(result, "location_confidence", 0.0),
            "episode_count": getattr(result, "episode_count", 0),
            "boundary_triggered": getattr(result, "boundary_triggered", False),
        }

        # Extras from step result
        extras = getattr(result, "extras", {}) or {}
        patch["source_file"] = extras.get("source_file", "")
        patch["heading_deg"] = extras.get("heading_deg", 0.0)
        patch["viewport_index"] = extras.get("viewport_index", 0)
        patch["total_viewports"] = extras.get("total_viewports", 0)
        patch["feature_summary"] = extras.get("feature_summary", {})
        patch["hypothesis"] = extras.get("hypothesis", {})

        # Viewport image
        viewport_b64 = extras.get("viewport_image_b64")
        if not viewport_b64:
            raw = getattr(result, "raw_data", None)
            if isinstance(raw, dict):
                viewport_b64 = raw.get("image_bytes_b64")
        patch["viewport_image_b64"] = viewport_b64
        if viewport_b64:
            self.state.push_viewport(viewport_b64)

        # Investigation state machine data
        if investigation_sm:
            patch["agent_state"] = investigation_sm.state.value
            patch["investigation_steps"] = investigation_sm.investigation_steps
            if investigation_sm.should_request_label():
                bundle = investigation_sm.get_evidence_bundle()
                patch["evidence_bundle"] = bundle.model_dump(mode="json")
            else:
                patch["evidence_bundle"] = None

        # Match candidates — use pre-computed if available, else compute
        if match_candidates is not None:
            patch["match_candidates"] = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in match_candidates
            ]
        else:
            resolver = self._location_resolver
            if resolver and hasattr(resolver, "get_all_match_scores"):
                scene_emb = extras.get("viewport_embedding") or extras.get("panoramic_embedding")
                if scene_emb:
                    candidates = resolver.get_all_match_scores(scene_emb)
                    patch["match_candidates"] = [c.model_dump() for c in candidates]

        # Feature arrays from perception
        if perception and hasattr(perception, "get_feature_details"):
            patch["feature_arrays"] = perception.get_feature_details()

        # Update aggregated features for current location
        resolver = self._location_resolver
        if resolver and hasattr(resolver, "update_aggregated_features"):
            current_lid = getattr(resolver, "_current_location_id", None)
            feature_summary = patch.get("feature_summary")
            if current_lid and feature_summary:
                resolver.update_aggregated_features(current_lid, feature_summary)

        # Compact message log entry
        msg_entry = {
            "step": patch["step"],
            "source": patch["source_file"],
            "heading": patch["heading_deg"],
            "location": patch["location_label"],
            "conf": round(patch["location_confidence"], 3),
            "state": patch.get("agent_state", "unknown"),
            "sig": patch.get("feature_summary", {}).get("scene_signature", ""),
        }
        current = self.state.snapshot()
        log = current.get("message_log", [])
        log.append(msg_entry)
        patch["message_log"] = log

        self.state.update(patch)

        # Push confidence timeline point
        self.state.push_confidence(
            step=patch["step"],
            confidence=patch["location_confidence"],
            state=patch.get("agent_state", "investigating_unknown"),
            label=patch["location_label"],
        )

    def set_location_resolver(self, resolver: Any) -> None:
        """Update the resolver reference (for lazy init)."""
        self._location_resolver = resolver
        _APIHandler._location_resolver = resolver

    def set_perception(self, perception: Any) -> None:
        """Update the perception reference (for lazy init)."""
        self._perception = perception
        _APIHandler._perception = perception

    def set_event_bus(self, event_bus: Any) -> None:
        """Update the event bus reference (for lazy init)."""
        self._event_bus = event_bus
        _APIHandler._event_bus = event_bus
