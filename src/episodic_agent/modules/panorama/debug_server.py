"""Web-based debug UI for the panorama harness.

Serves a single-page HTML dashboard on a configurable port (default 8780)
using only stdlib ``http.server`` + threading — no Flask/FastAPI dependency.

Dashboard panels
----------------
- **Panorama view**: Current image with viewport rectangle overlay
- **Evidence chart**: Bar chart of accumulated features per heading
- **Hypothesis tracker**: Confidence over time, label candidates
- **Message log**: Scrollable list of SensorMessage payloads (JSON)

The dashboard polls ``/api/state`` every 500 ms to refresh.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

logger = logging.getLogger(__name__)


# =====================================================================
# In-memory state shared between agent thread and HTTP thread
# =====================================================================

class _SharedState:
    """Thread-safe container for the most recent agent state snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
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
            "viewport_image_b64": None,   # JPEG of current viewport
        }

    def update(self, patch: dict[str, Any]) -> None:
        with self._lock:
            self._data.update(patch)
            # Keep message log bounded
            if len(self._data.get("message_log", [])) > 200:
                self._data["message_log"] = self._data["message_log"][-200:]
            if len(self._data.get("evidence_history", [])) > 500:
                self._data["evidence_history"] = self._data["evidence_history"][-500:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


_state = _SharedState()


# =====================================================================
# HTTP request handler
# =====================================================================

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>Panorama Debug</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#1e1e2e;color:#cdd6f4;padding:12px}
h1{font-size:1.2rem;margin-bottom:8px;color:#89b4fa}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.panel{background:#313244;border-radius:8px;padding:12px;overflow:auto}
.panel h2{font-size:.95rem;color:#a6adc8;margin-bottom:6px;border-bottom:1px solid #45475a;padding-bottom:4px}
#viewport-img{max-width:100%;border-radius:4px;border:2px solid #45475a}
.bar{height:14px;background:#89b4fa;border-radius:3px;margin:2px 0;transition:width .3s}
.bar-bg{background:#45475a;border-radius:3px;width:100%;height:14px}
.label{font-size:.8rem;color:#a6adc8}
.conf{font-size:1.5rem;font-weight:700;color:#a6e3a1}
.msg{font-family:monospace;font-size:.72rem;white-space:pre-wrap;background:#1e1e2e;border-radius:4px;padding:6px;margin:3px 0;max-height:120px;overflow-y:auto;color:#94e2d5}
.progress{font-size:.8rem;color:#f9e2af;margin-bottom:6px}
#log-container{max-height:400px;overflow-y:auto}
</style></head><body>
<h1>&#128270; Panorama Debug Dashboard</h1>
<div class="grid">
  <div class="panel" id="p-view">
    <h2>Viewport</h2>
    <div class="progress" id="progress"></div>
    <img id="viewport-img" alt="viewport" src="">
    <div class="label" id="source-label"></div>
  </div>
  <div class="panel" id="p-hyp">
    <h2>Hypothesis</h2>
    <div id="hyp-label" style="font-size:1.1rem;color:#f5c2e7"></div>
    <div class="conf" id="hyp-conf">0%</div>
    <h2 style="margin-top:10px">Evidence</h2>
    <div id="evidence"></div>
  </div>
  <div class="panel" style="grid-column:1/3" id="p-log">
    <h2>Sensor Messages <span id="step-label" style="float:right;color:#f9e2af"></span></h2>
    <div id="log-container"></div>
  </div>
</div>
<script>
function poll(){
  fetch('/api/state').then(r=>r.json()).then(d=>{
    // Viewport image
    const img=document.getElementById('viewport-img');
    if(d.viewport_image_b64) img.src='data:image/jpeg;base64,'+d.viewport_image_b64;
    document.getElementById('source-label').textContent=d.source_file||'';
    const pct=d.total_viewports>0?Math.round(d.viewport_index/d.total_viewports*100):0;
    document.getElementById('progress').textContent=
      `Step ${d.step} | heading ${(d.heading_deg||0).toFixed(0)}° | viewport ${d.viewport_index}/${d.total_viewports} (${pct}%)`;
    // Hypothesis
    const h=d.hypothesis||{};
    document.getElementById('hyp-label').textContent=
      `📍 ${d.location_label||'unknown'}`;
    document.getElementById('hyp-conf').textContent=
      `${(d.location_confidence*100).toFixed(0)}%`;
    // Evidence
    const ev=d.feature_summary||{};
    let evHtml='';
    if(ev.global_brightness!==undefined){
      const w=Math.round(ev.global_brightness*100);
      evHtml+=`<div class="label">Brightness</div><div class="bar-bg"><div class="bar" style="width:${w}%"></div></div>`;
    }
    if(ev.global_edge_density!==undefined){
      const w=Math.min(100,Math.round(ev.global_edge_density*800));
      evHtml+=`<div class="label">Edge density</div><div class="bar-bg"><div class="bar" style="width:${w}%;background:#f38ba8"></div></div>`;
    }
    if(ev.dominant_colors){
      evHtml+='<div class="label">Dominant colours</div><div style="display:flex;gap:4px;margin-top:2px">';
      ev.dominant_colors.forEach(c=>{evHtml+=`<div style="width:28px;height:28px;border-radius:4px;background:rgb(${c[0]},${c[1]},${c[2]})"></div>`;});
      evHtml+='</div>';
    }
    if(ev.scene_signature) evHtml+=`<div class="label" style="margin-top:4px">sig: ${ev.scene_signature}</div>`;
    document.getElementById('evidence').innerHTML=evHtml;
    // Episode count & boundary
    let step_info = `step ${d.step} | episodes ${d.episode_count}`;
    if(d.boundary_triggered) step_info += ' | BOUNDARY';
    document.getElementById('step-label').textContent=step_info;
    // Message log (most recent on top)
    const logs=d.message_log||[];
    const lc=document.getElementById('log-container');
    let logHtml='';
    for(let i=logs.length-1;i>=Math.max(0,logs.length-30);i--){
      logHtml+=`<div class="msg">${typeof logs[i]==='string'?logs[i]:JSON.stringify(logs[i],null,1)}</div>`;
    }
    lc.innerHTML=logHtml;
  }).catch(()=>{});
}
setInterval(poll,500);
poll();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML and the /api/state JSON endpoint."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default stderr logging
        pass

    def do_GET(self) -> None:
        if self.path == "/api/state":
            data = _state.snapshot()
            body = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_DASHBOARD_HTML.encode())


# =====================================================================
# Server wrapper
# =====================================================================

class PanoramaDebugServer:
    """Manages the HTTP debug server in a daemon thread."""

    def __init__(self, port: int = 8780) -> None:
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._server = HTTPServer(("", self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="panorama-debug-ui",
        )
        self._thread.start()
        logger.info(f"Debug UI started on http://localhost:{self.port}")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        logger.info("Debug UI stopped")

    def update_state(
        self,
        result: Any,
        sensor: Any | None = None,
        perception: Any | None = None,
    ) -> None:
        """Push the latest agent state into the shared buffer."""
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

        # Viewport image (from the raw SensorFrame, if available)
        if sensor and hasattr(sensor, "get_status"):
            # The viewport JPEG is in the SensorFrame raw_data — we
            # can read it from the perception's last frame if available.
            pass

        # Try to get viewport image from extras
        viewport_b64 = extras.get("viewport_image_b64")
        if not viewport_b64:
            # fallback: look in the raw data carried on the result
            raw = getattr(result, "raw_data", None)
            if isinstance(raw, dict):
                viewport_b64 = raw.get("image_bytes_b64")
        patch["viewport_image_b64"] = viewport_b64

        # Append a compact message log entry
        msg_entry = {
            "step": patch["step"],
            "source": patch["source_file"],
            "heading": patch["heading_deg"],
            "location": patch["location_label"],
            "conf": round(patch["location_confidence"], 3),
            "sig": patch.get("feature_summary", {}).get("scene_signature", ""),
        }

        current = _state.snapshot()
        log = current.get("message_log", [])
        log.append(msg_entry)
        patch["message_log"] = log

        _state.update(patch)
