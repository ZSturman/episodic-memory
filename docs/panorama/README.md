# Panorama / Video Harness

A standalone test harness that lets the episodic-memory agent **infer location purely from visual input** — panoramic images or video frames — without any Unity connection or ground-truth labels. Includes a full-featured Next.js observability dashboard, structured event streaming, an investigation state machine, and JSONL replay for post-hoc analysis.

---

## Quick Start

```bash
# Run on a folder of images
episodic-agent panorama ./my_images/

# Run with the full observability dashboard (auto-starts Next.js)
episodic-agent panorama ./my_images/ --debug-ui

# Use a specific memory directory (persists across runs)
episodic-agent panorama ./my_images/ --memory-dir ./my_memory

# Reset memory and start fresh
episodic-agent panorama ./my_images/ --reset-memory

# Auto-label locations (skip interactive prompts)
episodic-agent panorama ./my_images/ --auto-label
```

### Supported Input Formats

| Type   | Extensions                                  |
|--------|---------------------------------------------|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`    |
| Video  | `.mp4`, `.avi`, `.mov`, `.mkv`              |

Images are processed in **sorted filename order**. Each image is treated as a panoramic view — the agent extracts multiple viewport crops from left to right, simulating "looking around."

Video files are sampled at a configurable FPS (default: 2.0). Each sampled frame is treated the same as an image.

---

## How It Works

### 1. Viewport Scanning (Saccade Policy)

For each source image, the `SaccadePolicy` slides a viewport window across the image width, producing `N` crops (default 8). Each crop gets a heading in degrees (0 to 360), simulating the agent turning its head.

```
Image: [==============================]
        vp1   vp2   vp3   ...   vp8
        0     45    90          315
```

The policy has three modes:
- **SCANNING** — evenly-spaced viewports across the image (default)
- **INVESTIGATING** — centres on the most salient grid cell
- **CONFIRMING** — jumps to a diagnostic heading to verify a hypothesis

### 2. Feature Extraction

Each viewport crop is processed by `PanoramaFeatureExtractor`, which computes **hand-crafted features only** (no ML dependencies):

| Feature              | Dimensions | Method                                    |
|----------------------|------------|-------------------------------------------|
| Color histogram      | 24         | 8-bin RGB histogram x 3 channels          |
| Edge histogram       | 8          | Sobel gradient direction bins (magnitude-weighted) |
| Grid brightness      | 16         | 4x4 cell mean luminance (0-1)             |
| Per-cell edge density| 16         | Sum of each cell's edge histogram          |
| Dominant colors      | 3 x 3      | Top-3 quantised-frequency colours (RGB)   |
| Scene signature      | -          | MD5 hash of quantised brightness + colours |

These are packaged into an `ExtractedVisualFeatures` dataclass (from `schemas/visual.py`) with a **128-dimensional embedding** via `to_embedding()`:

```
[global_color_histogram(64)] + [edge_histogram(8)] + [cell_brightness(16)] + [padding(40)]
```

### 3. Panoramic Embedding

After all viewports for one image are processed, per-heading embeddings are averaged element-wise into a single **panoramic embedding** — the composite "scene fingerprint" for that location. The averaging happens inside `PanoramaFeatureExtractor.accumulate_panorama_embedding()`.

### 4. Location Resolution

The `LocationResolverReal` compares the panoramic embedding against stored **location fingerprints** using cosine distance:

- **match_threshold = 0.35** — embeddings closer than this are considered the same location
- **transition_threshold = 0.40** — embeddings farther than this trigger a location change
- **hysteresis_frames = 3** — a new location must be sustained for 3 steps before committing

### 5. Investigation State Machine

The `InvestigationStateMachine` controls *when* the agent should request a label from the user. Rather than prompting on the first ambiguous frame, the agent must accumulate evidence and detect a confidence plateau:

| State                       | Meaning                                               |
|-----------------------------|-------------------------------------------------------|
| `investigating_unknown`     | First encounter — gathering viewport evidence         |
| `matching_known`            | Scene resembles a stored location                     |
| `low_confidence_match`      | Best candidate is weak                                |
| `confident_match`           | High confidence in identity (no label needed)         |
| `novel_location_candidate`  | Evidence points to a genuinely new location           |
| `label_request`             | Evidence bundle ready — requesting a label            |

Transitions are adaptive — the investigation window length depends on how quickly confidence stabilises, bounded by configurable min/max steps.

**Key thresholds** (configurable via the factory):

| Parameter                    | Default | Purpose                                      |
|------------------------------|---------|----------------------------------------------|
| `min_investigation_steps`    | `5`     | Minimum viewports before any label request    |
| `max_investigation_steps`    | `20`    | Force a decision after this many steps        |
| `plateau_threshold`          | `0.05`  | Rolling std below which confidence is settled |
| `label_request_ceiling`      | `0.4`   | Max confidence for triggering a label request |
| `confident_match_threshold`  | `0.7`   | Auto-accept match above this confidence       |

### 6. Labeling Flow

When a new location is detected (no matching fingerprint), the investigation state machine accumulates evidence over N viewports, detects a confidence plateau, and then asks "What is this place?" via terminal or dashboard. The user types a label, and the fingerprint is stored.

When a known location is revisited (fingerprint matches), the agent proposes a hypothesis with confidence. The user can confirm or correct.

With `--auto-label`, locations are automatically assigned `location_N` labels without prompting.

---

## CLI Options

```
episodic-agent panorama [OPTIONS] IMAGE_DIR
```

| Option               | Default                    | Description                                      |
|----------------------|----------------------------|--------------------------------------------------|
| `IMAGE_DIR`          | *(required)*               | Path to folder containing images or videos        |
| `--steps`            | `0` (all)                  | Max steps to run (0 = process all sources)        |
| `--fps`              | `2.0`                      | Sampling rate for video files                     |
| `--seed`             | `42`                       | Random seed for reproducibility                   |
| `--output-dir`       | `runs/`                    | Parent directory for run outputs                  |
| `--memory-dir`       | `<output>/panorama_memory` | Directory for persistent memory                   |
| `--reset-memory`     | `false`                    | Wipe memory before starting                       |
| `--viewport-width`   | `256`                      | Width of each viewport crop (px)                  |
| `--viewport-height`  | `256`                      | Height of each viewport crop (px)                 |
| `--headings`         | `8`                        | Number of viewport positions per image            |
| `--auto-label`       | `false`                    | Auto-label locations without prompting            |
| `--quiet` / `-q`     | `false`                    | Suppress terminal debug output                    |
| `--verbose` / `-v`   | `false`                    | Enable verbose logging                            |
| `--debug-ui`         | `false`                    | Start API server + Next.js dashboard (see below)  |

---

## SensorFrame Message Schema

Every viewport crop produces a `SensorFrame` message — the same data contract used by all sensor pipelines:

```python
SensorFrame(
    frame_id=42,
    timestamp=datetime(...),
    sensor_type="panorama",
    raw_data={
        "image_bytes_b64": "<base64-encoded JPEG>"
    },
    extras={
        "transition": False,
        "source_file": "room_01.jpg",
        "source_index": 0,
        "heading_deg": 90.0,
        "viewport_index": 2,
        "total_viewports": 8,
        "camera_pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "forward": [0.0, 0.0, -1.0]
        }
    }
)
```

**Transition frames** are emitted between images (no `raw_data`) to signal a scene boundary. The `forward` vector rotates with heading: `[sin(t), 0, -cos(t)]`.

---

## Evidence Model

The agent does **not** receive any ground-truth location label. It must infer location entirely from accumulated visual evidence:

1. **Per-viewport features** — extracted from each crop independently
2. **Panoramic embedding** — element-wise mean of all heading embeddings for one image
3. **Location fingerprint** — stored embedding + label in `LocationResolverReal`
4. **Cosine distance** — used to match new observations against stored fingerprints
5. **Hysteresis** — prevents flickering between locations on ambiguous inputs
6. **Investigation gate** — ensures labels are only requested after evidence stabilisation

The `percept.extras` carries a `hypothesis` dict for inspection:

```python
{
    "label": "kitchen",
    "confidence": 0.82,
    "evidence_count": 8,
    "competing": []
}
```

---

## Observability Infrastructure

When `--debug-ui` is passed, the CLI spins up three interconnected systems:

### Event Bus (PanoramaEventBus)

A thread-safe ring buffer (default 1000 events) that modules emit into and the API server consumes. Supports subscriber callbacks for external sinks (e.g. JSONL logging).

Event types:
- `perception_update` — per-viewport feature extraction result
- `match_evaluation` — ranked location candidates with confidence scores
- `state_transition` — investigation SM changed state
- `investigation_window` — evidence bundle snapshot
- `label_request` — agent is requesting a label
- `memory_write` — a new location fingerprint was stored

### JSONL Event Log

Every event emitted to the bus is also appended to `<run_dir>/events.jsonl` for post-hoc replay. Each line is a JSON object with `event_type`, `timestamp`, `step`, `state`, `payload`, and optional `evidence_bundle`.

### API Server (PanoramaAPIServer)

A stdlib `http.server` JSON API on port **8780**. The Next.js dashboard proxies all `/api/*` requests to this server.

| Endpoint                       | Method | Description                                     |
|--------------------------------|--------|-------------------------------------------------|
| `/api/state`                   | GET    | Full agent state snapshot (backward-compatible)  |
| `/api/events?since_step=N`     | GET    | Event stream since step N                        |
| `/api/matches`                 | GET    | Current ranked match evaluation                  |
| `/api/evidence`                | GET    | Current evidence bundle (during investigation)   |
| `/api/memories`                | GET    | All location memory summaries                    |
| `/api/memories/:id`            | GET    | Detailed memory card for one location            |
| `/api/memories/:id/variance`   | GET    | Per-dimension embedding variance                 |
| `/api/memories/:id/embedding`  | GET    | Full centroid embedding vector                   |
| `/api/features`                | GET    | Current viewport raw feature arrays              |
| `/api/timeline?last=N`         | GET    | Confidence + state time series                   |
| `/api/config`                  | GET    | Current observability configuration              |
| `/api/graph/topology`          | GET    | Location graph (nodes + edges)                   |
| `/api/similarity/matrix`       | GET    | Pairwise cosine similarity matrix                |
| `/api/label`                   | POST   | Submit a label response from the dashboard       |
| `/api/replay/state`            | GET    | Current replay cursor and playback state         |
| `/api/replay/load`             | POST   | Load a JSONL events file for replay              |
| `/api/replay/control`          | POST   | Control replay (play/pause/seek/step/speed)      |

---

## Debug UI

### Terminal Output

Enabled by default (disable with `--quiet`). Shows per-step:
- Source file and heading progress bar
- Current location hypothesis and confidence
- Feature summary (brightness, edge density, dominant colors, scene signature)

### Next.js Dashboard (auto-started)

When `--debug-ui` is passed, the CLI **automatically starts the Next.js development server** from the `dashboard/` directory. If `node_modules/` is missing, `npm install` is run first. The dashboard is available at:

- **Dashboard**: `http://localhost:3000` (Next.js dev server)
- **API backend**: `http://localhost:8780` (Python API server)
- **Legacy debug UI**: `http://localhost:8781` (inline HTML fallback)

The Next.js `next.config.js` proxies all `/api/*` requests from `:3000` to `:8780`, so the dashboard communicates transparently with the Python backend.

> **Note**: If you prefer to run the dashboard manually (e.g. for development), you can run `npm run dev` in the `dashboard/` directory yourself before starting the agent with `--debug-ui`. The dashboard process is terminated automatically when the agent exits.

#### Dashboard Panels

The dashboard is organised into toggleable panels, grouped by three verbosity presets:

| Panel                   | Preset   | Description                                                    |
|-------------------------|----------|----------------------------------------------------------------|
| **Viewport**            | skim     | Current viewport crop image (JPEG)                             |
| **Recent Viewports**    | diagnose | Thumbnail strip of recent viewport crops                       |
| **Feature Panel**       | diagnose | Raw feature arrays — histograms, brightness grid, edge density |
| **Evidence Panel**      | diagnose | Evidence bundle during investigation — images, scores, history |
| **Match Panel**         | skim     | Ranked match candidates with confidence bars                   |
| **Confidence Timeline** | skim     | Time-series chart of confidence + state colour bands           |
| **Memory List**         | diagnose | Summary cards for all learned locations                        |
| **Memory Card Detail**  | deep     | Deep-dive into one location — centroid, variance, entities     |
| **Event Log**           | deep     | Scrollable structured event stream                             |
| **Label Request**       | skim     | Interactive prompt — submit a location label from the dashboard|
| **Location Graph**      | deep     | Node-edge graph of location transitions                        |
| **Similarity Heatmap**  | deep     | Pairwise cosine similarity matrix across all locations         |
| **Embedding Variance**  | deep     | Per-dimension variance chart for a selected location           |
| **Replay Controls**     | deep     | Load, play, pause, seek, step through a JSONL replay           |

#### Verbosity Presets

Switch presets from the sidebar to control how many panels are visible:

| Preset       | Active Panels                                                        |
|--------------|----------------------------------------------------------------------|
| **skim**     | Viewport, Matches, Confidence Timeline, Label Request                |
| **diagnose** | All of skim + Recent Viewports, Features, Evidence, Memory List      |
| **deep**     | All panels including graphs, heatmaps, variance, event log, replay   |

### Running Without the Dashboard

If you do not need the dashboard, omit `--debug-ui`. The agent still runs the full perception + location resolution pipeline, writes `run.jsonl`, and shows terminal output (unless `--quiet`).

---

## Replay System

Every `--debug-ui` run produces an `events.jsonl` file in the run directory. This file can be loaded into the dashboard for post-hoc analysis without a live agent.

### How to Replay

1. Start the dashboard (either via `--debug-ui` or manually with `npm run dev`).
2. Open the **Replay Controls** panel (visible in **deep** verbosity preset).
3. Load a JSONL file by entering its path and clicking Load.
4. Use play/pause/step/seek controls to scrub through the run.

### Replay API

| Endpoint              | Method | Body                                               |
|-----------------------|--------|----------------------------------------------------|
| `/api/replay/load`    | POST   | `{ "file": "/path/to/events.jsonl" }`              |
| `/api/replay/control` | POST   | `{ "action": "play" }` or `"pause"`, `"stop"`, `"step_forward"`, `"step_back"`, `"seek"`, `"set_speed"` |
| `/api/replay/state`   | GET    | Returns cursor, playing, speed, total events        |

The `seek` action accepts `{ "action": "seek", "cursor": 42 }`. The `set_speed` action accepts `{ "action": "set_speed", "speed": 2.0 }`.

Replayed events flow through the same event bus and API state, so all dashboard panels update exactly as they would during a live run.

---

## Memory Persistence

Memory is stored in the `--memory-dir` directory (default: `<output>/panorama_memory`):

| File                    | Store                    | Contents                          |
|-------------------------|--------------------------|-----------------------------------|
| `episodes.json`         | `PersistentEpisodeStore` | Episode boundaries and metadata   |
| `graph.json`            | `LabeledGraphStore`      | Location labels and co-occurrences|
| `location_fingerprints` | `LocationResolverReal`   | Stored embeddings per location    |

To **persist across runs**, point `--memory-dir` to a fixed directory:

```bash
# First run — learns "kitchen" and "hallway"
episodic-agent panorama ./session1/ --memory-dir ./my_memory

# Second run — recognises known locations from session 1
episodic-agent panorama ./session2/ --memory-dir ./my_memory
```

To **start fresh**:

```bash
episodic-agent panorama ./images/ --reset-memory
```

---

## Architecture

```
CLI: panorama --debug-ui
=========================

PanoramaSensorProvider --> PanoramaPerception
  (images to crops)        (features to 128-d embedding)
       |                          |
       | SaccadePolicy            | Percept
       | (viewport selection)     v
TerminalDebugger            AgentOrchestrator
                              (standard loop)
                                  |
                  +---------------+---------------+
                  v               v               v
          LocationResolver  BoundaryDetector  EpisodeStore
          EntityResolver    Retriever         GraphStore
          DialogManager

Observability Layer (--debug-ui only):

  PanoramaEventBus --> PanoramaAPIServer (:8780)
       |                      ^
       v                      | proxies /api/*
  InvestigationSM         Next.js Dashboard (:3000)
       |
       v
  events.jsonl --> ReplayController

  PanoramaDebugServer (:8781) — legacy inline HTML
```

The panorama harness reuses all standard orchestrator modules. Only `PanoramaSensorProvider` and `PanoramaPerception` are panorama-specific; everything else (location resolution, boundary detection, episode management, labeling) uses the same implementations as the Unity pipeline.

The observability layer (event bus, API server, investigation SM, dashboard, replay) is activated only when `--debug-ui` is passed.

---

## Module Reference

| Module                      | File                                                       | Role                                    |
|-----------------------------|------------------------------------------------------------|-----------------------------------------|
| `PanoramaSensorProvider`    | `src/episodic_agent/modules/panorama/sensor_provider.py`   | Image/video to viewport SensorFrames    |
| `PanoramaPerception`        | `src/episodic_agent/modules/panorama/perception.py`        | Features to 128-d embedding Percept     |
| `PanoramaFeatureExtractor`  | `src/episodic_agent/modules/panorama/feature_extractor.py` | Hand-crafted visual feature extraction  |
| `SaccadePolicy`             | `src/episodic_agent/modules/panorama/saccade.py`           | Viewport selection strategy             |
| `TerminalDebugger`          | `src/episodic_agent/modules/panorama/debug.py`             | Per-step terminal output                |
| `PanoramaDebugServer`       | `src/episodic_agent/modules/panorama/debug_server.py`      | Legacy inline HTML dashboard (:8781)    |
| `PanoramaAPIServer`         | `src/episodic_agent/modules/panorama/api_server.py`        | JSON REST API for dashboard (:8780)     |
| `PanoramaEventBus`          | `src/episodic_agent/modules/panorama/event_bus.py`         | Thread-safe event ring buffer           |
| `InvestigationStateMachine` | `src/episodic_agent/modules/panorama/investigation.py`     | Adaptive label-request gating           |
| `ReplayController`          | `src/episodic_agent/modules/panorama/replay.py`            | JSONL event playback for post-hoc runs  |

### Dashboard (Next.js)

The `dashboard/` directory contains a Next.js 14 app with:

| Directory                | Contents                                            |
|--------------------------|-----------------------------------------------------|
| `src/app/`               | Root layout and page                                |
| `src/components/panels/` | 14 panel components (Viewport, Match, Evidence, etc)|
| `src/components/layout/` | Sidebar (verbosity presets) and TopBar              |
| `src/hooks/`             | React hooks for polling each API endpoint           |
| `src/store/`             | Zustand store for panel visibility state            |
| `src/lib/api.ts`         | Typed API client (all endpoints)                    |
| `src/lib/types.ts`       | TypeScript types mirroring Python Pydantic schemas  |
| `next.config.js`         | Proxy rewrites: /api/* to localhost:8780            |

**Dependencies**: Next.js 14, React 18, Recharts (charts), Zustand (state), Tailwind CSS.

---

## Tests

```bash
# Run panorama tests only
python3 -m pytest tests/test_panorama.py -v

# Run all tests (including panorama)
python3 -m pytest tests/ -v
```

The test suite (32 tests) covers:
- `TestSaccadePolicy` — viewport geometry, heading spans, reset behaviour
- `TestPanoramaFeatureExtractor` — embedding shape, determinism, color discrimination
- `TestPanoramaSensorProvider` — file discovery, frame format, transition markers, exhaustion
- `TestPanoramaPerception` — percept structure, confidence variation, no-ground-truth invariant
- `TestTerminalDebugger` — crash-free output
- `TestPanoramaIntegration` — full orchestrator loop on synthetic images
