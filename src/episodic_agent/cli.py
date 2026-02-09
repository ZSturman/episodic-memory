"""CLI entry point for the episodic memory agent."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer

from episodic_agent.core.orchestrator import AgentOrchestrator
from episodic_agent.memory.stubs import InMemoryEpisodeStore, InMemoryGraphStore
from episodic_agent.metrics.logging import LogWriter
from episodic_agent.modules.stubs import (
    StubACFBuilder,
    StubBoundaryDetector,
    StubDialogManager,
    StubEntityResolver,
    StubEventResolver,
    StubLocationResolver,
    StubPerception,
    StubRetriever,
    StubSensorProvider,
)
from episodic_agent.utils.config import DEFAULT_FREEZE_INTERVAL
from episodic_agent.utils.profiles import (
    ModuleFactory,
    get_profile,
    list_profiles,
    PROFILES,
)

app = typer.Typer(
    name="episodic-agent",
    help="Event-segmented episodic memory agent",
    add_completion=False,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def format_step_summary(
    step: int,
    location: str,
    location_conf: float,
    entities: int,
    events: int,
    episodes: int,
    boundary: bool,
) -> str:
    """Format a single-line step summary for console output.
    
    Args:
        step: Current step number.
        location: Location label.
        location_conf: Location confidence.
        entities: Entity count.
        events: Event count.
        episodes: Episode count.
        boundary: Whether boundary was triggered.
        
    Returns:
        Formatted summary string.
    """
    boundary_marker = " [FREEZE]" if boundary else ""
    return (
        f"[{step:04d}] "
        f"loc={location}({location_conf:.2f}) "
        f"ent={entities} evt={events} ep={episodes}"
        f"{boundary_marker}"
    )


def format_unity_step_summary(
    step: int,
    location: str,
    location_conf: float,
    entity_summary: dict[str, int],
    episodes: int,
    boundary: bool,
    frame_id: int | None = None,
    connection_state: str = "unknown",
    recent_events: list[dict] | None = None,
    recent_deltas: list[dict] | None = None,
    event_count: int = 0,
    delta_count: int = 0,
) -> str:
    """Format an enhanced step summary for Unity mode.
    
    Args:
        step: Current step number.
        location: Location label.
        location_conf: Location confidence.
        entity_summary: Dict of category -> count.
        episodes: Episode count.
        boundary: Whether boundary was triggered.
        frame_id: Last frame ID from Unity.
        connection_state: WebSocket connection state.
        recent_events: Recent events in current ACF.
        recent_deltas: Recent deltas in current ACF.
        event_count: Total event count in ACF.
        delta_count: Total delta count in ACF.
        
    Returns:
        Formatted summary string.
    """
    boundary_marker = " 📦" if boundary else ""
    
    # Format entity summary
    if entity_summary:
        entity_str = " ".join(f"{cat}:{cnt}" for cat, cnt in sorted(entity_summary.items()))
    else:
        entity_str = "none"
    
    # Connection indicator
    conn_icon = "🟢" if connection_state == "connected" else "🟡" if "connect" in connection_state else "🔴"
    
    frame_str = f"#{frame_id}" if frame_id is not None else "#?"
    
    # Format recent events
    event_str = ""
    if recent_events:
        # Show last 2 events with labels
        event_labels = [e.get("label", "?")[:20] for e in recent_events[-2:]]
        event_str = f" 🎬{event_count}:[{', '.join(event_labels)}]"
    elif event_count > 0:
        event_str = f" 🎬{event_count}"
    
    # Format recent deltas
    delta_str = ""
    if delta_count > 0:
        delta_str = f" Δ{delta_count}"
    
    return (
        f"[{step:04d}] {conn_icon} {frame_str} "
        f"📍 {location}({location_conf:.0%}) "
        f"👁 [{entity_str}] "
        f"📚 {episodes}{delta_str}{event_str}"
        f"{boundary_marker}"
    )


@app.command()
def run(
    steps: int = typer.Option(
        100,
        "--steps",
        "-n",
        help="Number of steps to run (0 = infinite for Unity mode)",
    ),
    fps: float = typer.Option(
        10.0,
        "--fps",
        "-f",
        help="Target frames per second",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for deterministic behavior",
    ),
    freeze_interval: int = typer.Option(
        DEFAULT_FREEZE_INTERVAL,
        "--freeze-interval",
        "-i",
        help="Steps between episode freezes",
    ),
    output_dir: Path = typer.Option(
        Path("runs"),
        "--output",
        "-o",
        help="Output directory for run logs",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress per-step output",
    ),
    profile: str = typer.Option(
        "stub",
        "--profile",
        "-p",
        help="Module profile to use (stub, unity_cheat, unity_full)",
    ),
    unity_ws: Optional[str] = typer.Option(
        None,
        "--unity-ws",
        help="Unity WebSocket URL (overrides profile default)",
    ),
    auto_label: bool = typer.Option(
        False,
        "--auto-label",
        help="Auto-generate labels without prompting",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    log_sensor_data: bool = typer.Option(
        False,
        "--log-sensor-data",
        help="Log raw sensor data for debugging Unity communication",
    ),
) -> None:
    """Run the episodic memory agent loop.
    
    Executes the cognitive loop for the specified number of steps,
    writing JSONL logs to runs/<timestamp>/run.jsonl.
    
    Examples:
    
        # Run with simulated sensors for testing
        python -m episodic_agent.cli run --profile stub --steps 200
        
        # Run with Unity live sensor stream  
        python -m episodic_agent.cli run --profile unity_cheat --unity-ws ws://localhost:8765 --fps 10
        
        # Debug Unity communication with sensor data logging
        python -m episodic_agent.cli run --profile unity_full --unity-ws ws://localhost:8765 --log-sensor-data -v
    """
    setup_logging(verbose)
    
    # Enable sensor data logging if requested
    if log_sensor_data:
        import logging as log_module
        data_logger = log_module.getLogger("episodic_agent.modules.unity.sensor_provider.data")
        data_logger.setLevel(log_module.DEBUG)
        # Add a handler if none exists
        if not data_logger.handlers:
            handler = log_module.StreamHandler()
            handler.setFormatter(log_module.Formatter(
                "%(asctime)s [DATA] %(message)s",
                datefmt="%H:%M:%S",
            ))
            data_logger.addHandler(handler)
            data_logger.propagate = False
        typer.echo("📡 Sensor data logging enabled - raw Unity↔Python communication will be logged")
    
    # Generate run ID from timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.jsonl"
    
    # Calculate step interval from FPS
    step_interval = 1.0 / fps if fps > 0 else 0.0
    
    # Get profile and create modules
    try:
        profile_config = get_profile(profile)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    
    # Build parameter overrides
    overrides = {
        "freeze_interval": freeze_interval,
        "log_raw_data": log_sensor_data,  # Pass to sensor provider
    }
    if unity_ws:
        overrides["ws_url"] = unity_ws
    if auto_label:
        overrides["auto_label_locations"] = True
        overrides["auto_label_entities"] = True
    
    # Create module factory
    factory = ModuleFactory(
        profile=profile_config,
        run_dir=run_dir,
        seed=seed,
        **overrides,
    )
    
    # Create all modules
    modules = factory.create_modules()
    
    # Create orchestrator
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
        run_id=run_id,
    )
    
    # Determine if Unity mode for enhanced output
    is_unity = profile.lower().startswith("unity")
    infinite_mode = steps == 0 and is_unity
    
    # Print header
    if not quiet:
        typer.echo(f"Episodic Memory Agent")
        typer.echo(f"Mode: {profile_config.description}")
        typer.echo(f"Run ID: {run_id}")
        if is_unity:
            ws_url = overrides.get("ws_url") or profile_config.parameters.get("ws_url")
            typer.echo(f"Unity WS: {ws_url}")
        typer.echo(f"Steps: {'∞' if infinite_mode else steps}, FPS: {fps}, Seed: {seed}")
        typer.echo(f"Log: {log_path}")
        typer.echo("-" * 60)
    
    # Run the loop with logging
    step_count = 0
    
    try:
        with LogWriter(log_path) as logger:
            while True:
                step_start = time.perf_counter()
                
                try:
                    # Execute one step
                    result = orchestrator.step()
                    step_count += 1
                    
                    # Log the result
                    logger.write(result)
                    
                    # Print summary
                    if not quiet:
                        if is_unity:
                            # Get entity resolver for summary
                            entity_resolver = modules.get("entity_resolver")
                            if hasattr(entity_resolver, "get_visible_entity_summary"):
                                entity_summary = entity_resolver.get_visible_entity_summary()
                            else:
                                entity_summary = {}
                            
                            # Get sensor provider for connection status
                            sensor = modules.get("sensor")
                            if hasattr(sensor, "state"):
                                conn_state = sensor.state
                            else:
                                conn_state = "unknown"
                            
                            # Get recent events and deltas from ACF for display
                            recent_events = []
                            recent_deltas = []
                            event_count = 0
                            delta_count = 0
                            
                            if orchestrator.acf:
                                recent_events = orchestrator.acf.get_recent_events(3)
                                recent_deltas = orchestrator.acf.get_recent_deltas(3)
                                event_count = len(orchestrator.acf.events)
                                # Get delta count from extras
                                delta_count = len(orchestrator.acf.deltas)
                                if "deltas" in orchestrator.acf.extras:
                                    delta_count += len(orchestrator.acf.extras["deltas"])
                            
                            summary = format_unity_step_summary(
                                step=result.step_number,
                                location=result.location_label,
                                location_conf=result.location_confidence,
                                entity_summary=entity_summary,
                                episodes=result.episode_count,
                                boundary=result.boundary_triggered,
                                frame_id=result.frame_id,
                                connection_state=conn_state,
                                recent_events=recent_events,
                                recent_deltas=recent_deltas,
                                event_count=event_count,
                                delta_count=delta_count,
                            )
                        else:
                            summary = format_step_summary(
                                step=result.step_number,
                                location=result.location_label,
                                location_conf=result.location_confidence,
                                entities=result.entity_count,
                                events=result.event_count,
                                episodes=result.episode_count,
                                boundary=result.boundary_triggered,
                            )
                        typer.echo(summary)
                    
                except TimeoutError:
                    if not quiet:
                        typer.echo("[WARN] Frame timeout, retrying...")
                    continue
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if not quiet:
                        typer.echo(f"[ERROR] Step failed: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue
                
                # Check step limit
                if not infinite_mode and step_count >= steps:
                    break
                
                # Throttle to target FPS
                elapsed = time.perf_counter() - step_start
                if step_interval > elapsed:
                    time.sleep(step_interval - elapsed)
    
    except KeyboardInterrupt:
        if not quiet:
            typer.echo("\n" + "-" * 60)
            typer.echo("Interrupted by user")
    
    finally:
        # Cleanup
        sensor = modules.get("sensor")
        if hasattr(sensor, "stop"):
            sensor.stop()
    
    # Print summary
    if not quiet:
        typer.echo("-" * 60)
        typer.echo(f"Completed {step_count} steps")
        typer.echo(f"Episodes frozen: {orchestrator.episode_count}")
        typer.echo(f"Log written to: {log_path}")
        
        # Show sensor/validation stats for Unity mode
        if is_unity:
            sensor = modules.get("sensor")
            if hasattr(sensor, "get_status"):
                status = sensor.get_status()
                typer.echo("")
                typer.echo("📡 Sensor Connection Stats:")
                typer.echo(f"  Final state: {status.get('state', 'unknown')}")
                typer.echo(f"  Frames dropped: {status.get('dropped_frames', 0)}")
                typer.echo(f"  Validation errors: {status.get('validation_errors', 0)}")
                typer.echo(f"  Reconnects: {status.get('reconnect_count', 0)}")
                
                # Show gateway stats if available
                if status.get('gateway_stats'):
                    gw = status['gateway_stats']
                    typer.echo(f"  Gateway messages: {gw.get('total_messages', 0)}")
                    typer.echo(f"  Gateway error rate: {gw.get('error_rate', 0):.1%}")
            
            # Show recent validation errors if any
            if hasattr(sensor, "get_recent_errors"):
                errors = sensor.get_recent_errors(5)
                if errors:
                    typer.echo("")
                    typer.echo("⚠️ Recent Validation Errors:")
                    for err in errors:
                        typer.echo(f"  [{err['severity']}] {err['code']}: {err['message']}")
                        if err.get('suggestion'):
                            typer.echo(f"    → {err['suggestion']}")
        
        # Show learned locations/entities for Unity mode
        if is_unity:
            graph_store = modules.get("graph_store")
            if hasattr(graph_store, "get_nodes_by_type"):
                from episodic_agent.schemas import NodeType
                
                locations = graph_store.get_nodes_by_type(NodeType.LOCATION)
                entities = graph_store.get_nodes_by_type(NodeType.ENTITY)
                events = graph_store.get_nodes_by_type(NodeType.EVENT)
                
                typer.echo("")
                typer.echo("📍 Learned Locations:")
                typer.echo(f"  Count: {len(locations)}")
                for loc in locations:
                    typer.echo(f"  - {loc.label} (visits: {loc.access_count})")
                
                typer.echo(f"Learned entities: {len(entities)}")
                
                # Show learned events
                typer.echo(f"Learned event patterns: {len(events)}")
                for evt in events:
                    pattern = evt.extras.get("pattern_signature", "?")
                    typer.echo(f"  - {evt.label} [{pattern}]")
            
            # Show event resolver stats
            event_resolver = modules.get("event_resolver")
            if hasattr(event_resolver, "events_detected"):
                typer.echo("")
                typer.echo("Event Detection Statistics:")
                typer.echo(f"  State changes detected: {event_resolver.deltas_detected}")
                typer.echo(f"  Events recognized: {event_resolver.events_detected}")
                typer.echo(f"  Events labeled by user: {event_resolver.events_labeled}")
                typer.echo(f"  Events matched to patterns: {event_resolver.events_recognized}")
                typer.echo(f"  User prompts issued: {event_resolver.questions_asked}")


@app.command()
def panorama(
    image_dir: Path = typer.Argument(
        ...,
        help="Directory containing panorama images and/or videos",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    steps: int = typer.Option(
        0,
        "--steps",
        "-n",
        help="Max steps (0 = run until all images exhausted)",
    ),
    fps: float = typer.Option(
        2.0,
        "--fps",
        "-f",
        help="Target frames per second",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for deterministic behavior",
    ),
    output_dir: Path = typer.Option(
        Path("runs"),
        "--output",
        "-o",
        help="Output directory for run logs",
    ),
    memory_dir: Optional[Path] = typer.Option(
        None,
        "--memory-dir",
        "-m",
        help="Directory for persistent memory (default: <output>/panorama_memory)",
    ),
    reset_memory: bool = typer.Option(
        False,
        "--reset-memory",
        help="Clear memory before starting",
    ),
    viewport_width: int = typer.Option(
        256,
        "--viewport-width",
        help="Width of each viewport crop in pixels",
    ),
    viewport_height: int = typer.Option(
        256,
        "--viewport-height",
        help="Height of each viewport crop in pixels",
    ),
    headings: int = typer.Option(
        8,
        "--headings",
        help="Number of horizontal viewpoints per image",
    ),
    auto_label: bool = typer.Option(
        False,
        "--auto-label",
        help="Auto-generate labels without prompting",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress per-step output",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    debug_ui: bool = typer.Option(
        False,
        "--debug-ui",
        help="Launch browser debug dashboard on port 8780",
    ),
) -> None:
    """Explore panorama images / video and infer location from visual evidence.

    The agent loads images from IMAGE_DIR, simulates looking around by
    sliding a viewport across each image, extracts visual features, and
    accumulates evidence to infer location.  On first encounter it asks
    the user for a label; on revisit it proposes a hypothesis.

    Examples:

        episodic-agent panorama ./my_photos/
        episodic-agent panorama ./rooms/ --debug-ui --headings 12
        episodic-agent panorama ./rooms/ --reset-memory
    """
    setup_logging(verbose)

    # Run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{run_id}_panorama"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.jsonl"

    # Memory directory
    mem_dir = memory_dir or (output_dir / "panorama_memory")
    if reset_memory and mem_dir.exists():
        import shutil
        shutil.rmtree(mem_dir)
        typer.echo(f"Cleared memory: {mem_dir}")
    mem_dir.mkdir(parents=True, exist_ok=True)

    # Build profile overrides
    overrides: dict[str, Any] = {
        "image_dir": str(image_dir),
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
        "headings_per_image": headings,
    }
    if auto_label:
        overrides["auto_label_locations"] = True
        overrides["auto_label_entities"] = True

    # Get panorama profile
    try:
        profile_config = get_profile("panorama")
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    factory = ModuleFactory(
        profile=profile_config,
        run_dir=mem_dir,   # persistent memory lives here
        seed=seed,
        **overrides,
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
        run_id=run_id,
    )

    # Terminal debugger
    debugger = None
    if not quiet:
        from episodic_agent.modules.panorama.debug import TerminalDebugger
        debugger = TerminalDebugger(
            location_resolver=modules.get("location_resolver"),
        )

    # Web debug UI
    debug_server = None
    api_server = None
    event_bus = None
    investigation_sm = None

    if debug_ui:
        # Create observability infrastructure
        from episodic_agent.modules.panorama.event_bus import PanoramaEventBus
        from episodic_agent.modules.panorama.investigation import InvestigationStateMachine
        from episodic_agent.modules.panorama.api_server import PanoramaAPIServer
        from episodic_agent.modules.panorama.replay import ReplayController

        event_bus = PanoramaEventBus()
        replay_controller = ReplayController(event_bus=event_bus)
        investigation_sm = InvestigationStateMachine(
            min_investigation_steps=5,
            max_investigation_steps=20,
            label_request_ceiling=0.4,
            confident_match_threshold=0.7,
            plateau_threshold=0.05,
            min_images=2,
            event_bus=event_bus,
        )

        # Inject event bus into modules (post-hoc wiring)
        loc_resolver = modules["location_resolver"]
        if hasattr(loc_resolver, "_event_bus"):
            loc_resolver._event_bus = event_bus
            loc_resolver._investigation_sm = investigation_sm

        perc = modules["perception"]
        if hasattr(perc, "_event_bus"):
            perc._event_bus = event_bus
        if hasattr(perc, "_investigation_sm"):
            perc._investigation_sm = investigation_sm

        # Label callback: applies label via resolver and resets SM
        def _label_callback(label: str) -> None:
            if hasattr(loc_resolver, "apply_dashboard_label"):
                loc_resolver.apply_dashboard_label(label)

        # Start API server
        api_server = PanoramaAPIServer(
            port=8780,
            event_bus=event_bus,
            location_resolver=loc_resolver,
            perception=perc,
            label_callback=_label_callback,
            replay_controller=replay_controller,
            config={
                "transition_threshold": overrides.get("transition_threshold", 0.40),
                "hysteresis_frames": overrides.get("hysteresis_frames", 3),
                "match_threshold": overrides.get("match_threshold", 0.35),
                "headings_per_image": headings,
                "viewport_size": f"{viewport_width}x{viewport_height}",
                "auto_label": auto_label,
                "fps": fps,
                "investigation": {
                    "min_steps": 5,
                    "max_steps": 20,
                    "plateau_threshold": 0.05,
                    "label_request_ceiling": 0.4,
                    "confident_match_threshold": 0.7,
                },
            },
        )
        api_server.start()
        typer.echo(f"Panorama API: http://localhost:8780")

        # Register CLI label callback so dashboard reflects when CLI prompts for a label
        dm = modules.get("dialog_manager")
        if dm and hasattr(dm, "set_on_label_callback"):
            def _cli_label_notify(label: str) -> None:
                """Notify the API server when the CLI receives a label."""
                current_step = api_server.state.get_field("step") or 0
                api_server.state.update({
                    "cli_label_event": {
                        "label": label,
                        "step": current_step,
                        "timestamp": datetime.now().isoformat(),
                    },
                })
            dm.set_on_label_callback(_cli_label_notify)

        # JSONL event export — subscribe to event bus and write each event
        import json as _json
        _events_path = run_dir / "events.jsonl"
        _events_file = open(_events_path, "a", encoding="utf-8")

        def _jsonl_sink(event: Any) -> None:
            """Append event to events.jsonl."""
            try:
                if hasattr(event, "model_dump"):
                    record = event.model_dump(mode="json")
                elif hasattr(event, "dict"):
                    record = event.dict()
                else:
                    record = {"raw": str(event)}
                _events_file.write(_json.dumps(record, default=str) + "\n")
                _events_file.flush()
            except Exception:
                pass  # never crash the agent for logging

        event_bus.subscribe(_jsonl_sink)
        typer.echo(f"Events log: {_events_path}")

        # Also start legacy debug server for backward compat
        from episodic_agent.modules.panorama.debug_server import PanoramaDebugServer
        debug_server = PanoramaDebugServer(port=8781)
        debug_server.start()
        typer.echo(f"Legacy debug UI: http://localhost:8781")

        # Auto-start Next.js dashboard dev server
        dashboard_proc = None
        dashboard_dir = Path(__file__).resolve().parent.parent.parent / "dashboard"
        if dashboard_dir.is_dir() and (dashboard_dir / "package.json").exists():
            import subprocess
            node_modules = dashboard_dir / "node_modules"
            if not node_modules.is_dir():
                typer.echo("Installing dashboard dependencies…")
                subprocess.run(
                    ["npm", "install"],
                    cwd=str(dashboard_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            dashboard_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(dashboard_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            typer.echo(f"Dashboard: http://localhost:3000")
        else:
            typer.echo("Dashboard directory not found — run `npm run dev` manually in dashboard/")

    step_interval = 1.0 / fps if fps > 0 else 0.0
    infinite_mode = steps == 0

    # Header
    if not quiet:
        sensor = modules["sensor"]
        typer.echo("Episodic Memory Agent — Panorama Harness")
        typer.echo(f"Images: {image_dir}  ({sensor.source_count} sources)")
        typer.echo(f"Viewports: {headings} per image, {viewport_width}×{viewport_height}px")
        typer.echo(f"Memory: {mem_dir}")
        typer.echo(f"Run ID: {run_id}")
        typer.echo(f"Log: {log_path}")
        typer.echo("-" * 60)

    step_count = 0
    try:
        with LogWriter(log_path) as logger_w:
            while modules["sensor"].has_frames():
                step_start = time.perf_counter()

                try:
                    result = orchestrator.step()
                    step_count += 1
                    logger_w.write(result)

                    if debugger:
                        debugger.print_step(result, modules["sensor"], modules.get("perception"))

                    if debug_server:
                        debug_server.update_state(result, modules["sensor"], modules.get("perception"))

                    if api_server:
                        # Feed investigation SM with match data
                        _match_candidates = None
                        if investigation_sm and event_bus:
                            from episodic_agent.schemas.panorama_events import MatchEvaluation, MatchCandidate
                            extras = getattr(result, "extras", {}) or {}
                            scene_emb = extras.get("viewport_embedding") or extras.get("panoramic_embedding")
                            loc_resolver = modules["location_resolver"]
                            if scene_emb and hasattr(loc_resolver, "get_all_match_scores"):
                                _match_candidates = loc_resolver.get_all_match_scores(scene_emb)
                                margin = 0.0
                                if len(_match_candidates) >= 2:
                                    margin = _match_candidates[0].confidence - _match_candidates[1].confidence
                                evaluation = MatchEvaluation(
                                    candidates=_match_candidates,
                                    top_margin=margin,
                                    current_location_id=getattr(loc_resolver, "_current_location_id", None),
                                )
                                viewport_b64 = extras.get("viewport_image_b64")
                                feature_summary = extras.get("feature_summary")
                                source_file = extras.get("source_file")
                                investigation_sm.update(
                                    evaluation, viewport_b64, feature_summary,
                                    source_file=source_file,
                                )

                        api_server.update_state(
                            result,
                            modules["sensor"],
                            modules.get("perception"),
                            investigation_sm,
                            match_candidates=_match_candidates,
                        )

                except TimeoutError:
                    continue
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if not quiet:
                        typer.echo(f"[ERROR] Step failed: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue

                if not infinite_mode and step_count >= steps:
                    break

                elapsed = time.perf_counter() - step_start
                if step_interval > elapsed:
                    time.sleep(step_interval - elapsed)

    except KeyboardInterrupt:
        if not quiet:
            typer.echo("\n" + "-" * 60)
            typer.echo("Interrupted by user")

    finally:
        sensor = modules.get("sensor")
        if hasattr(sensor, "stop"):
            sensor.stop()
        if debug_server:
            debug_server.stop()
        if api_server:
            api_server.stop()
        # Stop dashboard dev server
        if 'dashboard_proc' in dir() and dashboard_proc and dashboard_proc.poll() is None:
            dashboard_proc.terminate()
            dashboard_proc.wait(timeout=5)
        # Close JSONL event log
        if '_events_file' in dir() and _events_file and not _events_file.closed:
            _events_file.close()

    # Summary
    if not quiet:
        typer.echo("-" * 60)
        typer.echo(f"Completed {step_count} steps")
        typer.echo(f"Episodes frozen: {orchestrator.episode_count}")
        typer.echo(f"Log: {log_path}")
        typer.echo(f"Memory: {mem_dir}")

        graph_store = modules.get("graph_store")
        if hasattr(graph_store, "get_nodes_by_type"):
            from episodic_agent.schemas import NodeType
            locations = graph_store.get_nodes_by_type(NodeType.LOCATION)
            typer.echo(f"\nLearned locations ({len(locations)}):")
            for loc in locations:
                typer.echo(f"  - {loc.label} (visits: {loc.access_count})")


@app.command()
def profiles() -> None:
    """List available profiles."""
    typer.echo("Available profiles:\n")
    for name, desc in list_profiles():
        typer.echo(f"  {name:15s} - {desc}")


@app.command()
def report(
    run_folder: Path = typer.Argument(
        ...,
        help="Path to run folder (e.g., runs/20260202_174416)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    html: bool = typer.Option(
        False,
        "--html",
        "-H",
        help="Open HTML report in browser after generation",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Only save reports, don't print to console",
    ),
) -> None:
    """Generate report from a run folder.
    
    Generates both text and HTML reports from run data.
    
    Example:
        python -m episodic_agent report runs/20260202_174416
        python -m episodic_agent report runs/20260202_174416 --html
    """
    from episodic_agent.report import ReportGenerator
    
    typer.echo(f"Generating reports for {run_folder}...")
    
    try:
        generator = ReportGenerator(run_folder)
        text_path, html_path = generator.save_reports()
        
        if not quiet:
            # Print text report to console
            text_report = generator.generate_text_report()
            typer.echo("")
            typer.echo(text_report)
            typer.echo("")
        
        typer.echo(f"Text report: {text_path}")
        typer.echo(f"HTML report: {html_path}")
        
        if html:
            import webbrowser
            webbrowser.open(f"file://{html_path.absolute()}")
            typer.echo("Opened HTML report in browser.")
            
    except Exception as e:
        typer.echo(f"Error generating report: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def scenario(
    scenario_name: str = typer.Argument(
        ...,
        help="Scenario to run: walk_rooms, toggle_drawer_light, spawn_move_ball, mixed",
    ),
    profile: str = typer.Option(
        "unity_full",
        "--profile",
        "-p",
        help="Module profile to use (unity_cheat, unity_full)",
    ),
    output_dir: Path = typer.Option(
        Path("runs"),
        "--output",
        "-o",
        help="Output directory for run logs",
    ),
    unity_ws: Optional[str] = typer.Option(
        None,
        "--unity-ws",
        help="Unity WebSocket URL (overrides profile default)",
    ),
    replay_file: Optional[Path] = typer.Option(
        None,
        "--replay",
        "-r",
        help="JSONL file for sensor replay (offline mode)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress per-step output",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for deterministic behavior",
    ),
) -> None:
    """Run an automated scenario.
    
    Executes pre-defined test scenarios for evaluation.
    
    Available scenarios:
      - walk_rooms: Walk through all rooms
      - toggle_drawer_light: Toggle drawer and lights
      - spawn_move_ball: Spawn and move a ball
      - mixed: All above combined
    
    Example:
        python -m episodic_agent scenario walk_rooms
        python -m episodic_agent scenario mixed --profile unity_full
        python -m episodic_agent scenario walk_rooms --replay replay.jsonl
    """
    from datetime import datetime
    import json
    
    from episodic_agent.scenarios.runner import ScenarioRunner
    from episodic_agent.scenarios.definitions import (
        WalkRoomsScenario,
        ToggleDrawerLightScenario,
        SpawnMoveBallScenario,
        MixedScenario,
    )
    
    setup_logging(verbose=False)
    
    # Map scenario names to classes
    scenarios = {
        "walk_rooms": WalkRoomsScenario,
        "toggle_drawer_light": ToggleDrawerLightScenario,
        "spawn_move_ball": SpawnMoveBallScenario,
        "mixed": MixedScenario,
    }
    
    if scenario_name not in scenarios:
        typer.echo(f"Unknown scenario: {scenario_name}", err=True)
        typer.echo(f"Available: {', '.join(scenarios.keys())}", err=True)
        raise typer.Exit(1)
    
    # Get profile config
    try:
        profile_config = get_profile(profile)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{timestamp}_scenario_{scenario_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"Running scenario: {scenario_name}")
    typer.echo(f"Profile: {profile}")
    typer.echo(f"Output: {run_dir}")
    
    # Create scenario instance
    scenario_cls = scenarios[scenario_name]
    scenario_instance = scenario_cls()
    
    # Create runner with profile and output dir
    runner = ScenarioRunner(
        output_dir=output_dir,
        profile=profile,
        verbose=not quiet,
    )
    
    typer.echo("-" * 60)
    typer.echo("Starting scenario...")
    
    # Run scenario
    result = runner.run(scenario_instance)
    
    # Print results
    typer.echo("-" * 60)
    typer.echo(f"Scenario: {result.scenario_name}")
    typer.echo(f"Success: {'✓' if result.success else '✗'}")
    typer.echo(f"Steps: {result.steps_completed}")
    typer.echo(f"Episodes: {result.episodes_created}")
    typer.echo(f"Duration: {result.duration_seconds:.1f}s")
    
    if result.error:
        typer.echo(f"Error: {result.error}", err=True)
    
    # Save result
    result_path = run_dir / "scenario_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "scenario_name": result.scenario_name,
            "success": result.success,
            "steps_completed": result.steps_completed,
            "episodes_created": result.episodes_created,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }, f, indent=2)
    
    typer.echo(f"Result saved: {result_path}")
    
    # Generate report
    typer.echo("")
    typer.echo("Generating report...")
    from episodic_agent.report import ReportGenerator
    generator = ReportGenerator(run_dir)
    text_path, html_path = generator.save_reports()
    typer.echo(f"Report: {html_path}")


@app.command()
def visualize(
    run_folder: Path = typer.Argument(
        ...,
        help="Path to run folder (e.g., runs/20260202_174416)",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    html: bool = typer.Option(
        False,
        "--html",
        "-H",
        help="Generate interactive HTML report",
    ),
    csv: bool = typer.Option(
        False,
        "--csv",
        "-c",
        help="Export data to CSV for external analysis",
    ),
    plot: bool = typer.Option(
        False,
        "--plot",
        "-p",
        help="Generate matplotlib plots (requires matplotlib)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Custom output path for generated files",
    ),
) -> None:
    """Visualize and analyze run data.
    
    Load a completed run and generate visualizations for analysis.
    
    Examples:
        # Print summary
        python -m episodic_agent visualize runs/20260202_174416
        
        # Generate HTML report
        python -m episodic_agent visualize runs/20260202_174416 --html
        
        # Export CSV
        python -m episodic_agent visualize runs/20260202_174416 --csv
        
        # Generate matplotlib plots
        python -m episodic_agent visualize runs/20260202_174416 --plot
    """
    from episodic_agent.visualize import RunVisualizer, plot_with_matplotlib
    
    try:
        viz = RunVisualizer(run_folder)
        viz.load()
        
        # Always print summary
        viz.print_summary()
        
        if html:
            output_path = output if output else None
            path = viz.generate_html_report(output_path)
            typer.echo(f"Generated HTML report: {path}")
        
        if csv:
            output_path = output if output else None
            path = viz.export_csv(output_path)
            typer.echo(f"Exported CSV: {path}")
        
        if plot:
            output_path = output if output else None
            plot_with_matplotlib(run_folder, output_path)
        
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from episodic_agent import __version__
    typer.echo(f"episodic-agent v{__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
