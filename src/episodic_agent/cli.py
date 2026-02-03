"""CLI entry point for the episodic memory agent."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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
        help="Module profile to use (stub, unity_cheat)",
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
) -> None:
    """Run the episodic memory agent loop.
    
    Executes the cognitive loop for the specified number of steps,
    writing JSONL logs to runs/<timestamp>/run.jsonl.
    
    Examples:
    
        # Run with stub modules (Phase 1 testing)
        python -m episodic_agent.cli run --profile stub --steps 200
        
        # Run with Unity integration
        python -m episodic_agent.cli run --profile unity_cheat --unity-ws ws://localhost:8765 --fps 10
    """
    setup_logging(verbose)
    
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
    is_unity = profile.lower() == "unity_cheat"
    infinite_mode = steps == 0 and is_unity
    
    # Print header
    if not quiet:
        typer.echo(f"Episodic Memory Agent - Phase 5")
        typer.echo(f"Run ID: {run_id}")
        typer.echo(f"Profile: {profile_config.name} - {profile_config.description}")
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
                            
                            # Get recent events and deltas from ACF (Phase 5)
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
        
        # Show learned locations/entities for Unity mode
        if is_unity:
            graph_store = modules.get("graph_store")
            if hasattr(graph_store, "get_nodes_by_type"):
                from episodic_agent.schemas import NodeType
                
                locations = graph_store.get_nodes_by_type(NodeType.LOCATION)
                entities = graph_store.get_nodes_by_type(NodeType.ENTITY)
                events = graph_store.get_nodes_by_type(NodeType.EVENT)
                
                typer.echo(f"Learned locations: {len(locations)}")
                for loc in locations:
                    typer.echo(f"  - {loc.label} (visits: {loc.access_count})")
                
                typer.echo(f"Learned entities: {len(entities)}")
                
                # Show learned events (Phase 5)
                typer.echo(f"Learned event patterns: {len(events)}")
                for evt in events:
                    pattern = evt.extras.get("pattern_signature", "?")
                    typer.echo(f"  - {evt.label} [{pattern}]")
            
            # Show event resolver stats (Phase 5)
            event_resolver = modules.get("event_resolver")
            if hasattr(event_resolver, "events_detected"):
                typer.echo("")
                typer.echo("Phase 5 Statistics:")
                typer.echo(f"  Deltas detected: {event_resolver.deltas_detected}")
                typer.echo(f"  Events detected: {event_resolver.events_detected}")
                typer.echo(f"  Events labeled: {event_resolver.events_labeled}")
                typer.echo(f"  Events recognized: {event_resolver.events_recognized}")
                typer.echo(f"  Questions asked: {event_resolver.questions_asked}")


@app.command()
def profiles() -> None:
    """List available profiles."""
    typer.echo("Available profiles:\n")
    for name, desc in list_profiles():
        typer.echo(f"  {name:15s} - {desc}")


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
