"""CLI entry point for the episodic memory agent."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

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

app = typer.Typer(
    name="episodic-agent",
    help="Event-segmented episodic memory agent",
    add_completion=False,
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


@app.command()
def run(
    steps: int = typer.Option(
        100,
        "--steps",
        "-n",
        help="Number of steps to run",
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
) -> None:
    """Run the episodic memory agent loop.
    
    Executes the cognitive loop for the specified number of steps,
    writing JSONL logs to runs/<timestamp>/run.jsonl.
    """
    # Generate run ID from timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.jsonl"
    
    # Calculate step interval from FPS
    step_interval = 1.0 / fps if fps > 0 else 0.0
    
    # Initialize all stub modules with seed
    sensor = StubSensorProvider(max_frames=steps, seed=seed)
    perception = StubPerception(seed=seed)
    acf_builder = StubACFBuilder(seed=seed)
    location_resolver = StubLocationResolver(seed=seed)
    entity_resolver = StubEntityResolver(seed=seed)
    event_resolver = StubEventResolver(seed=seed)
    retriever = StubRetriever(seed=seed)
    boundary_detector = StubBoundaryDetector(freeze_interval=freeze_interval, seed=seed)
    dialog_manager = StubDialogManager(auto_accept=True, seed=seed)
    episode_store = InMemoryEpisodeStore()
    graph_store = InMemoryGraphStore()
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        sensor=sensor,
        perception=perception,
        acf_builder=acf_builder,
        location_resolver=location_resolver,
        entity_resolver=entity_resolver,
        event_resolver=event_resolver,
        retriever=retriever,
        boundary_detector=boundary_detector,
        dialog_manager=dialog_manager,
        episode_store=episode_store,
        graph_store=graph_store,
        run_id=run_id,
    )
    
    # Print header
    if not quiet:
        typer.echo(f"Episodic Memory Agent - Phase 1")
        typer.echo(f"Run ID: {run_id}")
        typer.echo(f"Steps: {steps}, FPS: {fps}, Seed: {seed}, Freeze Interval: {freeze_interval}")
        typer.echo(f"Log: {log_path}")
        typer.echo("-" * 60)
    
    # Run the loop with logging
    with LogWriter(log_path) as logger:
        for _ in range(steps):
            step_start = time.perf_counter()
            
            # Execute one step
            result = orchestrator.step()
            
            # Log the result
            logger.write(result)
            
            # Print summary
            if not quiet:
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
            
            # Throttle to target FPS
            elapsed = time.perf_counter() - step_start
            if step_interval > elapsed:
                time.sleep(step_interval - elapsed)
    
    # Print summary
    if not quiet:
        typer.echo("-" * 60)
        typer.echo(f"Completed {steps} steps")
        typer.echo(f"Episodes frozen: {orchestrator.episode_count}")
        typer.echo(f"Log written to: {log_path}")


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
