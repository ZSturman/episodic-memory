"""Metrics evaluation for episodic memory agent.

Computes per-run metrics including:
- Location accuracy (using Unity room GUID as ground truth)
- Entity recognition accuracy (GUID match rate)
- Event detection accuracy (state-transition match rate)
- Question rate (questions/min)
- Episode segmentation rate (episodes/min)
- Memory growth (nodes/edges/episodes over time)

Persists metrics as metrics.json in the run folder.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from episodic_agent.core.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class LocationMetrics:
    """Metrics for location recognition."""
    
    total_steps: int = 0
    steps_with_location: int = 0
    steps_with_ground_truth: int = 0
    correct_matches: int = 0
    unique_locations: int = 0
    location_changes: int = 0
    
    @property
    def accuracy(self) -> float:
        """Location accuracy (correct/total with ground truth)."""
        if self.steps_with_ground_truth == 0:
            return 0.0
        return self.correct_matches / self.steps_with_ground_truth
    
    @property
    def coverage(self) -> float:
        """Location coverage (steps with location/total)."""
        if self.total_steps == 0:
            return 0.0
        return self.steps_with_location / self.total_steps
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_steps": self.total_steps,
            "steps_with_location": self.steps_with_location,
            "steps_with_ground_truth": self.steps_with_ground_truth,
            "correct_matches": self.correct_matches,
            "unique_locations": self.unique_locations,
            "location_changes": self.location_changes,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
        }


@dataclass
class EntityMetrics:
    """Metrics for entity recognition."""
    
    total_entities_seen: int = 0
    total_entities_ground_truth: int = 0
    correct_matches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    unique_entities: int = 0
    guid_match_rate: float = 0.0
    
    @property
    def precision(self) -> float:
        """Precision (correct/predicted)."""
        total_predicted = self.correct_matches + self.false_positives
        if total_predicted == 0:
            return 0.0
        return self.correct_matches / total_predicted
    
    @property
    def recall(self) -> float:
        """Recall (correct/ground truth)."""
        total_actual = self.correct_matches + self.false_negatives
        if total_actual == 0:
            return 0.0
        return self.correct_matches / total_actual
    
    @property
    def f1_score(self) -> float:
        """F1 score."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entities_seen": self.total_entities_seen,
            "total_entities_ground_truth": self.total_entities_ground_truth,
            "correct_matches": self.correct_matches,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "unique_entities": self.unique_entities,
            "guid_match_rate": self.guid_match_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


@dataclass
class EventMetrics:
    """Metrics for event detection."""
    
    total_deltas_detected: int = 0
    total_events_detected: int = 0
    total_events_labeled: int = 0
    total_events_recognized: int = 0
    events_with_ground_truth: int = 0
    correct_event_matches: int = 0
    state_change_accuracy: float = 0.0
    
    @property
    def detection_rate(self) -> float:
        """Event detection rate."""
        if self.events_with_ground_truth == 0:
            return 0.0
        return self.correct_event_matches / self.events_with_ground_truth
    
    @property
    def labeling_rate(self) -> float:
        """Rate of events that got labels."""
        if self.total_events_detected == 0:
            return 0.0
        return self.total_events_labeled / self.total_events_detected
    
    @property
    def recognition_rate(self) -> float:
        """Rate of events recognized from prior learning."""
        if self.total_events_detected == 0:
            return 0.0
        return self.total_events_recognized / self.total_events_detected
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_deltas_detected": self.total_deltas_detected,
            "total_events_detected": self.total_events_detected,
            "total_events_labeled": self.total_events_labeled,
            "total_events_recognized": self.total_events_recognized,
            "events_with_ground_truth": self.events_with_ground_truth,
            "correct_event_matches": self.correct_event_matches,
            "state_change_accuracy": self.state_change_accuracy,
            "detection_rate": self.detection_rate,
            "labeling_rate": self.labeling_rate,
            "recognition_rate": self.recognition_rate,
        }


@dataclass
class RateMetrics:
    """Rate-based metrics over time."""
    
    total_duration_seconds: float = 0.0
    total_steps: int = 0
    total_episodes: int = 0
    total_questions: int = 0
    
    @property
    def steps_per_minute(self) -> float:
        """Steps per minute."""
        if self.total_duration_seconds == 0:
            return 0.0
        return self.total_steps / (self.total_duration_seconds / 60)
    
    @property
    def episodes_per_minute(self) -> float:
        """Episodes per minute."""
        if self.total_duration_seconds == 0:
            return 0.0
        return self.total_episodes / (self.total_duration_seconds / 60)
    
    @property
    def questions_per_minute(self) -> float:
        """Questions per minute."""
        if self.total_duration_seconds == 0:
            return 0.0
        return self.total_questions / (self.total_duration_seconds / 60)
    
    @property
    def avg_episode_duration(self) -> float:
        """Average episode duration in seconds."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_duration_seconds / self.total_episodes
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_duration_seconds": self.total_duration_seconds,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "total_questions": self.total_questions,
            "steps_per_minute": self.steps_per_minute,
            "episodes_per_minute": self.episodes_per_minute,
            "questions_per_minute": self.questions_per_minute,
            "avg_episode_duration": self.avg_episode_duration,
        }


@dataclass
class MemoryGrowthMetrics:
    """Metrics for memory growth over time."""
    
    final_node_count: int = 0
    final_edge_count: int = 0
    final_episode_count: int = 0
    
    # Growth over time (sampled at intervals)
    node_count_timeline: list[tuple[int, int]] = field(default_factory=list)
    edge_count_timeline: list[tuple[int, int]] = field(default_factory=list)
    episode_count_timeline: list[tuple[int, int]] = field(default_factory=list)
    
    # By type
    nodes_by_type: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_node_count": self.final_node_count,
            "final_edge_count": self.final_edge_count,
            "final_episode_count": self.final_episode_count,
            "node_count_timeline": self.node_count_timeline,
            "edge_count_timeline": self.edge_count_timeline,
            "episode_count_timeline": self.episode_count_timeline,
            "nodes_by_type": self.nodes_by_type,
        }


@dataclass
class PredictionMetrics:
    """Metrics for prediction and prediction errors."""
    
    total_predictions: int = 0
    total_prediction_errors: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    avg_error_magnitude: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "total_prediction_errors": self.total_prediction_errors,
            "errors_by_type": self.errors_by_type,
            "avg_error_magnitude": self.avg_error_magnitude,
        }


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval system."""
    
    total_retrievals: int = 0
    avg_cues_per_retrieval: float = 0.0
    avg_nodes_activated: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_retrieval_certainty: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_retrievals": self.total_retrievals,
            "avg_cues_per_retrieval": self.avg_cues_per_retrieval,
            "avg_nodes_activated": self.avg_nodes_activated,
            "avg_retrieval_time_ms": self.avg_retrieval_time_ms,
            "avg_retrieval_certainty": self.avg_retrieval_certainty,
        }


class MetricsCollector:
    """Collects and computes metrics from a run.
    
    Can collect metrics from:
    - Live orchestrator state
    - Log files (JSONL)
    - Module statistics
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self.location = LocationMetrics()
        self.entity = EntityMetrics()
        self.event = EventMetrics()
        self.rates = RateMetrics()
        self.memory = MemoryGrowthMetrics()
        self.prediction = PredictionMetrics()
        self.retrieval = RetrievalMetrics()
        
        # Internal tracking
        self._step_locations: list[str] = []
        self._step_ground_truth: list[str | None] = []
        self._seen_entities: set[str] = set()
        self._seen_locations: set[str] = set()
        
        # Timeline sampling
        self._sample_interval = 50  # Sample every N steps

    def collect(
        self,
        orchestrator: "AgentOrchestrator | None" = None,
        modules: dict[str, Any] | None = None,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Collect all metrics.
        
        Args:
            orchestrator: Optional orchestrator for live state.
            modules: Optional module dict for statistics.
            run_dir: Optional run directory for log files.
            
        Returns:
            Complete metrics dictionary.
        """
        # Collect from log file if available
        if run_dir:
            log_path = run_dir / "run.jsonl"
            if log_path.exists():
                self._collect_from_log(log_path)
        
        # Collect from modules
        if modules:
            self._collect_from_modules(modules)
        
        # Collect from orchestrator
        if orchestrator:
            self._collect_from_orchestrator(orchestrator)
        
        # Build final metrics
        return self.to_dict()

    def _collect_from_log(self, log_path: Path) -> None:
        """Collect metrics from run log file.
        
        Args:
            log_path: Path to run.jsonl file.
        """
        first_timestamp = None
        last_timestamp = None
        prev_location = None
        
        step_count = 0
        
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Parse timestamp
                ts_str = data.get("timestamp")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if first_timestamp is None:
                            first_timestamp = ts
                        last_timestamp = ts
                    except ValueError:
                        pass
                
                # Track steps
                step_count += 1
                self.location.total_steps = step_count
                
                # Location metrics
                location = data.get("location_label", "unknown")
                if location and location != "unknown":
                    self.location.steps_with_location += 1
                    self._seen_locations.add(location)
                    
                    if location != prev_location and prev_location is not None:
                        self.location.location_changes += 1
                    prev_location = location
                
                # Entity metrics
                entity_count = data.get("entity_count", 0)
                self.entity.total_entities_seen += entity_count
                
                # Event metrics from step result
                if "events_detected_total" in data:
                    self.event.total_events_detected = data["events_detected_total"]
                if "events_labeled_total" in data:
                    self.event.total_events_labeled = data["events_labeled_total"]
                if "events_recognized_total" in data:
                    self.event.total_events_recognized = data["events_recognized_total"]
                if "deltas_total" in data:
                    self.event.total_deltas_detected = data["deltas_total"]
                if "questions_asked_total" in data:
                    self.rates.total_questions = data["questions_asked_total"]
                
                # Episode count
                episode_count = data.get("episode_count", 0)
                self.rates.total_episodes = episode_count
                
                # Timeline sampling
                if step_count % self._sample_interval == 0:
                    self.memory.episode_count_timeline.append((step_count, episode_count))
        
        # Calculate duration
        if first_timestamp and last_timestamp:
            self.rates.total_duration_seconds = (
                last_timestamp - first_timestamp
            ).total_seconds()
        
        self.rates.total_steps = step_count
        self.location.unique_locations = len(self._seen_locations)

    def _collect_from_modules(self, modules: dict[str, Any]) -> None:
        """Collect metrics from module instances.
        
        Args:
            modules: Dictionary of module instances.
        """
        # Graph store metrics
        graph_store = modules.get("graph_store")
        if graph_store:
            if hasattr(graph_store, "node_count"):
                self.memory.final_node_count = graph_store.node_count()
            if hasattr(graph_store, "edge_count"):
                self.memory.final_edge_count = graph_store.edge_count()
            
            # Nodes by type
            if hasattr(graph_store, "get_nodes_by_type"):
                from episodic_agent.schemas import NodeType
                for node_type in NodeType:
                    nodes = graph_store.get_nodes_by_type(node_type)
                    self.memory.nodes_by_type[node_type.value] = len(nodes)
        
        # Episode store metrics
        episode_store = modules.get("episode_store")
        if episode_store:
            if hasattr(episode_store, "count"):
                self.memory.final_episode_count = episode_store.count()
        
        # Event resolver metrics
        event_resolver = modules.get("event_resolver")
        if event_resolver:
            if hasattr(event_resolver, "events_detected"):
                self.event.total_events_detected = event_resolver.events_detected
            if hasattr(event_resolver, "events_labeled"):
                self.event.total_events_labeled = event_resolver.events_labeled
            if hasattr(event_resolver, "events_recognized"):
                self.event.total_events_recognized = event_resolver.events_recognized
            if hasattr(event_resolver, "deltas_detected"):
                self.event.total_deltas_detected = event_resolver.deltas_detected
            if hasattr(event_resolver, "questions_asked"):
                self.rates.total_questions = event_resolver.questions_asked
        
        # Retriever metrics
        retriever = modules.get("retriever")
        if retriever:
            if hasattr(retriever, "retrievals_performed"):
                self.retrieval.total_retrievals = retriever.retrievals_performed
            if hasattr(retriever, "_total_cues_used") and retriever.retrievals_performed > 0:
                self.retrieval.avg_cues_per_retrieval = (
                    retriever._total_cues_used / retriever.retrievals_performed
                )
            if hasattr(retriever, "_total_nodes_activated") and retriever.retrievals_performed > 0:
                self.retrieval.avg_nodes_activated = (
                    retriever._total_nodes_activated / retriever.retrievals_performed
                )
        
        # Prediction module metrics
        # (Would need to track in prediction module)
        
        # Boundary detector metrics
        boundary = modules.get("boundary_detector")
        if boundary and hasattr(boundary, "boundary_counts"):
            # Could add boundary-specific metrics

            pass

    def _collect_from_orchestrator(
        self,
        orchestrator: "AgentOrchestrator",
    ) -> None:
        """Collect metrics from orchestrator state.
        
        Args:
            orchestrator: Agent orchestrator.
        """
        self.rates.total_steps = orchestrator.step_number
        self.rates.total_episodes = orchestrator.episode_count

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to dictionary.
        
        Returns:
            Complete metrics dictionary.
        """
        return {
            "collected_at": datetime.now().isoformat(),
            "location": self.location.to_dict(),
            "entity": self.entity.to_dict(),
            "event": self.event.to_dict(),
            "rates": self.rates.to_dict(),
            "memory": self.memory.to_dict(),
            "prediction": self.prediction.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> dict[str, Any]:
        """Build a summary of key metrics.
        
        Returns:
            Summary dictionary.
        """
        return {
            "total_steps": self.rates.total_steps,
            "total_episodes": self.rates.total_episodes,
            "total_duration_seconds": self.rates.total_duration_seconds,
            "unique_locations": self.location.unique_locations,
            "location_coverage": self.location.coverage,
            "episodes_per_minute": self.rates.episodes_per_minute,
            "questions_per_minute": self.rates.questions_per_minute,
            "events_detected": self.event.total_events_detected,
            "events_labeled": self.event.total_events_labeled,
            "final_node_count": self.memory.final_node_count,
            "final_edge_count": self.memory.final_edge_count,
        }


def compute_metrics(
    run_dir: Path,
    orchestrator: "AgentOrchestrator | None" = None,
    modules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function to compute and save metrics.
    
    Args:
        run_dir: Run directory.
        orchestrator: Optional orchestrator.
        modules: Optional modules dict.
        
    Returns:
        Metrics dictionary.
    """
    collector = MetricsCollector()
    metrics = collector.collect(
        orchestrator=orchestrator,
        modules=modules,
        run_dir=run_dir,
    )
    
    # Save to file
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    return metrics
