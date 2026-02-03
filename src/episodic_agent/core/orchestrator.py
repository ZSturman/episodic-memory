"""Agent orchestrator - runs the cognitive loop.

The orchestrator enforces the strict step order:
1. Sensor → get frame
2. Perception → process frame
3. ACF update → incorporate percept
4. Location resolution → where am I
5. Entity resolution → what's here
6. Event resolution → what changed
7. Retrieval → query memory
8. Boundary check → should freeze episode
9. (Optional) Freeze episode
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from episodic_agent.schemas import (
    ActiveContextFrame,
    Episode,
    StepResult,
)

if TYPE_CHECKING:
    from episodic_agent.core.interfaces import (
        ACFBuilder,
        BoundaryDetector,
        DialogManager,
        EntityResolver,
        EpisodeStore,
        EventResolver,
        GraphStore,
        LocationResolver,
        PerceptionModule,
        Retriever,
        SensorProvider,
    )


class AgentOrchestrator:
    """Orchestrates the agent's cognitive loop.
    
    The step() method enforces an immutable call order that cannot
    be reordered: sensor → perception → ACF → location → entities →
    events → retrieval → boundary → freeze.
    
    All dependencies are injected via constructor, allowing modules
    to be swapped without changing the orchestrator.
    """

    def __init__(
        self,
        sensor: SensorProvider,
        perception: PerceptionModule,
        acf_builder: ACFBuilder,
        location_resolver: LocationResolver,
        entity_resolver: EntityResolver,
        event_resolver: EventResolver,
        retriever: Retriever,
        boundary_detector: BoundaryDetector,
        dialog_manager: DialogManager,
        episode_store: EpisodeStore,
        graph_store: GraphStore,
        run_id: str,
    ) -> None:
        """Initialize the orchestrator with all required modules.
        
        Args:
            sensor: Provider for sensor frames.
            perception: Module for processing frames into percepts.
            acf_builder: Builder for active context frames.
            location_resolver: Resolver for location identification.
            entity_resolver: Resolver for entity identification.
            event_resolver: Resolver for event detection.
            retriever: Module for memory retrieval.
            boundary_detector: Detector for episode boundaries.
            dialog_manager: Manager for user interactions.
            episode_store: Storage for frozen episodes.
            graph_store: Storage for graph memory.
            run_id: Identifier for this run (used in logging).
        """
        self._sensor = sensor
        self._perception = perception
        self._acf_builder = acf_builder
        self._location_resolver = location_resolver
        self._entity_resolver = entity_resolver
        self._event_resolver = event_resolver
        self._retriever = retriever
        self._boundary_detector = boundary_detector
        self._dialog_manager = dialog_manager
        self._episode_store = episode_store
        self._graph_store = graph_store
        
        self._run_id = run_id
        self._step_number = 0
        self._acf: ActiveContextFrame | None = None
        self._episode_start_time: datetime | None = None

    @property
    def acf(self) -> ActiveContextFrame | None:
        """Get the current active context frame."""
        return self._acf

    @property
    def step_number(self) -> int:
        """Get the current step number."""
        return self._step_number

    @property
    def episode_count(self) -> int:
        """Get the number of frozen episodes."""
        return self._episode_store.count()

    def step(self) -> StepResult:
        """Execute one step of the cognitive loop.
        
        The order of operations is fixed and cannot be changed:
        1. sensor.get_frame()
        2. perception.process(frame)
        3. acf_builder.update_acf(acf, percept)
        4. location_resolver.resolve(percept, acf)
        5. entity_resolver.resolve(percept, acf)
        6. event_resolver.resolve(percept, acf)
        7. retriever.retrieve(acf)
        8. boundary_detector.check(acf)
        9. freeze_episode() if boundary triggered
        
        Returns:
            StepResult containing all information for logging.
        """
        self._step_number += 1
        
        # Initialize ACF if needed
        if self._acf is None:
            self._acf = self._acf_builder.create_acf()
            self._episode_start_time = datetime.now()
        
        # === STEP 1: SENSOR ===
        frame = self._sensor.get_frame()
        
        # === STEP 2: PERCEPTION ===
        percept = self._perception.process(frame)
        
        # === STEP 3: ACF UPDATE ===
        self._acf = self._acf_builder.update_acf(self._acf, percept)
        self._acf.step_count += 1
        self._acf.touch()
        
        # === STEP 4: LOCATION RESOLUTION ===
        location_label, location_confidence = self._location_resolver.resolve(
            percept, self._acf
        )
        self._acf.location_label = location_label
        self._acf.location_confidence = location_confidence
        
        # === STEP 5: ENTITY RESOLUTION ===
        entities = self._entity_resolver.resolve(percept, self._acf)
        self._acf.entities = entities
        
        # === STEP 6: EVENT RESOLUTION ===
        events = self._event_resolver.resolve(percept, self._acf)
        self._acf.events.extend(events)
        
        # === STEP 7: RETRIEVAL ===
        # Retrieval is called but results aren't used in Phase 1
        _ = self._retriever.retrieve(self._acf)
        
        # === STEP 8: BOUNDARY CHECK ===
        should_freeze, boundary_reason = self._boundary_detector.check(self._acf)
        
        # === STEP 9: FREEZE EPISODE (if triggered) ===
        if should_freeze:
            self._freeze_episode(boundary_reason or "unknown")
        
        # Gather Phase 5 metrics from event resolver
        delta_count = 0
        deltas_total = 0
        events_detected_total = 0
        events_labeled_total = 0
        events_recognized_total = 0
        questions_asked_total = 0
        
        # Get deltas from current step (stored in ACF extras by event resolver)
        if "deltas" in self._acf.extras:
            deltas_total = len(self._acf.extras["deltas"])
        deltas_total += len(self._acf.deltas)
        
        # Get event resolver stats if available
        if hasattr(self._event_resolver, "events_detected"):
            events_detected_total = self._event_resolver.events_detected
        if hasattr(self._event_resolver, "events_labeled"):
            events_labeled_total = self._event_resolver.events_labeled
        if hasattr(self._event_resolver, "events_recognized"):
            events_recognized_total = self._event_resolver.events_recognized
        if hasattr(self._event_resolver, "questions_asked"):
            questions_asked_total = self._event_resolver.questions_asked
        if hasattr(self._event_resolver, "deltas_detected"):
            delta_count = self._event_resolver.deltas_detected
        
        # Build step result for logging
        return StepResult(
            run_id=self._run_id,
            timestamp=datetime.now(),
            step_number=self._step_number,
            frame_id=frame.frame_id,
            acf_id=self._acf.acf_id,
            location_label=self._acf.location_label,
            location_confidence=self._acf.location_confidence,
            entity_count=len(self._acf.entities),
            event_count=len(self._acf.events),
            episode_count=self._episode_store.count(),
            boundary_triggered=should_freeze,
            boundary_reason=boundary_reason,
            delta_count=delta_count,
            deltas_total=deltas_total,
            events_detected_total=events_detected_total,
            events_labeled_total=events_labeled_total,
            events_recognized_total=events_recognized_total,
            questions_asked_total=questions_asked_total,
        )

    def _freeze_episode(self, reason: str) -> Episode:
        """Freeze the current ACF into an episode.
        
        Creates an immutable Episode from the current ACF state,
        stores it, and resets for a new episode.
        
        Args:
            reason: Why the boundary was triggered.
            
        Returns:
            The frozen Episode.
        """
        if self._acf is None:
            raise RuntimeError("Cannot freeze episode: no active ACF")
        
        now = datetime.now()
        
        # Gather deltas from ACF (from extras where event resolver stores them)
        deltas = list(self._acf.deltas)
        if "deltas" in self._acf.extras:
            deltas.extend(self._acf.extras.get("deltas", []))
        
        episode = Episode(
            episode_id=f"ep_{uuid.uuid4().hex[:12]}",
            created_at=now,
            start_time=self._episode_start_time or self._acf.created_at,
            end_time=now,
            step_count=self._acf.step_count,
            location_label=self._acf.location_label,
            location_confidence=self._acf.location_confidence,
            location_embedding=self._acf.location_embedding,
            entities=list(self._acf.entities),
            events=list(self._acf.events),
            deltas=deltas,
            episode_embedding=None,  # Could compute aggregate embedding
            source_acf_id=self._acf.acf_id,
            boundary_reason=reason,
        )
        
        # Store the episode
        self._episode_store.store(episode)
        
        # Reset ACF for new episode
        self._acf = self._acf_builder.create_acf()
        self._episode_start_time = now
        
        return episode

    def has_more_frames(self) -> bool:
        """Check if more sensor frames are available.
        
        Returns:
            True if the sensor has more frames.
        """
        return self._sensor.has_frames()
