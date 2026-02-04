"""Event resolver for state-change detection from deltas.

Creates event candidates from detected deltas:
- State changes
- Appearance/disappearance events
- Movement events

Integrates with graph memory for event node persistence and
dialog manager for event labeling when patterns are unknown.

ARCHITECTURAL INVARIANT: No predefined semantic mappings.
Event types are learned from user interaction, not hardcoded.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from episodic_agent.core.interfaces import EventResolver
from episodic_agent.modules.delta_detector import DeltaDetector
from episodic_agent.schemas.events import (
    Delta,
    EventCandidate,
    DELTA_TYPE_NEW,
    DELTA_TYPE_MISSING,
    DELTA_TYPE_MOVED,
    DELTA_TYPE_CHANGED,
    EVENT_TYPE_UNKNOWN,
)
from episodic_agent.schemas.graph import (
    GraphEdge, 
    GraphNode, 
    NODE_TYPE_EVENT,
    NODE_TYPE_ENTITY,
    EDGE_TYPE_IN_EVENT,
    EDGE_TYPE_OCCURRED_IN,
    EDGE_TYPE_TRIGGERED_BY,
    EDGE_TYPE_INVOLVES,
)
from episodic_agent.utils.config import (
    CONFIDENCE_T_HIGH,
    DEFAULT_EMBEDDING_DIM,
    EVENT_RECOGNITION_THRESHOLD,
    LEARNED_EVENT_CONFIDENCE_BOOST,
)

if TYPE_CHECKING:
    from episodic_agent.memory.graph_store import LabeledGraphStore
    from episodic_agent.modules.dialog import DialogManager
    from episodic_agent.schemas import ActiveContextFrame, Percept

logger = logging.getLogger(__name__)


# =============================================================================
# ARCHITECTURAL INVARIANT: No predefined state-to-event mappings
# =============================================================================
# 
# REMOVED: STATE_TRANSITION_MAP that mapped state changes to event types.
# All event semantics are now LEARNED from user interaction.
# 
# Instead of hardcoding "closed -> open = OPENED", the system:
#   1. Detects the structural change (state changed from X to Y)
#   2. Creates an unknown event candidate
#   3. Asks user to label it (or matches against learned patterns)
#   4. Stores the learned pattern for future recognition
# =============================================================================


class EventResolverStateChange(EventResolver):
    """Event resolver that detects state-change events from deltas.
    
    Features:
    - Integrates DeltaDetector for change detection
    - Maps state transitions to event types
    - Creates event candidates with pre/post signatures
    - Persists learned events to graph memory
    - Prompts for labels on unknown event patterns
    - Recognizes previously learned events automatically
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        dialog_manager: "DialogManager",
        delta_detector: DeltaDetector | None = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        auto_label_events: bool = False,
        prompt_for_unknown_events: bool = True,
    ) -> None:
        """Initialize the event resolver.
        
        Args:
            graph_store: Graph store for event persistence.
            dialog_manager: Dialog manager for label requests.
            delta_detector: Optional delta detector (creates one if None).
            embedding_dim: Embedding dimension for event nodes.
            auto_label_events: If True, auto-generate event labels.
            prompt_for_unknown_events: If True, prompt for unknown events.
        """
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        self._delta_detector = delta_detector or DeltaDetector()
        self._embedding_dim = embedding_dim
        self._auto_label_events = auto_label_events
        self._prompt_for_unknown_events = prompt_for_unknown_events
        
        # Cache of learned event patterns (signature -> node_id)
        self._event_pattern_cache: dict[str, str] = {}
        
        # Statistics
        self._events_detected = 0
        self._events_labeled = 0
        self._events_recognized = 0
        self._questions_asked = 0

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[dict]:
        """Detect events from perception changes.
        
        Runs delta detection, then creates event candidates from
        detected deltas. Returns events as dicts for ACF storage.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            List of event dictionaries for ACF storage.
        """
        # Run delta detection
        deltas = self._delta_detector.detect(percept, acf)
        
        # Store deltas in ACF extras for persistence
        if "deltas" not in acf.extras:
            acf.extras["deltas"] = []
        
        for delta in deltas:
            acf.extras["deltas"].append(self._delta_to_dict(delta))
        
        # Create event candidates from deltas
        event_dicts: list[dict] = []
        
        for delta in deltas:
            event = self._delta_to_event(delta, acf)
            if event:
                event_dicts.append(event.to_dict())
                self._events_detected += 1
        
        return event_dicts

    def _delta_to_event(
        self,
        delta: Delta,
        acf: ActiveContextFrame,
    ) -> EventCandidate | None:
        """Convert a delta to an event candidate.
        
        Args:
            delta: The detected delta.
            acf: Current active context frame.
            
        Returns:
            EventCandidate if this delta represents a recognizable event.
        """
        # Use string comparisons instead of enum values
        if delta.delta_type == DELTA_TYPE_CHANGED:
            return self._handle_state_change(delta, acf)
        elif delta.delta_type == DELTA_TYPE_NEW:
            return self._handle_appearance(delta, acf)
        elif delta.delta_type == DELTA_TYPE_MISSING:
            return self._handle_disappearance(delta, acf)
        elif delta.delta_type == DELTA_TYPE_MOVED:
            return self._handle_movement(delta, acf)
        
        return None

    def _handle_state_change(
        self,
        delta: Delta,
        acf: ActiveContextFrame,
    ) -> EventCandidate | None:
        """Create event from state change delta.
        
        Args:
            delta: State change delta.
            acf: Current context.
            
        Returns:
            Event candidate for state change.
        """
        if not delta.pre_state or not delta.post_state:
            return None
        
        # Normalize states for matching
        pre_state = delta.pre_state.lower()
        post_state = delta.post_state.lower()
        
        # Build state signature (structural, not semantic)
        # REMOVED: entity_category from signature - backend learns semantics
        pre_signature = f"state:{pre_state}"
        post_signature = f"state:{post_state}"
        pattern_signature = f"{pre_signature}->{post_signature}"
        
        # REMOVED: STATE_TRANSITION_MAP lookup
        # Event types are now LEARNED from user interaction, not predefined
        # Start with unknown type - will be labeled by user or matched to learned patterns
        event_type = EVENT_TYPE_UNKNOWN
        
        # Check if we've seen this pattern before (learned from user)
        known_event_node = self._find_known_event(pattern_signature)
        
        if known_event_node:
            # Recognized event - use learned label
            return self._create_recognized_event(
                delta, event_type, known_event_node,
                pre_signature, post_signature, pattern_signature
            )
        else:
            # New event pattern - will prompt user for label
            return self._create_new_event(
                delta, event_type, acf,
                pre_signature, post_signature, pattern_signature
            )

    def _handle_appearance(
        self,
        delta: Delta,
        acf: ActiveContextFrame,
    ) -> EventCandidate:
        """Create event from entity appearance delta.
        
        Args:
            delta: New entity delta.
            acf: Current context.
            
        Returns:
            Event candidate for appearance.
        """
        # REMOVED: entity_category - backend learns semantics from user
        pattern_signature = "entity:appeared"
        
        # Generate structural label - user will provide semantic label
        label = f"{delta.entity_label or 'entity'}_appeared" if delta.entity_label else "unknown_appeared"
        
        event = EventCandidate(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type="appeared",  # Structural type, user may relabel
            label=label,
            involved_entity_ids=[delta.entity_id] if delta.entity_id else [],
            involved_entity_labels=[delta.entity_label] if delta.entity_label else [],
            post_state_signature=pattern_signature,
            source_delta_ids=[delta.delta_id],
            location_label=delta.location_label,
            step_number=delta.step_number,
            confidence=delta.confidence * 0.9,
            evidence=delta.evidence + [f"Entity appeared: {delta.entity_label}"],
        )
        
        logger.info(f"Event detected: {label}")
        self._dialog_manager.notify(f"🎬 Event: {label}")
        
        return event

    def _handle_disappearance(
        self,
        delta: Delta,
        acf: ActiveContextFrame,
    ) -> EventCandidate:
        """Create event from entity disappearance delta.
        
        Args:
            delta: Missing entity delta.
            acf: Current context.
            
        Returns:
            Event candidate for disappearance.
        """
        # REMOVED: entity_category - backend learns semantics from user
        pattern_signature = "entity:disappeared"
        
        label = f"{delta.entity_label or 'entity'}_disappeared" if delta.entity_label else "unknown_disappeared"
        
        event = EventCandidate(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type="disappeared",  # Structural type, user may relabel
            label=label,
            involved_entity_ids=[delta.entity_id] if delta.entity_id else [],
            involved_entity_labels=[delta.entity_label] if delta.entity_label else [],
            pre_state_signature=pattern_signature,
            source_delta_ids=[delta.delta_id],
            location_label=delta.location_label,
            step_number=delta.step_number,
            confidence=delta.confidence * 0.85,
            evidence=delta.evidence + [f"Entity disappeared: {delta.entity_label}"],
        )
        
        logger.info(f"Event detected: {label}")
        self._dialog_manager.notify(f"🎬 Event: {label}")
        
        return event

    def _handle_movement(
        self,
        delta: Delta,
        acf: ActiveContextFrame,
    ) -> EventCandidate:
        """Create event from entity movement delta.
        
        Args:
            delta: Movement delta.
            acf: Current context.
            
        Returns:
            Event candidate for movement.
        """
        # REMOVED: entity_category - backend learns semantics from user
        label = f"{delta.entity_label or 'entity'}_moved" if delta.entity_label else "unknown_moved"
        
        event = EventCandidate(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type="moved",  # Structural type, user may relabel
            label=label,
            involved_entity_ids=[delta.entity_id] if delta.entity_id else [],
            involved_entity_labels=[delta.entity_label] if delta.entity_label else [],
            source_delta_ids=[delta.delta_id],
            location_label=delta.location_label,
            step_number=delta.step_number,
            confidence=delta.confidence * 0.9,
            evidence=delta.evidence + [
                f"Entity moved: {delta.entity_label}",
                f"Distance: {delta.position_delta:.2f}",
            ],
            extras={
                "position_delta": delta.position_delta,
                "pre_position": delta.pre_position,
                "post_position": delta.post_position,
            },
        )
        
        logger.info(f"Event detected: {label} (moved {delta.position_delta:.2f})")
        
        return event

    def _find_known_event(self, pattern_signature: str) -> GraphNode | None:
        """Look up a known event by pattern signature.
        
        Args:
            pattern_signature: The pre->post state signature.
            
        Returns:
            Event GraphNode if pattern is known, None otherwise.
        """
        # Check cache first
        if pattern_signature in self._event_pattern_cache:
            node_id = self._event_pattern_cache[pattern_signature]
            return self._graph_store.get_node(node_id)
        
        # Search graph store for event nodes with this signature
        event_nodes = self._graph_store.get_nodes_by_type(NODE_TYPE_EVENT)
        
        for node in event_nodes:
            node_signature = node.extras.get("pattern_signature")
            if node_signature == pattern_signature:
                self._event_pattern_cache[pattern_signature] = node.node_id
                return node
        
        return None

    def _create_recognized_event(
        self,
        delta: Delta,
        event_type: EventType,
        known_node: GraphNode,
        pre_signature: str,
        post_signature: str,
        pattern_signature: str,
    ) -> EventCandidate:
        """Create event candidate for a recognized pattern.
        
        Args:
            delta: The source delta.
            event_type: Determined event type.
            known_node: The known event node.
            pre_signature: Pre-state signature.
            post_signature: Post-state signature.
            pattern_signature: Full pattern signature.
            
        Returns:
            Event candidate with recognized label.
        """
        # Update access stats on known node
        self._graph_store.update_node_access(known_node.node_id)
        self._events_recognized += 1
        
        # Use the learned label
        label = known_node.label
        confidence = min(1.0, delta.confidence + LEARNED_EVENT_CONFIDENCE_BOOST)
        
        event = EventCandidate(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            label=label,
            labels=known_node.labels,
            is_learned=True,
            involved_entity_ids=[delta.entity_id] if delta.entity_id else [],
            involved_entity_labels=[delta.entity_label] if delta.entity_label else [],
            pre_state_signature=pre_signature,
            post_state_signature=post_signature,
            source_delta_ids=[delta.delta_id],
            location_label=delta.location_label,
            step_number=delta.step_number,
            confidence=confidence,
            evidence=delta.evidence + [
                f"Recognized event pattern: {pattern_signature}",
                f"Matched known event: {label}",
            ],
            extras={
                "event_node_id": known_node.node_id,
                "pattern_signature": pattern_signature,
            },
        )
        
        logger.info(f"Recognized event: {label} (confidence: {confidence:.2f})")
        self._dialog_manager.notify(f"🎬 {label} ({delta.entity_label})")
        
        return event

    def _create_new_event(
        self,
        delta: Delta,
        event_type: EventType,
        acf: ActiveContextFrame,
        pre_signature: str,
        post_signature: str,
        pattern_signature: str,
    ) -> EventCandidate:
        """Create event candidate for a new pattern.
        
        May prompt user for label if enabled.
        
        Args:
            delta: The source delta.
            event_type: Determined event type.
            acf: Current context.
            pre_signature: Pre-state signature.
            post_signature: Post-state signature.
            pattern_signature: Full pattern signature.
            
        Returns:
            Event candidate with generated or user-provided label.
        """
        # Generate default label from entity (no predefined event type mapping)
        # REMOVED: EventType enum comparisons - all labels are learned
        entity_part = delta.entity_label or "entity"
        
        # Create structural label based on state transition
        # Semantic meaning will be learned from user interaction
        default_label = f"{entity_part}_{delta.pre_state}_to_{delta.post_state}"
        
        # Ask user for label if enabled
        if self._prompt_for_unknown_events and not self._auto_label_events:
            self._dialog_manager.notify(
                f"🆕 New event pattern detected: {delta.entity_label} "
                f"({delta.pre_state} → {delta.post_state})"
            )
            
            suggestions = [
                default_label,
                f"{delta.pre_state}_to_{delta.post_state}",
                pattern_signature.replace("->", "_to_"),
            ]
            
            label = self._dialog_manager.ask_label(
                f"What should I call this event? ({delta.pre_state} → {delta.post_state})",
                suggestions=suggestions,
            )
            self._questions_asked += 1
        else:
            label = default_label
        
        # Create event node in graph for future recognition
        event_node = self._create_event_node(
            label=label,
            event_type=event_type,
            pre_signature=pre_signature,
            post_signature=post_signature,
            pattern_signature=pattern_signature,
            delta=delta,
        )
        
        self._events_labeled += 1
        
        event = EventCandidate(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,
            label=label,
            is_learned=True,
            involved_entity_ids=[delta.entity_id] if delta.entity_id else [],
            involved_entity_labels=[delta.entity_label] if delta.entity_label else [],
            pre_state_signature=pre_signature,
            post_state_signature=post_signature,
            source_delta_ids=[delta.delta_id],
            location_label=delta.location_label,
            step_number=delta.step_number,
            confidence=delta.confidence,
            evidence=delta.evidence + [
                f"New event pattern: {pattern_signature}",
                f"Learned label: {label}",
            ],
            extras={
                "event_node_id": event_node.node_id,
                "pattern_signature": pattern_signature,
            },
        )
        
        logger.info(f"Learned new event: {label}")
        self._dialog_manager.notify(f"✅ Learned event: {label}")
        
        return event

    def _create_event_node(
        self,
        label: str,
        event_type: EventType,
        pre_signature: str,
        post_signature: str,
        pattern_signature: str,
        delta: Delta,
    ) -> GraphNode:
        """Create and persist an event node in the graph.
        
        Args:
            label: Event label.
            event_type: Event type enum.
            pre_signature: Pre-state signature.
            post_signature: Post-state signature.
            pattern_signature: Full pattern signature.
            delta: Source delta.
            
        Returns:
            The created GraphNode.
        """
        node = GraphNode(
            node_id=f"event_{uuid.uuid4().hex[:12]}",
            node_type=NODE_TYPE_EVENT,
            label=label,
            confidence=CONFIDENCE_T_HIGH,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            extras={
                "event_type": event_type,  # Now a string, not enum
                "pre_signature": pre_signature,
                "post_signature": post_signature,
                "pattern_signature": pattern_signature,
                # REMOVED: entity_category - backend learns semantics from user
                "source": "state_change_resolver",
            },
        )
        
        self._graph_store.add_node(node)
        self._event_pattern_cache[pattern_signature] = node.node_id
        
        # Create "involves" edge if we have entity info
        if delta.entity_id:
            # Find entity node
            entity_nodes = self._graph_store.get_nodes_by_type(NODE_TYPE_ENTITY)
            for entity_node in entity_nodes:
                if entity_node.source_id == delta.entity_id:
                    edge = GraphEdge(
                        edge_id=f"edge_{uuid.uuid4().hex[:12]}",
                        edge_type=EDGE_TYPE_INVOLVES,
                        source_node_id=node.node_id,
                        target_node_id=entity_node.node_id,
                        weight=1.0,
                        confidence=CONFIDENCE_T_HIGH,
                        extras={"source": "state_change_resolver"},
                    )
                    self._graph_store.add_edge(edge)
                    break
        
        return node

    def _delta_to_dict(self, delta: Delta) -> dict[str, Any]:
        """Convert a delta to a dictionary for storage."""
        return {
            "delta_id": delta.delta_id,
            "delta_type": delta.delta_type,  # Now a string, not enum
            "timestamp": delta.timestamp.isoformat(),
            "entity_id": delta.entity_id,
            "entity_label": delta.entity_label,
            # REMOVED: entity_category - backend learns semantics from user
            "pre_state": delta.pre_state,
            "post_state": delta.post_state,
            "pre_position": delta.pre_position,
            "post_position": delta.post_position,
            "position_delta": delta.position_delta,
            "location_label": delta.location_label,
            "step_number": delta.step_number,
            "confidence": delta.confidence,
            "evidence": delta.evidence,
        }

    def get_delta_detector(self) -> DeltaDetector:
        """Get the underlying delta detector."""
        return self._delta_detector

    @property
    def events_detected(self) -> int:
        """Total events detected."""
        return self._events_detected

    @property
    def events_labeled(self) -> int:
        """Number of new event labels learned."""
        return self._events_labeled

    @property
    def events_recognized(self) -> int:
        """Number of events recognized from learned patterns."""
        return self._events_recognized

    @property
    def questions_asked(self) -> int:
        """Number of labeling questions asked."""
        return self._questions_asked

    @property
    def deltas_detected(self) -> int:
        """Total deltas detected (from delta detector)."""
        return self._delta_detector.total_deltas_detected
