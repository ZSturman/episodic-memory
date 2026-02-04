"""Event Learning Pipeline for emergent event recognition.

ARCHITECTURAL INVARIANT: No predefined semantic event types.
All event semantics are learned from user interaction and stored memory.

The pipeline implements:
1. Event detection from deltas (structural)
2. Pattern matching against learned events
3. Confidence-based action selection:
   - High confidence → auto-accept (recognized event)
   - Medium confidence → confirm with user
   - Low confidence → request new label
4. Event storage with salience weights
5. Graph memory integration

This module replaces hardcoded event type mappings with learned patterns.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

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
    NODE_TYPE_LOCATION,
    EDGE_TYPE_IN_EVENT,
    EDGE_TYPE_OCCURRED_IN,
    EDGE_TYPE_TRIGGERED_BY,
    EDGE_TYPE_INVOLVES,
    EDGE_TYPE_SIMILAR_TO,
)
from episodic_agent.utils.config import (
    CONFIDENCE_T_HIGH,
    CONFIDENCE_T_LOW,
    EVENT_RECOGNITION_THRESHOLD,
    LEARNED_EVENT_CONFIDENCE_BOOST,
)

if TYPE_CHECKING:
    from episodic_agent.memory.graph_store import LabeledGraphStore
    from episodic_agent.modules.dialog import DialogManager
    from episodic_agent.modules.delta_detector import DeltaDetector
    from episodic_agent.schemas import ActiveContextFrame, Percept

logger = logging.getLogger(__name__)


# =============================================================================
# Confidence Action Thresholds
# =============================================================================

# Above this → auto-accept (recognized pattern)
CONFIDENCE_AUTO_ACCEPT: float = 0.8

# Between LOW and AUTO_ACCEPT → confirm with user
CONFIDENCE_CONFIRM: float = 0.5

# Below this → request new label (novel event)
CONFIDENCE_REQUEST_LABEL: float = 0.3


class ConfidenceAction(str, Enum):
    """Action to take based on confidence level."""
    
    AUTO_ACCEPT = "auto_accept"      # High confidence - accept automatically
    CONFIRM = "confirm"              # Medium confidence - ask user to confirm
    REQUEST_LABEL = "request_label"  # Low confidence - ask for new label
    REJECT = "reject"                # Very low confidence - likely noise


@dataclass
class SalienceWeights:
    """Salience weights for event memory links.
    
    ARCHITECTURAL INVARIANT: These weights enable emergent prioritization
    of memories based on learned significance, not predefined importance.
    """
    
    # How much prediction error this event caused (surprise)
    prediction_error_weight: float = 0.0
    
    # User explicitly labeled this event (attention)
    user_label_weight: float = 0.0
    
    # How novel was this event pattern (uniqueness)
    novelty_weight: float = 0.0
    
    # Visual/perceptual distinctiveness
    visual_stimuli_weight: float = 0.0
    
    # Frequency of occurrence (familiarity, inverse novelty)
    frequency_weight: float = 0.0
    
    def total_salience(self) -> float:
        """Compute total salience score."""
        return (
            self.prediction_error_weight * 0.25 +
            self.user_label_weight * 0.30 +
            self.novelty_weight * 0.20 +
            self.visual_stimuli_weight * 0.15 +
            self.frequency_weight * 0.10
        )
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for storage."""
        return {
            "prediction_error_weight": self.prediction_error_weight,
            "user_label_weight": self.user_label_weight,
            "novelty_weight": self.novelty_weight,
            "visual_stimuli_weight": self.visual_stimuli_weight,
            "frequency_weight": self.frequency_weight,
            "total_salience": self.total_salience(),
        }


@dataclass
class LearnedEventPattern:
    """A learned event pattern for future recognition.
    
    Stores the signature and learned label so future occurrences
    can be automatically recognized.
    """
    
    pattern_id: str
    pattern_signature: str
    
    # Learned label from user
    label: str
    event_type: str
    
    # Context where this pattern was learned
    example_entity_labels: list[str] = field(default_factory=list)
    example_location: str | None = None
    
    # Recognition statistics
    times_seen: int = 1
    times_confirmed: int = 0
    last_seen: datetime = field(default_factory=datetime.now)
    
    # Salience of this pattern
    salience: SalienceWeights = field(default_factory=SalienceWeights)
    
    # Graph node ID if stored
    node_id: str | None = None


@dataclass
class EventPipelineResult:
    """Result from processing an event through the pipeline."""
    
    # The event candidate
    event: EventCandidate
    
    # Action taken
    action: ConfidenceAction
    
    # Whether user was prompted
    user_prompted: bool = False
    user_response: str | None = None
    
    # Whether event was stored to graph
    stored_to_graph: bool = False
    graph_node_id: str | None = None
    
    # Matched pattern (if recognized)
    matched_pattern: LearnedEventPattern | None = None
    match_confidence: float = 0.0
    
    # Salience weights assigned
    salience: SalienceWeights = field(default_factory=SalienceWeights)


class EventLearningPipeline:
    """Full event learning pipeline with confidence-based actions.
    
    Implements the emergent event learning loop:
    1. Detect: DeltaDetector finds structural changes
    2. Propose: Create event candidate from delta
    3. Match: Check against learned patterns
    4. Decide: Based on confidence, auto-accept / confirm / label
    5. Learn: Store new pattern if labeled
    6. Link: Create graph edges to entities, location, episode
    
    ARCHITECTURAL INVARIANT: No predefined event semantics.
    All labels come from user interaction or pattern matching
    against previously learned events.
    """
    
    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        dialog_manager: "DialogManager",
        delta_detector: "DeltaDetector | None" = None,
        
        # Confidence thresholds
        confidence_auto_accept: float = CONFIDENCE_AUTO_ACCEPT,
        confidence_confirm: float = CONFIDENCE_CONFIRM,
        confidence_request_label: float = CONFIDENCE_REQUEST_LABEL,
        
        # Behavior flags
        auto_label_novel_events: bool = False,
        prompt_for_confirmation: bool = True,
        store_to_graph: bool = True,
    ) -> None:
        """Initialize the event learning pipeline.
        
        Args:
            graph_store: Graph store for pattern and event persistence.
            dialog_manager: Dialog manager for user interaction.
            delta_detector: Optional delta detector instance.
            confidence_auto_accept: Threshold for auto-accepting events.
            confidence_confirm: Threshold for confirmation prompts.
            confidence_request_label: Threshold for label requests.
            auto_label_novel_events: Auto-generate labels for novel events.
            prompt_for_confirmation: Whether to prompt for medium-confidence.
            store_to_graph: Whether to store events to graph.
        """
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        self._delta_detector = delta_detector
        
        self._confidence_auto_accept = confidence_auto_accept
        self._confidence_confirm = confidence_confirm
        self._confidence_request_label = confidence_request_label
        
        self._auto_label_novel_events = auto_label_novel_events
        self._prompt_for_confirmation = prompt_for_confirmation
        self._store_to_graph = store_to_graph
        
        # Learned patterns cache (signature → pattern)
        self._learned_patterns: dict[str, LearnedEventPattern] = {}
        
        # Load existing patterns from graph
        self._load_patterns_from_graph()
        
        # Statistics
        self._events_detected = 0
        self._events_auto_accepted = 0
        self._events_confirmed = 0
        self._events_labeled = 0
        self._events_rejected = 0
        self._patterns_learned = 0
    
    def _load_patterns_from_graph(self) -> None:
        """Load existing event patterns from graph store."""
        try:
            event_nodes = self._graph_store.get_nodes_by_type(NODE_TYPE_EVENT)
            for node in event_nodes:
                if node.extras and "pattern_signature" in node.extras:
                    signature = node.extras["pattern_signature"]
                    pattern = LearnedEventPattern(
                        pattern_id=node.node_id,
                        pattern_signature=signature,
                        label=node.label,
                        event_type=node.extras.get("event_type", "unknown"),
                        times_seen=node.extras.get("times_seen", 1),
                        times_confirmed=node.extras.get("times_confirmed", 0),
                        node_id=node.node_id,
                    )
                    self._learned_patterns[signature] = pattern
            
            logger.info(f"[EVENT] Loaded {len(self._learned_patterns)} event patterns from graph")
        except Exception as e:
            logger.warning(f"[EVENT] Could not load patterns from graph: {e}")
    
    def process_delta(
        self,
        delta: Delta,
        acf: "ActiveContextFrame",
        prediction_error: float = 0.0,
    ) -> EventPipelineResult | None:
        """Process a single delta through the event pipeline.
        
        Args:
            delta: The detected delta to process.
            acf: Current active context frame.
            prediction_error: Optional prediction error score for salience.
            
        Returns:
            EventPipelineResult if an event was detected, None otherwise.
        """
        self._events_detected += 1
        
        # Step 1: Create event candidate from delta
        event = self._create_event_candidate(delta, acf)
        if not event:
            return None
        
        # Step 2: Try to match against learned patterns
        matched_pattern, match_confidence = self._match_pattern(event)
        
        # Step 3: Determine action based on confidence
        action = self._determine_action(match_confidence, matched_pattern)
        
        # Step 4: Execute action
        result = self._execute_action(
            event, action, matched_pattern, match_confidence, acf, prediction_error
        )
        
        return result
    
    def process_deltas(
        self,
        deltas: list[Delta],
        acf: "ActiveContextFrame",
        prediction_error: float = 0.0,
    ) -> list[EventPipelineResult]:
        """Process multiple deltas through the pipeline.
        
        Args:
            deltas: List of detected deltas.
            acf: Current active context frame.
            prediction_error: Prediction error score for salience.
            
        Returns:
            List of EventPipelineResults for events detected.
        """
        results = []
        for delta in deltas:
            result = self.process_delta(delta, acf, prediction_error)
            if result:
                results.append(result)
        return results
    
    def _create_event_candidate(
        self,
        delta: Delta,
        acf: "ActiveContextFrame",
    ) -> EventCandidate | None:
        """Create event candidate from delta.
        
        Args:
            delta: The detected delta.
            acf: Current context.
            
        Returns:
            EventCandidate or None if delta doesn't warrant an event.
        """
        # Build pattern signature
        if delta.delta_type == DELTA_TYPE_CHANGED:
            if not delta.pre_state or not delta.post_state:
                return None
            pre_sig = f"state:{delta.pre_state.lower()}"
            post_sig = f"state:{delta.post_state.lower()}"
            pattern_signature = f"{pre_sig}->{post_sig}"
            event_type = "state_change"
            
        elif delta.delta_type == DELTA_TYPE_NEW:
            pattern_signature = "entity:appeared"
            event_type = "appeared"
            pre_sig = None
            post_sig = "entity:present"
            
        elif delta.delta_type == DELTA_TYPE_MISSING:
            pattern_signature = "entity:disappeared"
            event_type = "disappeared"
            pre_sig = "entity:present"
            post_sig = None
            
        elif delta.delta_type == DELTA_TYPE_MOVED:
            pattern_signature = "entity:moved"
            event_type = "moved"
            pre_sig = f"position:{delta.pre_position}"
            post_sig = f"position:{delta.post_position}"
            
        else:
            return None
        
        # Create candidate (label will be determined by pipeline)
        return EventCandidate(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type,  # Structural type
            label=EVENT_TYPE_UNKNOWN,  # Will be labeled through pipeline
            involved_entity_ids=[delta.entity_id] if delta.entity_id else [],
            involved_entity_labels=[delta.entity_label] if delta.entity_label else [],
            pre_state_signature=pre_sig,
            post_state_signature=post_sig,
            source_delta_ids=[delta.delta_id],
            location_label=delta.location_label,
            step_number=delta.step_number,
            confidence=delta.confidence,
            evidence=delta.evidence.copy(),
            extras={"pattern_signature": pattern_signature},
        )
    
    def _match_pattern(
        self,
        event: EventCandidate,
    ) -> tuple[LearnedEventPattern | None, float]:
        """Match event against learned patterns.
        
        Args:
            event: The event candidate to match.
            
        Returns:
            Tuple of (matched_pattern, confidence) or (None, 0.0).
        """
        pattern_signature = event.extras.get("pattern_signature", "")
        
        # Exact match first
        if pattern_signature in self._learned_patterns:
            pattern = self._learned_patterns[pattern_signature]
            # Confidence based on how many times we've seen and confirmed this
            base_confidence = min(0.9, 0.6 + pattern.times_confirmed * 0.1)
            return pattern, base_confidence
        
        # Fuzzy match - look for similar patterns
        best_match = None
        best_score = 0.0
        
        for sig, pattern in self._learned_patterns.items():
            score = self._compute_pattern_similarity(pattern_signature, sig)
            if score > best_score and score > 0.5:  # Minimum similarity threshold
                best_match = pattern
                best_score = score
        
        if best_match:
            return best_match, best_score * 0.8  # Reduce confidence for fuzzy match
        
        return None, 0.0
    
    def _compute_pattern_similarity(self, sig1: str, sig2: str) -> float:
        """Compute similarity between two pattern signatures.
        
        Args:
            sig1: First pattern signature.
            sig2: Second pattern signature.
            
        Returns:
            Similarity score in [0, 1].
        """
        # Simple token-based similarity
        tokens1 = set(sig1.lower().replace("->", " ").replace(":", " ").split())
        tokens2 = set(sig2.lower().replace("->", " ").replace(":", " ").split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_action(
        self,
        match_confidence: float,
        matched_pattern: LearnedEventPattern | None,
    ) -> ConfidenceAction:
        """Determine what action to take based on confidence.
        
        ARCHITECTURAL INVARIANT: Confidence-based action selection
        - High → auto-accept (trust the learned pattern)
        - Medium → confirm (ask user to verify)
        - Low → request label (novel event, user must teach)
        
        Args:
            match_confidence: Confidence score from pattern matching.
            matched_pattern: The matched pattern, if any.
            
        Returns:
            Action to take.
        """
        if match_confidence >= self._confidence_auto_accept:
            return ConfidenceAction.AUTO_ACCEPT
        
        if match_confidence >= self._confidence_confirm:
            return ConfidenceAction.CONFIRM
        
        if match_confidence >= self._confidence_request_label or matched_pattern is None:
            return ConfidenceAction.REQUEST_LABEL
        
        return ConfidenceAction.REJECT
    
    def _execute_action(
        self,
        event: EventCandidate,
        action: ConfidenceAction,
        matched_pattern: LearnedEventPattern | None,
        match_confidence: float,
        acf: "ActiveContextFrame",
        prediction_error: float,
    ) -> EventPipelineResult:
        """Execute the determined action.
        
        Args:
            event: The event candidate.
            action: Action to execute.
            matched_pattern: Matched pattern if any.
            match_confidence: Match confidence score.
            acf: Current context.
            prediction_error: Prediction error for salience.
            
        Returns:
            EventPipelineResult with action outcome.
        """
        result = EventPipelineResult(
            event=event,
            action=action,
            matched_pattern=matched_pattern,
            match_confidence=match_confidence,
        )
        
        if action == ConfidenceAction.AUTO_ACCEPT:
            result = self._handle_auto_accept(result, matched_pattern)
            self._events_auto_accepted += 1
            
        elif action == ConfidenceAction.CONFIRM:
            result = self._handle_confirm(result, matched_pattern, acf)
            self._events_confirmed += 1
            
        elif action == ConfidenceAction.REQUEST_LABEL:
            result = self._handle_request_label(result, acf, prediction_error)
            self._events_labeled += 1
            
        else:  # REJECT
            self._events_rejected += 1
            logger.debug(f"[EVENT] Rejected low-confidence event: {event.event_id}")
            return result
        
        # Compute salience
        result.salience = self._compute_salience(
            result, matched_pattern, prediction_error
        )
        
        # Store to graph if enabled
        if self._store_to_graph:
            result = self._store_event_to_graph(result, acf)
        
        return result
    
    def _handle_auto_accept(
        self,
        result: EventPipelineResult,
        matched_pattern: LearnedEventPattern | None,
    ) -> EventPipelineResult:
        """Handle auto-accept action for high-confidence events.
        
        Args:
            result: The pipeline result to update.
            matched_pattern: The matched pattern.
            
        Returns:
            Updated result.
        """
        if matched_pattern:
            # Apply learned label
            result.event.label = matched_pattern.label
            result.event.event_type = matched_pattern.event_type
            result.event.is_learned = True
            result.event.confidence = min(0.95, result.match_confidence + LEARNED_EVENT_CONFIDENCE_BOOST)
            
            # Update pattern statistics
            matched_pattern.times_seen += 1
            matched_pattern.last_seen = datetime.now()
            
            logger.info(
                f"[EVENT] Auto-accepted: '{result.event.label}' "
                f"(confidence={result.event.confidence:.2f})"
            )
            self._dialog_manager.notify(f"🎬 Recognized event: {result.event.label}")
        
        return result
    
    def _handle_confirm(
        self,
        result: EventPipelineResult,
        matched_pattern: LearnedEventPattern | None,
        acf: "ActiveContextFrame",
    ) -> EventPipelineResult:
        """Handle confirm action for medium-confidence events.
        
        Args:
            result: The pipeline result to update.
            matched_pattern: The matched pattern (may be None).
            acf: Current context.
            
        Returns:
            Updated result.
        """
        if not self._prompt_for_confirmation:
            # Treat as auto-accept if prompting disabled
            return self._handle_auto_accept(result, matched_pattern)
        
        # Build confirmation message
        if matched_pattern:
            suggested_label = matched_pattern.label
            message = (
                f"Is this a '{suggested_label}' event? "
                f"(entity: {', '.join(result.event.involved_entity_labels) or 'unknown'})"
            )
        else:
            suggested_label = self._generate_structural_label(result.event)
            message = (
                f"Detected event: {suggested_label}. "
                f"Is this correct?"
            )
        
        result.user_prompted = True
        
        # Ask for confirmation
        confirmed = self._dialog_manager.confirm(message, default=True)
        result.user_response = "confirmed" if confirmed else "rejected"
        
        if confirmed:
            result.event.label = suggested_label
            result.event.is_learned = True
            result.event.confidence = min(0.9, result.match_confidence + 0.1)
            
            if matched_pattern:
                matched_pattern.times_confirmed += 1
                matched_pattern.last_seen = datetime.now()
            
            logger.info(f"[EVENT] User confirmed: '{result.event.label}'")
        else:
            # User rejected - ask for correct label
            return self._handle_request_label(result, acf, 0.0)
        
        return result
    
    def _handle_request_label(
        self,
        result: EventPipelineResult,
        acf: "ActiveContextFrame",
        prediction_error: float,
    ) -> EventPipelineResult:
        """Handle request-label action for novel events.
        
        Args:
            result: The pipeline result to update.
            acf: Current context.
            prediction_error: Prediction error for salience.
            
        Returns:
            Updated result.
        """
        result.user_prompted = True
        
        # Generate suggestions based on structure
        suggestions = self._generate_label_suggestions(result.event)
        
        # Build prompt
        entity_info = ", ".join(result.event.involved_entity_labels) or "entity"
        prompt = (
            f"New event detected: {entity_info} "
            f"({result.event.pre_state_signature or ''} → {result.event.post_state_signature or ''})\n"
            f"What should this event be called?"
        )
        
        if self._auto_label_novel_events:
            # Auto-generate label
            label = suggestions[0] if suggestions else self._generate_structural_label(result.event)
            result.user_response = f"auto:{label}"
        else:
            # Ask user
            label = self._dialog_manager.ask_label(prompt, suggestions)
            result.user_response = label
        
        result.event.label = label
        result.event.is_learned = True
        result.event.confidence = 0.7  # User-provided labels get decent confidence
        
        # Learn this pattern
        self._learn_pattern(result.event, prediction_error)
        
        logger.info(f"[EVENT] User labeled new event: '{label}'")
        self._dialog_manager.notify(f"📝 Learned new event: {label}")
        
        return result
    
    def _learn_pattern(
        self,
        event: EventCandidate,
        prediction_error: float,
    ) -> LearnedEventPattern:
        """Learn a new event pattern from user labeling.
        
        Args:
            event: The labeled event.
            prediction_error: Prediction error for salience.
            
        Returns:
            The created pattern.
        """
        pattern_signature = event.extras.get("pattern_signature", "")
        
        pattern = LearnedEventPattern(
            pattern_id=f"pat_{uuid.uuid4().hex[:12]}",
            pattern_signature=pattern_signature,
            label=event.label,
            event_type=event.event_type,
            example_entity_labels=event.involved_entity_labels.copy(),
            example_location=event.location_label,
            times_seen=1,
            times_confirmed=1,  # User labeling counts as confirmation
            salience=SalienceWeights(
                user_label_weight=1.0,  # User explicitly labeled
                novelty_weight=1.0,  # First time seeing this pattern
                prediction_error_weight=min(1.0, prediction_error),
            ),
        )
        
        self._learned_patterns[pattern_signature] = pattern
        self._patterns_learned += 1
        
        logger.info(f"[EVENT] Learned pattern: '{pattern_signature}' → '{event.label}'")
        
        return pattern
    
    def _generate_structural_label(self, event: EventCandidate) -> str:
        """Generate a structural label for an event.
        
        Args:
            event: The event to label.
            
        Returns:
            Generated structural label.
        """
        entity = event.involved_entity_labels[0] if event.involved_entity_labels else "entity"
        
        if event.pre_state_signature and event.post_state_signature:
            # State change
            pre = event.pre_state_signature.replace("state:", "")
            post = event.post_state_signature.replace("state:", "")
            return f"{entity}_{pre}_to_{post}"
        elif event.event_type == "appeared":
            return f"{entity}_appeared"
        elif event.event_type == "disappeared":
            return f"{entity}_disappeared"
        elif event.event_type == "moved":
            return f"{entity}_moved"
        else:
            return f"{entity}_event"
    
    def _generate_label_suggestions(self, event: EventCandidate) -> list[str]:
        """Generate label suggestions for an event.
        
        Args:
            event: The event needing a label.
            
        Returns:
            List of suggested labels.
        """
        suggestions = []
        entity = event.involved_entity_labels[0] if event.involved_entity_labels else ""
        
        # Generate from structure
        structural = self._generate_structural_label(event)
        suggestions.append(structural)
        
        # Common patterns for state changes
        if event.pre_state_signature and event.post_state_signature:
            post = event.post_state_signature.replace("state:", "").lower()
            if entity:
                suggestions.append(f"{entity}_{post}")
            suggestions.append(post)
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _compute_salience(
        self,
        result: EventPipelineResult,
        matched_pattern: LearnedEventPattern | None,
        prediction_error: float,
    ) -> SalienceWeights:
        """Compute salience weights for an event.
        
        Args:
            result: The pipeline result.
            matched_pattern: Matched pattern if any.
            prediction_error: Prediction error score.
            
        Returns:
            Computed salience weights.
        """
        salience = SalienceWeights()
        
        # Prediction error contributes to salience (surprising events are salient)
        salience.prediction_error_weight = min(1.0, prediction_error)
        
        # User interaction increases salience
        if result.user_prompted:
            salience.user_label_weight = 0.8
            if result.action == ConfidenceAction.REQUEST_LABEL:
                salience.user_label_weight = 1.0  # Maximum for new labels
        
        # Novelty - inverse of times seen
        if matched_pattern:
            salience.novelty_weight = max(0.1, 1.0 - (matched_pattern.times_seen * 0.1))
            salience.frequency_weight = min(1.0, matched_pattern.times_seen * 0.1)
        else:
            salience.novelty_weight = 1.0  # Novel event
            salience.frequency_weight = 0.0
        
        return salience
    
    def _store_event_to_graph(
        self,
        result: EventPipelineResult,
        acf: "ActiveContextFrame",
    ) -> EventPipelineResult:
        """Store event to graph memory.
        
        Creates event node and edges to:
        - Involved entities (INVOLVES)
        - Location (OCCURRED_IN)
        - Source deltas (TRIGGERED_BY)
        - Similar patterns (SIMILAR_TO)
        
        Args:
            result: The pipeline result.
            acf: Current context.
            
        Returns:
            Updated result with graph node ID.
        """
        event = result.event
        
        # Create event node
        node = GraphNode(
            node_id=event.event_id,
            node_type=NODE_TYPE_EVENT,
            label=event.label,
            labels=[event.label],
            created_at=event.timestamp,
            activation=result.salience.total_salience(),
            embedding=None,  # TODO: Generate event embedding
            extras={
                "event_type": event.event_type,
                "pattern_signature": event.extras.get("pattern_signature"),
                "pre_state_signature": event.pre_state_signature,
                "post_state_signature": event.post_state_signature,
                "is_learned": event.is_learned,
                "confidence": event.confidence,
                "salience": result.salience.to_dict(),
                "step_number": event.step_number,
            },
        )
        
        try:
            self._graph_store.add_node(node)
            result.stored_to_graph = True
            result.graph_node_id = node.node_id
            
            # Create edges to involved entities
            for entity_id in event.involved_entity_ids:
                edge = GraphEdge(
                    edge_id=f"edge_{uuid.uuid4().hex[:12]}",
                    source_id=event.event_id,
                    target_id=entity_id,
                    edge_type=EDGE_TYPE_INVOLVES,
                    weight=result.salience.total_salience(),
                )
                self._graph_store.add_edge(edge)
            
            # Create edge to location
            if event.location_label:
                # Find or create location node
                location_nodes = [
                    n for n in self._graph_store.get_nodes_by_type(NODE_TYPE_LOCATION)
                    if n.label == event.location_label
                ]
                if location_nodes:
                    edge = GraphEdge(
                        edge_id=f"edge_{uuid.uuid4().hex[:12]}",
                        source_id=event.event_id,
                        target_id=location_nodes[0].node_id,
                        edge_type=EDGE_TYPE_OCCURRED_IN,
                        weight=1.0,
                    )
                    self._graph_store.add_edge(edge)
            
            logger.debug(f"[EVENT] Stored event to graph: {event.event_id}")
            
        except Exception as e:
            logger.warning(f"[EVENT] Failed to store event to graph: {e}")
        
        return result
    
    def get_learned_patterns(self) -> list[LearnedEventPattern]:
        """Get all learned event patterns.
        
        Returns:
            List of learned patterns.
        """
        return list(self._learned_patterns.values())
    
    def get_statistics(self) -> dict[str, Any]:
        """Get pipeline statistics.
        
        Returns:
            Dictionary of statistics.
        """
        return {
            "events_detected": self._events_detected,
            "events_auto_accepted": self._events_auto_accepted,
            "events_confirmed": self._events_confirmed,
            "events_labeled": self._events_labeled,
            "events_rejected": self._events_rejected,
            "patterns_learned": self._patterns_learned,
            "total_patterns": len(self._learned_patterns),
        }
    
    def reset_statistics(self) -> None:
        """Reset pipeline statistics."""
        self._events_detected = 0
        self._events_auto_accepted = 0
        self._events_confirmed = 0
        self._events_labeled = 0
        self._events_rejected = 0
