"""Prediction module for generating expectations and computing prediction errors.

Generates lightweight predictions from semantic priors in the graph:
- "In location L, expect entities {E...}" based on typical_in edges
- Compute prediction errors each step (missing, unexpected, state change)
- Attach prediction errors to ACF for retrieval and boundary detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from episodic_agent.memory.graph_store import LabeledGraphStore
    from episodic_agent.schemas import ActiveContextFrame, Percept


logger = logging.getLogger(__name__)


class PredictionErrorType(str, Enum):
    """Types of prediction errors."""
    
    UNEXPECTED_MISSING = "unexpected_missing"    # Expected entity not present
    UNEXPECTED_NEW = "unexpected_new"            # Entity present but not expected
    UNEXPECTED_STATE = "unexpected_state"        # Entity in unexpected state
    LOCATION_MISMATCH = "location_mismatch"      # Location different from expected


@dataclass
class Prediction:
    """A single prediction about what should be present/true."""
    
    prediction_id: str
    prediction_type: str  # "entity_present", "entity_state", "location"
    target_id: str        # Entity/location ID being predicted
    target_label: str     # Human-readable label
    expected_value: Any   # What we expect (True for present, state value, etc.)
    confidence: float     # How confident in this prediction (0-1)
    source: str           # Where this prediction came from
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass  
class PredictionError:
    """A detected prediction error."""
    
    error_id: str
    error_type: PredictionErrorType
    prediction: Prediction | None
    actual_value: Any
    magnitude: float       # How "wrong" we were (0-1 scale)
    entity_id: str | None
    entity_label: str | None
    location_label: str | None
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "error_id": self.error_id,
            "type": self.error_type.value,
            "prediction_id": self.prediction.prediction_id if self.prediction else None,
            "actual_value": self.actual_value,
            "magnitude": self.magnitude,
            "entity_id": self.entity_id,
            "entity_label": self.entity_label,
            "location_label": self.location_label,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class PredictionGenerator:
    """Generates predictions based on learned semantic priors.
    
    Uses the graph store to generate predictions:
    - typical_in edges: "In location L, expect entities with this edge"
    - Recent history: What entities were recently seen here
    - State priors: Expected states for certain entity types
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        min_confidence: float = 0.3,
        typical_in_weight: float = 0.7,
        recency_weight: float = 0.3,
    ) -> None:
        """Initialize the prediction generator.
        
        Args:
            graph_store: Graph store for semantic priors.
            min_confidence: Minimum confidence to generate prediction.
            typical_in_weight: Weight for typical_in based predictions.
            recency_weight: Weight for recency-based predictions.
        """
        self._graph_store = graph_store
        self._min_confidence = min_confidence
        self._typical_in_weight = typical_in_weight
        self._recency_weight = recency_weight
        
        # Cache for efficiency
        self._location_entity_cache: dict[str, list[tuple[str, str, float]]] = {}
        self._cache_valid = False
        
        # Statistics
        self._predictions_generated = 0
        self._cache_hits = 0

    def generate_predictions(
        self,
        location_label: str,
        location_confidence: float,
    ) -> list[Prediction]:
        """Generate predictions for what should be present at a location.
        
        Args:
            location_label: Current location label.
            location_confidence: Confidence in location identification.
            
        Returns:
            List of predictions about expected entities.
        """
        if location_label == "unknown" or location_confidence < self._min_confidence:
            return []
        
        predictions = []
        prediction_count = 0
        
        # Get entities typically found in this location
        typical_entities = self._get_typical_entities(location_label)
        
        for entity_id, entity_label, edge_confidence in typical_entities:
            # Combine edge confidence with location confidence
            pred_confidence = edge_confidence * location_confidence * self._typical_in_weight
            
            if pred_confidence >= self._min_confidence:
                predictions.append(Prediction(
                    prediction_id=f"pred_{prediction_count:04d}",
                    prediction_type="entity_present",
                    target_id=entity_id,
                    target_label=entity_label,
                    expected_value=True,
                    confidence=pred_confidence,
                    source=f"typical_in:{location_label}",
                ))
                prediction_count += 1
        
        self._predictions_generated += len(predictions)
        return predictions

    def _get_typical_entities(
        self,
        location_label: str,
    ) -> list[tuple[str, str, float]]:
        """Get entities typically found at a location.
        
        Args:
            location_label: Location to query.
            
        Returns:
            List of (entity_id, entity_label, confidence) tuples.
        """
        # Check cache
        if self._cache_valid and location_label in self._location_entity_cache:
            self._cache_hits += 1
            return self._location_entity_cache[location_label]
        
        typical = []
        
        # Find location node(s)
        from episodic_agent.schemas import EdgeType, NodeType
        
        location_nodes = self._graph_store.get_nodes_by_label(location_label)
        location_ids = {
            n.node_id for n in location_nodes 
            if n.node_type == NodeType.LOCATION
        }
        
        if not location_ids:
            return []
        
        # Find entities with typical_in edges to this location
        for edge in self._graph_store.get_all_edges():
            if edge.edge_type == EdgeType.TYPICAL_IN:
                if edge.target_node_id in location_ids:
                    # Source is entity, target is location
                    entity_node = self._graph_store.get_node(edge.source_node_id)
                    if entity_node and entity_node.node_type == NodeType.ENTITY:
                        typical.append((
                            entity_node.source_id or entity_node.node_id,
                            entity_node.label,
                            edge.confidence,
                        ))
        
        # Cache result
        self._location_entity_cache[location_label] = typical
        self._cache_valid = True
        
        return typical

    def invalidate_cache(self) -> None:
        """Invalidate the prediction cache (call when graph changes)."""
        self._cache_valid = False
        self._location_entity_cache.clear()


class PredictionErrorComputer:
    """Computes prediction errors by comparing predictions to observations.
    
    Detects:
    - Missing entities: predicted but not observed
    - Unexpected entities: observed but not predicted
    - State mismatches: entity in unexpected state
    """

    def __init__(
        self,
        prediction_generator: PredictionGenerator,
        missing_threshold: float = 0.5,
        unexpected_threshold: float = 0.5,
        state_change_weight: float = 1.0,
        track_history: bool = True,
    ) -> None:
        """Initialize the prediction error computer.
        
        Args:
            prediction_generator: Generator for predictions.
            missing_threshold: Min prediction confidence to flag as missing.
            unexpected_threshold: Min observation confidence to flag as unexpected.
            state_change_weight: Weight for state change errors.
            track_history: Whether to track error history.
        """
        self._generator = prediction_generator
        self._missing_threshold = missing_threshold
        self._unexpected_threshold = unexpected_threshold
        self._state_change_weight = state_change_weight
        self._track_history = track_history
        
        # History
        self._error_history: list[PredictionError] = []
        self._error_count = 0
        
        # Previous step predictions for comparison
        self._previous_predictions: list[Prediction] = []
        
        # Statistics
        self._total_errors = 0
        self._errors_by_type: dict[str, int] = {
            PredictionErrorType.UNEXPECTED_MISSING.value: 0,
            PredictionErrorType.UNEXPECTED_NEW.value: 0,
            PredictionErrorType.UNEXPECTED_STATE.value: 0,
        }

    @property
    def error_history(self) -> list[PredictionError]:
        """Get prediction error history."""
        return self._error_history

    @property
    def total_errors(self) -> int:
        """Get total error count."""
        return self._total_errors

    def compute_errors(
        self,
        acf: "ActiveContextFrame",
        percept: "Percept | None" = None,
    ) -> list[PredictionError]:
        """Compute prediction errors for current step.
        
        Args:
            acf: Active context frame with current observations.
            percept: Optional current percept for additional info.
            
        Returns:
            List of prediction errors detected.
        """
        import uuid
        
        errors = []
        
        # Generate predictions for current location
        predictions = self._generator.generate_predictions(
            acf.location_label,
            acf.location_confidence,
        )
        
        # Get observed entity IDs
        observed_ids = {
            e.candidate_id for e in acf.entities
            if e.candidate_id
        }
        observed_labels = {
            e.label.lower() for e in acf.entities
            if e.label and e.label != "unknown"
        }
        
        # Check for missing entities (predicted but not observed)
        for pred in predictions:
            if pred.prediction_type == "entity_present":
                if pred.confidence >= self._missing_threshold:
                    # Check if entity is observed (by ID or label)
                    is_present = (
                        pred.target_id in observed_ids or
                        pred.target_label.lower() in observed_labels
                    )
                    
                    if not is_present:
                        self._error_count += 1
                        error = PredictionError(
                            error_id=f"err_{uuid.uuid4().hex[:8]}",
                            error_type=PredictionErrorType.UNEXPECTED_MISSING,
                            prediction=pred,
                            actual_value=False,
                            magnitude=pred.confidence,
                            entity_id=pred.target_id,
                            entity_label=pred.target_label,
                            location_label=acf.location_label,
                            details={"source": pred.source},
                        )
                        errors.append(error)
        
        # Check for unexpected entities (observed but not predicted)
        predicted_ids = {p.target_id for p in predictions}
        predicted_labels = {p.target_label.lower() for p in predictions}
        
        for entity in acf.entities:
            if entity.confidence >= self._unexpected_threshold:
                is_predicted = (
                    entity.candidate_id in predicted_ids or
                    (entity.label and entity.label.lower() in predicted_labels)
                )
                
                if not is_predicted:
                    # Check if this entity is new to this location
                    # (we don't flag entities we've never seen anywhere)
                    self._error_count += 1
                    error = PredictionError(
                        error_id=f"err_{uuid.uuid4().hex[:8]}",
                        error_type=PredictionErrorType.UNEXPECTED_NEW,
                        prediction=None,
                        actual_value=True,
                        magnitude=entity.confidence * 0.5,  # Lower magnitude for unexpected new
                        entity_id=entity.candidate_id,
                        entity_label=entity.label,
                        location_label=acf.location_label,
                        details={"category": entity.category},
                    )
                    errors.append(error)
        
        # Check for state changes in deltas
        for delta in acf.deltas:
            if delta.get("delta_type") == "state_changed":
                self._error_count += 1
                error = PredictionError(
                    error_id=f"err_{uuid.uuid4().hex[:8]}",
                    error_type=PredictionErrorType.UNEXPECTED_STATE,
                    prediction=None,
                    actual_value=delta.get("post_state"),
                    magnitude=self._state_change_weight * delta.get("confidence", 0.5),
                    entity_id=delta.get("entity_id"),
                    entity_label=delta.get("entity_label"),
                    location_label=acf.location_label,
                    details={
                        "pre_state": delta.get("pre_state"),
                        "post_state": delta.get("post_state"),
                    },
                )
                errors.append(error)
        
        # Update statistics
        self._total_errors += len(errors)
        for error in errors:
            self._errors_by_type[error.error_type.value] = (
                self._errors_by_type.get(error.error_type.value, 0) + 1
            )
        
        # Store history
        if self._track_history:
            self._error_history.extend(errors)
            # Keep limited history
            if len(self._error_history) > 1000:
                self._error_history = self._error_history[-500:]
        
        # Store predictions for next step
        self._previous_predictions = predictions
        
        return errors

    def attach_errors_to_acf(
        self,
        acf: "ActiveContextFrame",
        errors: list[PredictionError],
    ) -> None:
        """Attach prediction errors to ACF extras.
        
        Args:
            acf: Active context frame to update.
            errors: Prediction errors to attach.
        """
        acf.extras["prediction_errors"] = [e.to_dict() for e in errors]
        acf.extras["prediction_error_count"] = len(errors)
        acf.extras["prediction_error_magnitude"] = (
            sum(e.magnitude for e in errors) if errors else 0.0
        )

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of prediction errors.
        
        Returns:
            Summary dictionary.
        """
        return {
            "total_errors": self._total_errors,
            "by_type": dict(self._errors_by_type),
            "history_size": len(self._error_history),
        }


class PredictionModule:
    """Combined prediction generation and error computation.
    
    Convenience wrapper that combines PredictionGenerator and
    PredictionErrorComputer for use in the agent loop.
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        min_confidence: float = 0.3,
        missing_threshold: float = 0.5,
        unexpected_threshold: float = 0.5,
    ) -> None:
        """Initialize the prediction module.
        
        Args:
            graph_store: Graph store for semantic priors.
            min_confidence: Minimum confidence for predictions.
            missing_threshold: Threshold for missing entity errors.
            unexpected_threshold: Threshold for unexpected entity errors.
        """
        self._generator = PredictionGenerator(
            graph_store=graph_store,
            min_confidence=min_confidence,
        )
        self._error_computer = PredictionErrorComputer(
            prediction_generator=self._generator,
            missing_threshold=missing_threshold,
            unexpected_threshold=unexpected_threshold,
        )

    @property
    def generator(self) -> PredictionGenerator:
        """Get the prediction generator."""
        return self._generator

    @property
    def error_computer(self) -> PredictionErrorComputer:
        """Get the error computer."""
        return self._error_computer

    def process(
        self,
        acf: "ActiveContextFrame",
        percept: "Percept | None" = None,
    ) -> list[PredictionError]:
        """Generate predictions and compute errors for current step.
        
        Args:
            acf: Active context frame.
            percept: Optional current percept.
            
        Returns:
            List of prediction errors.
        """
        errors = self._error_computer.compute_errors(acf, percept)
        self._error_computer.attach_errors_to_acf(acf, errors)
        return errors

    def invalidate_cache(self) -> None:
        """Invalidate prediction cache."""
        self._generator.invalidate_cache()

    def get_statistics(self) -> dict[str, Any]:
        """Get prediction statistics.
        
        Returns:
            Statistics dictionary.
        """
        return {
            "predictions_generated": self._generator._predictions_generated,
            "cache_hits": self._generator._cache_hits,
            "error_summary": self._error_computer.get_error_summary(),
        }
