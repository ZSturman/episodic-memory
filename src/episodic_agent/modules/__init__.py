"""Module implementations."""

from episodic_agent.modules.acf_stability import (
    ACFFingerprint,
    ACFStabilityGuard,
    StabilityDecision,
    StabilityState,
    VariationType,
)
from episodic_agent.modules.arbitrator import (
    ArbitrationDecision,
    ArbitrationOutcome,
    ConflictType,
    MotionPerceptionArbitrator,
    MotionSignal,
    PerceptionSignal,
    SignalSource,
)
from episodic_agent.modules.boundary import (
    BoundaryReason,
    BoundaryState,
    HysteresisBoundaryDetector,
)
from episodic_agent.modules.delta_detector import DeltaDetector, EntitySnapshot
from episodic_agent.modules.dialog import (
    AutoAcceptDialogManager,
    CLIDialogManager,
    DialogManager,
    NonBlockingDialogManager,
)
from episodic_agent.modules.event_pipeline import (
    ConfidenceAction,
    EventLearningPipeline,
    EventPipelineResult,
    LearnedEventPattern,
    SalienceWeights,
)
from episodic_agent.modules.event_resolver import EventResolverStateChange
from episodic_agent.modules.label_manager import LabelManager
from episodic_agent.modules.prediction import (
    Prediction,
    PredictionError,
    PredictionErrorComputer,
    PredictionErrorType,
    PredictionGenerator,
    PredictionModule,
)
from episodic_agent.modules.retriever import (
    ActivatedNode,
    CueToken,
    RetrieverSpreadingActivation,
    SpreadingActivationResult,
)
from episodic_agent.modules.consolidation import (
    ConsolidationModule,
    ConsolidationOperation,
    ConsolidationResult,
    ConsolidationScheduler,
    DecayOperation,
    MergeOperation,
    OperationType,
    PruneOperation,
    RelabelOperation,
    DEFAULT_INACTIVITY_THRESHOLD,
    MERGE_CONFIDENCE_THRESHOLD,
    MERGE_SIMILARITY_THRESHOLD,
)

__all__ = [
    # Phase 3: ACF Stability
    "ACFFingerprint",
    "ACFStabilityGuard",
    "StabilityDecision",
    "StabilityState",
    "VariationType",
    # Phase 3: Motion-Perception Arbitration
    "ArbitrationDecision",
    "ArbitrationOutcome",
    "ConflictType",
    "MotionPerceptionArbitrator",
    "MotionSignal",
    "PerceptionSignal",
    "SignalSource",
    # Phase 4: Event Learning Pipeline
    "ConfidenceAction",
    "EventLearningPipeline",
    "EventPipelineResult",
    "LearnedEventPattern",
    "SalienceWeights",
    # Existing
    "ActivatedNode",
    "AutoAcceptDialogManager",
    "BoundaryReason",
    "BoundaryState",
    "CLIDialogManager",
    "CueToken",
    "DeltaDetector",
    "DialogManager",
    "EntitySnapshot",
    "EventResolverStateChange",
    "HysteresisBoundaryDetector",
    "LabelManager",
    "NonBlockingDialogManager",
    "Prediction",
    "PredictionError",
    "PredictionErrorComputer",
    "PredictionErrorType",
    "PredictionGenerator",
    "PredictionModule",
    "RetrieverSpreadingActivation",
    "SpreadingActivationResult",
    # Phase 7: Consolidation
    "ConsolidationModule",
    "ConsolidationOperation",
    "ConsolidationResult",
    "ConsolidationScheduler",
    "DecayOperation",
    "MergeOperation",
    "OperationType",
    "PruneOperation",
    "RelabelOperation",
    "DEFAULT_INACTIVITY_THRESHOLD",
    "MERGE_CONFIDENCE_THRESHOLD",
    "MERGE_SIMILARITY_THRESHOLD",
]
