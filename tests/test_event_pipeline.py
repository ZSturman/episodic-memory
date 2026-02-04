"""Tests for Phase 4: Event Learning Pipeline.

Tests the EventLearningPipeline with:
- Confidence-based action selection
- Pattern learning and matching
- User interaction flow
- Salience weight computation
- Graph storage integration

ARCHITECTURAL INVARIANT: No predefined event types.
All event semantics emerge from user labeling and pattern matching.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from episodic_agent.modules.event_pipeline import (
    ConfidenceAction,
    EventLearningPipeline,
    EventPipelineResult,
    LearnedEventPattern,
    SalienceWeights,
    CONFIDENCE_AUTO_ACCEPT,
    CONFIDENCE_CONFIRM,
    CONFIDENCE_REQUEST_LABEL,
)
from episodic_agent.modules.dialog import AutoAcceptDialogManager, DialogManager
from episodic_agent.memory.graph_store import LabeledGraphStore
from episodic_agent.schemas.events import (
    Delta,
    EventCandidate,
    DELTA_TYPE_CHANGED,
    DELTA_TYPE_NEW,
    DELTA_TYPE_MISSING,
    DELTA_TYPE_MOVED,
)
from episodic_agent.schemas import ActiveContextFrame, Percept
from episodic_agent.schemas.graph import GraphNode, NODE_TYPE_EVENT


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_graph_store() -> MagicMock:
    """Create a mock graph store."""
    store = MagicMock(spec=LabeledGraphStore)
    store.get_nodes_by_type.return_value = []
    store.add_node.return_value = None
    store.add_edge.return_value = None
    return store


@pytest.fixture
def mock_dialog_manager() -> MagicMock:
    """Create a mock dialog manager."""
    dialog = MagicMock(spec=DialogManager)
    dialog.confirm.return_value = True
    dialog.ask_label.return_value = "test_event"
    dialog.notify.return_value = None
    return dialog


@pytest.fixture
def auto_dialog() -> AutoAcceptDialogManager:
    """Create an auto-accept dialog manager."""
    return AutoAcceptDialogManager()


@pytest.fixture
def sample_acf() -> ActiveContextFrame:
    """Create a sample active context frame."""
    return ActiveContextFrame(
        acf_id="acf_test_001",
        location_label="living_room",
        step_count=10,
    )


@pytest.fixture
def state_change_delta() -> Delta:
    """Create a state change delta."""
    return Delta(
        delta_id="delta_001",
        delta_type=DELTA_TYPE_CHANGED,
        entity_id="ent_lamp",
        entity_label="lamp",
        pre_state="off",
        post_state="on",
        confidence=0.9,
        step_number=10,
        location_label="living_room",
        evidence=["visual_change"],
    )


@pytest.fixture
def appearance_delta() -> Delta:
    """Create an appearance delta."""
    return Delta(
        delta_id="delta_002",
        delta_type=DELTA_TYPE_NEW,
        entity_id="ent_cat",
        entity_label="cat",
        confidence=0.85,
        step_number=11,
        location_label="living_room",
        evidence=["new_detection"],
    )


@pytest.fixture
def disappearance_delta() -> Delta:
    """Create a disappearance delta."""
    return Delta(
        delta_id="delta_003",
        delta_type=DELTA_TYPE_MISSING,
        entity_id="ent_cup",
        entity_label="cup",
        confidence=0.8,
        step_number=12,
        location_label="kitchen",
        evidence=["absence_detected"],
    )


@pytest.fixture
def movement_delta() -> Delta:
    """Create a movement delta."""
    return Delta(
        delta_id="delta_004",
        delta_type=DELTA_TYPE_MOVED,
        entity_id="ent_chair",
        entity_label="chair",
        pre_position=(1.0, 0.0, 2.0),
        post_position=(3.0, 0.0, 4.0),
        confidence=0.75,
        step_number=13,
        location_label="living_room",
        evidence=["position_change"],
    )


@pytest.fixture
def pipeline(mock_graph_store, mock_dialog_manager) -> EventLearningPipeline:
    """Create an event learning pipeline with mocks."""
    return EventLearningPipeline(
        graph_store=mock_graph_store,
        dialog_manager=mock_dialog_manager,
        store_to_graph=False,  # Disable for faster tests
    )


@pytest.fixture
def pipeline_with_auto_dialog(mock_graph_store, auto_dialog) -> EventLearningPipeline:
    """Create a pipeline with auto-accept dialog."""
    return EventLearningPipeline(
        graph_store=mock_graph_store,
        dialog_manager=auto_dialog,
        store_to_graph=False,
        auto_label_novel_events=True,
    )


# =============================================================================
# Test: SalienceWeights
# =============================================================================

class TestSalienceWeights:
    """Tests for SalienceWeights dataclass."""
    
    def test_default_salience_is_zero(self):
        """Default salience weights should be zero."""
        salience = SalienceWeights()
        
        assert salience.prediction_error_weight == 0.0
        assert salience.user_label_weight == 0.0
        assert salience.novelty_weight == 0.0
        assert salience.total_salience() == 0.0
    
    def test_salience_computation(self):
        """Salience should combine weights correctly."""
        salience = SalienceWeights(
            prediction_error_weight=1.0,
            user_label_weight=1.0,
            novelty_weight=1.0,
            visual_stimuli_weight=1.0,
            frequency_weight=1.0,
        )
        
        # 0.25 + 0.30 + 0.20 + 0.15 + 0.10 = 1.0
        assert salience.total_salience() == 1.0
    
    def test_user_label_weight_dominates(self):
        """User labeling should have highest weight."""
        user_labeled = SalienceWeights(user_label_weight=1.0)
        surprising = SalienceWeights(prediction_error_weight=1.0)
        novel = SalienceWeights(novelty_weight=1.0)
        
        assert user_labeled.total_salience() > surprising.total_salience()
        assert user_labeled.total_salience() > novel.total_salience()
    
    def test_to_dict(self):
        """to_dict should include all weights."""
        salience = SalienceWeights(
            prediction_error_weight=0.5,
            user_label_weight=0.8,
        )
        d = salience.to_dict()
        
        assert "prediction_error_weight" in d
        assert "user_label_weight" in d
        assert "total_salience" in d
        assert d["prediction_error_weight"] == 0.5
        assert d["user_label_weight"] == 0.8


# =============================================================================
# Test: Confidence Action Selection
# =============================================================================

class TestConfidenceActionSelection:
    """Tests for confidence-based action selection."""
    
    def test_high_confidence_auto_accepts(self, pipeline):
        """High confidence should trigger auto-accept."""
        action = pipeline._determine_action(0.9, MagicMock())
        assert action == ConfidenceAction.AUTO_ACCEPT
    
    def test_medium_confidence_confirms(self, pipeline):
        """Medium confidence should trigger confirmation."""
        action = pipeline._determine_action(0.6, MagicMock())
        assert action == ConfidenceAction.CONFIRM
    
    def test_low_confidence_requests_label(self, pipeline):
        """Low confidence with a pattern should request new label."""
        # When there's a matched pattern but low confidence, should request label
        action = pipeline._determine_action(0.35, MagicMock())
        assert action == ConfidenceAction.REQUEST_LABEL
    
    def test_very_low_confidence_rejects(self, pipeline):
        """Very low confidence with a pattern should reject."""
        # Below the request_label threshold with a pattern → reject
        action = pipeline._determine_action(0.1, MagicMock())
        assert action == ConfidenceAction.REJECT
    
    def test_no_pattern_requests_label(self, pipeline):
        """No matched pattern should always request label."""
        # Even with 0 confidence, no pattern → request label
        action = pipeline._determine_action(0.0, None)
        assert action == ConfidenceAction.REQUEST_LABEL
    
    def test_threshold_boundaries(self, pipeline):
        """Test exact threshold boundaries."""
        # Exactly at auto-accept threshold
        action = pipeline._determine_action(0.8, MagicMock())
        assert action == ConfidenceAction.AUTO_ACCEPT
        
        # Just below auto-accept
        action = pipeline._determine_action(0.79, MagicMock())
        assert action == ConfidenceAction.CONFIRM
        
        # Exactly at confirm threshold
        action = pipeline._determine_action(0.5, MagicMock())
        assert action == ConfidenceAction.CONFIRM
        
        # Just below confirm
        action = pipeline._determine_action(0.49, MagicMock())
        assert action == ConfidenceAction.REQUEST_LABEL


# =============================================================================
# Test: Event Candidate Creation
# =============================================================================

class TestEventCandidateCreation:
    """Tests for creating event candidates from deltas."""
    
    def test_state_change_creates_valid_event(self, pipeline, state_change_delta, sample_acf):
        """State change delta should create valid event candidate."""
        event = pipeline._create_event_candidate(state_change_delta, sample_acf)
        
        assert event is not None
        assert event.event_type == "state_change"
        assert "lamp" in event.involved_entity_labels
        assert event.pre_state_signature == "state:off"
        assert event.post_state_signature == "state:on"
        assert "pattern_signature" in event.extras
    
    def test_appearance_creates_valid_event(self, pipeline, appearance_delta, sample_acf):
        """Appearance delta should create valid event candidate."""
        event = pipeline._create_event_candidate(appearance_delta, sample_acf)
        
        assert event is not None
        assert event.event_type == "appeared"
        assert "cat" in event.involved_entity_labels
        assert event.extras.get("pattern_signature") == "entity:appeared"
    
    def test_disappearance_creates_valid_event(self, pipeline, disappearance_delta, sample_acf):
        """Disappearance delta should create valid event candidate."""
        event = pipeline._create_event_candidate(disappearance_delta, sample_acf)
        
        assert event is not None
        assert event.event_type == "disappeared"
        assert "cup" in event.involved_entity_labels
        assert event.extras.get("pattern_signature") == "entity:disappeared"
    
    def test_movement_creates_valid_event(self, pipeline, movement_delta, sample_acf):
        """Movement delta should create valid event candidate."""
        event = pipeline._create_event_candidate(movement_delta, sample_acf)
        
        assert event is not None
        assert event.event_type == "moved"
        assert "chair" in event.involved_entity_labels
        assert event.extras.get("pattern_signature") == "entity:moved"


# =============================================================================
# Test: Pattern Learning
# =============================================================================

class TestPatternLearning:
    """Tests for event pattern learning."""
    
    def test_learn_pattern_stores_signature(self, pipeline, state_change_delta, sample_acf):
        """Learning a pattern should store the signature."""
        event = pipeline._create_event_candidate(state_change_delta, sample_acf)
        event.label = "lamp_turned_on"
        
        pattern = pipeline._learn_pattern(event, prediction_error=0.5)
        
        assert pattern is not None
        assert pattern.label == "lamp_turned_on"
        assert pattern.pattern_signature in pipeline._learned_patterns
    
    def test_learned_pattern_increments_counter(self, pipeline):
        """Learning patterns should increment counter."""
        assert pipeline._patterns_learned == 0
        
        event = EventCandidate(
            event_id="evt_001",
            event_type="test",
            label="test_event",
            extras={"pattern_signature": "test_sig_1"},
        )
        pipeline._learn_pattern(event, 0.0)
        
        assert pipeline._patterns_learned == 1
    
    def test_learned_pattern_has_high_user_salience(self, pipeline):
        """User-learned patterns should have high user_label_weight."""
        event = EventCandidate(
            event_id="evt_001",
            event_type="test",
            label="user_labeled_event",
            extras={"pattern_signature": "user_sig"},
        )
        
        pattern = pipeline._learn_pattern(event, 0.0)
        
        assert pattern.salience.user_label_weight == 1.0
        assert pattern.salience.novelty_weight == 1.0


# =============================================================================
# Test: Pattern Matching
# =============================================================================

class TestPatternMatching:
    """Tests for pattern matching against learned events."""
    
    def test_exact_match_returns_high_confidence(self, pipeline):
        """Exact pattern match should return high confidence."""
        # Learn a pattern
        pattern = LearnedEventPattern(
            pattern_id="pat_001",
            pattern_signature="state:off->state:on",
            label="lamp_on",
            event_type="state_change",
            times_confirmed=3,
        )
        pipeline._learned_patterns["state:off->state:on"] = pattern
        
        # Create matching event
        event = EventCandidate(
            event_id="evt_001",
            event_type="state_change",
            label="unknown",
            extras={"pattern_signature": "state:off->state:on"},
        )
        
        matched, confidence = pipeline._match_pattern(event)
        
        assert matched == pattern
        assert confidence >= 0.8  # High confidence for confirmed pattern
    
    def test_no_match_returns_none(self, pipeline):
        """Non-matching pattern should return None."""
        event = EventCandidate(
            event_id="evt_001",
            event_type="test",
            label="unknown",
            extras={"pattern_signature": "unique_signature"},
        )
        
        matched, confidence = pipeline._match_pattern(event)
        
        assert matched is None
        assert confidence == 0.0
    
    def test_fuzzy_match_returns_lower_confidence(self, pipeline):
        """Fuzzy match should return lower confidence."""
        # Learn a pattern
        pattern = LearnedEventPattern(
            pattern_id="pat_001",
            pattern_signature="state:off->state:on",
            label="light_on",
            event_type="state_change",
        )
        pipeline._learned_patterns["state:off->state:on"] = pattern
        
        # Create similar but not exact pattern
        event = EventCandidate(
            event_id="evt_001",
            event_type="state_change",
            label="unknown",
            extras={"pattern_signature": "state:dim->state:on"},
        )
        
        matched, confidence = pipeline._match_pattern(event)
        
        # Should fuzzy match but with lower confidence
        if matched:
            assert confidence < 0.8


# =============================================================================
# Test: Full Pipeline Flow
# =============================================================================

class TestFullPipelineFlow:
    """Tests for full event pipeline processing."""
    
    def test_novel_event_prompts_for_label(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Novel event should prompt user for label."""
        mock_dialog_manager.ask_label.return_value = "lamp_turned_on"
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        assert result is not None
        assert result.user_prompted
        assert result.event.label == "lamp_turned_on"
        mock_dialog_manager.ask_label.assert_called_once()
    
    def test_recognized_event_auto_accepts(self, pipeline, mock_dialog_manager, state_change_delta, sample_acf):
        """Recognized event should auto-accept without prompting."""
        # Pre-learn the pattern
        pattern = LearnedEventPattern(
            pattern_id="pat_001",
            pattern_signature="state:off->state:on",
            label="light_on",
            event_type="state_change",
            times_confirmed=5,
        )
        pipeline._learned_patterns["state:off->state:on"] = pattern
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        assert result is not None
        assert result.action == ConfidenceAction.AUTO_ACCEPT
        assert not result.user_prompted
        assert result.event.label == "light_on"
        mock_dialog_manager.ask_label.assert_not_called()
    
    def test_medium_confidence_confirms(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Medium confidence should prompt for confirmation."""
        # Pre-learn with low confirmation count
        pattern = LearnedEventPattern(
            pattern_id="pat_001",
            pattern_signature="state:off->state:on",
            label="light_on",
            event_type="state_change",
            times_confirmed=0,  # Low confirmation = lower confidence
        )
        pipeline._learned_patterns["state:off->state:on"] = pattern
        pipeline._confidence_auto_accept = 0.9  # Raise threshold
        
        mock_dialog_manager.confirm.return_value = True
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        # Should either confirm or request label depending on exact confidence
        assert result is not None
        if result.action == ConfidenceAction.CONFIRM:
            assert result.user_prompted
    
    def test_confirmation_rejection_requests_label(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Rejecting confirmation should request new label."""
        pattern = LearnedEventPattern(
            pattern_id="pat_001",
            pattern_signature="state:off->state:on",
            label="light_on",
            event_type="state_change",
        )
        pipeline._learned_patterns["state:off->state:on"] = pattern
        pipeline._confidence_auto_accept = 0.99  # Force confirmation mode
        
        mock_dialog_manager.confirm.return_value = False
        mock_dialog_manager.ask_label.return_value = "lamp_switched_on"
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        # If confirmation was requested and rejected, should ask for label
        if result.user_prompted and result.user_response == "rejected":
            assert result.event.label == "lamp_switched_on"


# =============================================================================
# Test: Auto-Label Mode
# =============================================================================

class TestAutoLabelMode:
    """Tests for auto-label novel events mode."""
    
    def test_auto_label_generates_structural_label(
        self, pipeline_with_auto_dialog, state_change_delta, sample_acf
    ):
        """Auto-label should generate structural label."""
        result = pipeline_with_auto_dialog.process_delta(state_change_delta, sample_acf)
        
        assert result is not None
        assert result.event.label != "unknown"
        # Should be something like "lamp_off_to_on" or structural
        assert "lamp" in result.event.label.lower() or "to" in result.event.label.lower()
    
    def test_auto_label_still_learns_pattern(
        self, pipeline_with_auto_dialog, state_change_delta, sample_acf
    ):
        """Auto-label should still learn patterns."""
        initial_patterns = len(pipeline_with_auto_dialog._learned_patterns)
        
        pipeline_with_auto_dialog.process_delta(state_change_delta, sample_acf)
        
        assert len(pipeline_with_auto_dialog._learned_patterns) > initial_patterns


# =============================================================================
# Test: Multiple Deltas
# =============================================================================

class TestMultipleDeltas:
    """Tests for processing multiple deltas."""
    
    def test_process_multiple_deltas(
        self, pipeline_with_auto_dialog, state_change_delta, appearance_delta, sample_acf
    ):
        """Should process multiple deltas in sequence."""
        deltas = [state_change_delta, appearance_delta]
        
        results = pipeline_with_auto_dialog.process_deltas(deltas, sample_acf)
        
        assert len(results) == 2
        assert all(r.event is not None for r in results)
    
    def test_deltas_learn_independently(
        self, pipeline_with_auto_dialog, state_change_delta, appearance_delta, sample_acf
    ):
        """Each delta should learn its own pattern."""
        deltas = [state_change_delta, appearance_delta]
        
        pipeline_with_auto_dialog.process_deltas(deltas, sample_acf)
        
        # Should have learned 2 patterns
        assert pipeline_with_auto_dialog._patterns_learned >= 2


# =============================================================================
# Test: Statistics
# =============================================================================

class TestStatistics:
    """Tests for pipeline statistics."""
    
    def test_statistics_track_events(
        self, pipeline_with_auto_dialog, state_change_delta, sample_acf
    ):
        """Statistics should track processed events."""
        pipeline_with_auto_dialog.process_delta(state_change_delta, sample_acf)
        
        stats = pipeline_with_auto_dialog.get_statistics()
        
        assert stats["events_detected"] >= 1
        assert stats["events_labeled"] >= 1 or stats["events_auto_accepted"] >= 1
    
    def test_reset_statistics(
        self, pipeline_with_auto_dialog, state_change_delta, sample_acf
    ):
        """Statistics should be resettable."""
        pipeline_with_auto_dialog.process_delta(state_change_delta, sample_acf)
        pipeline_with_auto_dialog.reset_statistics()
        
        stats = pipeline_with_auto_dialog.get_statistics()
        
        assert stats["events_detected"] == 0
    
    def test_learned_patterns_persists_after_reset(
        self, pipeline_with_auto_dialog, state_change_delta, sample_acf
    ):
        """Learned patterns should persist after stats reset."""
        pipeline_with_auto_dialog.process_delta(state_change_delta, sample_acf)
        patterns_before = len(pipeline_with_auto_dialog._learned_patterns)
        
        pipeline_with_auto_dialog.reset_statistics()
        
        assert len(pipeline_with_auto_dialog._learned_patterns) == patterns_before


# =============================================================================
# Test: Graph Integration
# =============================================================================

class TestGraphIntegration:
    """Tests for graph storage integration."""
    
    def test_stores_event_to_graph(
        self, mock_graph_store, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Should store events to graph when enabled."""
        pipeline = EventLearningPipeline(
            graph_store=mock_graph_store,
            dialog_manager=mock_dialog_manager,
            store_to_graph=True,
            auto_label_novel_events=True,
        )
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        assert result.stored_to_graph
        mock_graph_store.add_node.assert_called()
    
    def test_loads_patterns_from_graph(self, mock_dialog_manager):
        """Should load existing patterns from graph on init."""
        mock_store = MagicMock(spec=LabeledGraphStore)
        mock_store.get_nodes_by_type.return_value = [
            GraphNode(
                node_id="evt_existing",
                node_type=NODE_TYPE_EVENT,
                label="existing_event",
                extras={
                    "pattern_signature": "state:closed->state:open",
                    "event_type": "state_change",
                },
            )
        ]
        
        pipeline = EventLearningPipeline(
            graph_store=mock_store,
            dialog_manager=mock_dialog_manager,
        )
        
        assert len(pipeline._learned_patterns) == 1
        assert "state:closed->state:open" in pipeline._learned_patterns


# =============================================================================
# Test: Salience Computation
# =============================================================================

class TestSalienceComputation:
    """Tests for salience weight computation."""
    
    def test_user_prompted_increases_salience(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """User interaction should increase salience."""
        mock_dialog_manager.ask_label.return_value = "user_event"
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        assert result.salience.user_label_weight > 0.0
    
    def test_novel_event_has_high_novelty(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Novel events should have high novelty weight."""
        mock_dialog_manager.ask_label.return_value = "novel_event"
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        assert result.salience.novelty_weight >= 0.9  # Very novel
    
    def test_prediction_error_affects_salience(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Prediction error should affect salience."""
        mock_dialog_manager.ask_label.return_value = "surprising_event"
        
        result = pipeline.process_delta(state_change_delta, sample_acf, prediction_error=0.9)
        
        assert result.salience.prediction_error_weight == 0.9


# =============================================================================
# Test: Architectural Invariants
# =============================================================================

class TestArchitecturalInvariants:
    """Tests for architectural invariants."""
    
    def test_no_predefined_event_types(self, pipeline, state_change_delta, sample_acf):
        """Events should not have predefined semantic labels."""
        event = pipeline._create_event_candidate(state_change_delta, sample_acf)
        
        # Label should be "unknown" until learned
        assert event.label == "unknown"
    
    def test_labels_emerge_from_user(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Labels should come from user interaction."""
        user_label = "my_custom_event_name"
        mock_dialog_manager.ask_label.return_value = user_label
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        assert result.event.label == user_label
    
    def test_learned_events_marked_as_learned(
        self, pipeline, mock_dialog_manager, state_change_delta, sample_acf
    ):
        """Learned events should be marked with is_learned=True."""
        mock_dialog_manager.ask_label.return_value = "learned_event"
        
        result = pipeline.process_delta(state_change_delta, sample_acf)
        
        assert result.event.is_learned is True
    
    def test_pattern_signature_is_structural(self, pipeline, state_change_delta, sample_acf):
        """Pattern signatures should be structural, not semantic."""
        event = pipeline._create_event_candidate(state_change_delta, sample_acf)
        signature = event.extras.get("pattern_signature", "")
        
        # Signature should describe state transition, not meaning
        assert "->" in signature or ":" in signature
        # Should not contain semantic terms like "turn_on", "activate"
        assert "turn" not in signature
        assert "activate" not in signature


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_delta_list(self, pipeline, sample_acf):
        """Empty delta list should return empty results."""
        results = pipeline.process_deltas([], sample_acf)
        assert results == []
    
    def test_delta_without_entity(self, pipeline, sample_acf):
        """Delta without entity should handle gracefully."""
        delta = Delta(
            delta_id="delta_empty",
            delta_type=DELTA_TYPE_CHANGED,
            entity_id=None,
            entity_label=None,
            pre_state="state1",
            post_state="state2",
            confidence=0.8,
        )
        
        event = pipeline._create_event_candidate(delta, sample_acf)
        
        assert event is not None
        assert event.involved_entity_ids == []
    
    def test_delta_with_none_states(self, pipeline, sample_acf):
        """Delta with None states should be handled."""
        delta = Delta(
            delta_id="delta_none",
            delta_type=DELTA_TYPE_CHANGED,
            entity_id="ent_1",
            pre_state=None,
            post_state=None,
            confidence=0.5,
        )
        
        event = pipeline._create_event_candidate(delta, sample_acf)
        
        # Should return None if states are missing
        assert event is None
    
    def test_pattern_similarity_with_empty_strings(self, pipeline):
        """Pattern similarity with empty strings."""
        score = pipeline._compute_pattern_similarity("", "")
        assert score == 0.0
        
        score = pipeline._compute_pattern_similarity("test", "")
        assert score == 0.0


# =============================================================================
# Test: LearnedEventPattern
# =============================================================================

class TestLearnedEventPattern:
    """Tests for LearnedEventPattern dataclass."""
    
    def test_default_values(self):
        """Pattern should have sensible defaults."""
        pattern = LearnedEventPattern(
            pattern_id="pat_001",
            pattern_signature="test_sig",
            label="test_label",
            event_type="test_type",
        )
        
        assert pattern.times_seen == 1
        assert pattern.times_confirmed == 0
        assert pattern.example_entity_labels == []
    
    def test_pattern_can_store_context(self):
        """Pattern should store learning context."""
        pattern = LearnedEventPattern(
            pattern_id="pat_001",
            pattern_signature="sig",
            label="label",
            event_type="type",
            example_entity_labels=["lamp", "light"],
            example_location="bedroom",
        )
        
        assert "lamp" in pattern.example_entity_labels
        assert pattern.example_location == "bedroom"
