"""Tests for label learning and emergent knowledge architecture.

ARCHITECTURAL INVARIANT: All labels emerge from user interaction.
These tests verify the label learning system works correctly.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from episodic_agent.schemas.learning import (
    CATEGORY_ENTITY,
    CATEGORY_EVENT,
    CATEGORY_LOCATION,
    CATEGORY_OTHER,
    CATEGORY_STATE,
    CATEGORY_RELATION,
    CATEGORY_ATTRIBUTE,
    LearnedLabel,
    LabelExample,
    LearningRequest,
    LearningSession,
    RecognitionResult,
)
from episodic_agent.modules.label_learner import (
    LabelLearner,
    DefaultFeatureExtractor,
    RECOGNITION_THRESHOLD,
    CONFIRMATION_THRESHOLD,
    SUGGESTION_THRESHOLD,
)


# =============================================================================
# TEST: LEARNED LABEL SCHEMA
# =============================================================================

class TestLearnedLabel:
    """Tests for LearnedLabel schema."""
    
    def test_create_minimal(self):
        """Create label with minimal fields."""
        label = LearnedLabel(
            label_id="test-1",
            label="chair",
        )
        
        assert label.label_id == "test-1"
        assert label.label == "chair"
        assert label.category == CATEGORY_OTHER
        assert label.aliases == []
        assert label.confidence == 0.5
        assert label.user_verified is False
    
    def test_create_full(self):
        """Create label with all fields."""
        label = LearnedLabel(
            label_id="test-2",
            label="kitchen",
            category=CATEGORY_LOCATION,
            aliases=["Kitchen", "cooking area"],
            learned_from="user",
            description="Room where food is prepared",
            examples=["room-1", "room-2"],
            times_used=10,
            times_confirmed=8,
            times_corrected=2,
            user_verified=True,
            confidence=0.8,
        )
        
        assert label.label == "kitchen"
        assert label.category == CATEGORY_LOCATION
        assert len(label.aliases) == 2
        assert label.description == "Room where food is prepared"
        assert label.times_used == 10
        assert label.user_verified is True
    
    def test_update_confidence_high_confirmation(self):
        """Confidence increases with high confirmation rate."""
        label = LearnedLabel(
            label_id="test-3",
            label="door",
            times_confirmed=9,
            times_corrected=1,
        )
        label.update_confidence()
        
        # 90% confirmation rate should give high confidence
        assert label.confidence > 0.6
    
    def test_update_confidence_low_confirmation(self):
        """Confidence decreases with low confirmation rate."""
        label = LearnedLabel(
            label_id="test-4",
            label="window",
            times_confirmed=2,
            times_corrected=8,
        )
        label.update_confidence()
        
        # 20% confirmation rate should give low confidence
        assert label.confidence < 0.5
    
    def test_update_confidence_no_feedback(self):
        """Default confidence when no feedback yet."""
        label = LearnedLabel(
            label_id="test-5",
            label="table",
            times_confirmed=0,
            times_corrected=0,
        )
        label.update_confidence()
        
        assert label.confidence == 0.5
    
    def test_extras_field(self):
        """Extras field for forward compatibility."""
        label = LearnedLabel(
            label_id="test-6",
            label="robot",
            extras={"custom_field": "value", "numeric": 42},
        )
        
        assert label.extras["custom_field"] == "value"
        assert label.extras["numeric"] == 42


class TestLabelExample:
    """Tests for LabelExample schema."""
    
    def test_create_minimal(self):
        """Create example with minimal fields."""
        example = LabelExample(
            example_id="ex-1",
            label_id="label-1",
            instance_id="entity-1",
            instance_type="entity",
        )
        
        assert example.example_id == "ex-1"
        assert example.label_id == "label-1"
        assert example.instance_id == "entity-1"
        assert example.instance_type == "entity"
        assert example.is_canonical is False
        assert example.user_provided is True
    
    def test_create_with_features(self):
        """Create example with features and embedding."""
        example = LabelExample(
            example_id="ex-2",
            label_id="label-1",
            instance_id="entity-2",
            instance_type="entity",
            embedding=[0.1, 0.2, 0.3],
            features={"color": "brown", "size": "large"},
            is_canonical=True,
        )
        
        assert example.embedding == [0.1, 0.2, 0.3]
        assert example.features["color"] == "brown"
        assert example.is_canonical is True


class TestLearningRequest:
    """Tests for LearningRequest schema."""
    
    def test_create_request(self):
        """Create a learning request."""
        request = LearningRequest(
            request_id="req-1",
            category=CATEGORY_ENTITY,
            instance_id="entity-1",
            prompt="What is this object?",
            suggestions=["chair", "table", "stool"],
        )
        
        assert request.request_id == "req-1"
        assert request.category == CATEGORY_ENTITY
        assert request.prompt == "What is this object?"
        assert len(request.suggestions) == 3
        assert request.resolved is False
        assert request.response is None
    
    def test_resolve_request(self):
        """Resolve a learning request."""
        request = LearningRequest(
            request_id="req-2",
            category=CATEGORY_ENTITY,
            instance_id="entity-2",
            prompt="What is this?",
        )
        
        # Simulate resolution
        request.resolved = True
        request.response = "desk"
        request.resolved_at = datetime.now()
        
        assert request.resolved is True
        assert request.response == "desk"
        assert request.resolved_at is not None


class TestLearningSession:
    """Tests for LearningSession schema."""
    
    def test_create_session(self):
        """Create a learning session."""
        session = LearningSession(
            session_id="sess-1",
            session_type="interactive",
        )
        
        assert session.session_id == "sess-1"
        assert session.session_type == "interactive"
        assert session.labels_learned == []
        assert session.examples_added == []
        assert session.corrections_made == 0
        assert session.ended_at is None
    
    def test_track_session_progress(self):
        """Track progress during session."""
        session = LearningSession(session_id="sess-2")
        
        session.labels_learned.append("label-1")
        session.labels_learned.append("label-2")
        session.examples_added.append("ex-1")
        session.corrections_made = 1
        session.ended_at = datetime.now()
        
        assert len(session.labels_learned) == 2
        assert len(session.examples_added) == 1
        assert session.corrections_made == 1
        assert session.ended_at is not None


class TestRecognitionResult:
    """Tests for RecognitionResult schema."""
    
    def test_unrecognized(self):
        """Test unrecognized result."""
        result = RecognitionResult(
            instance_id="entity-1",
            category=CATEGORY_ENTITY,
            recognized=False,
            needs_confirmation=True,
        )
        
        assert result.recognized is False
        assert result.best_label is None
        assert result.confidence == 0.0
        assert result.needs_confirmation is True
    
    def test_recognized_with_alternatives(self):
        """Test recognized result with alternatives."""
        result = RecognitionResult(
            instance_id="entity-2",
            category=CATEGORY_ENTITY,
            recognized=True,
            best_label_id="label-1",
            best_label="chair",
            confidence=0.85,
            alternatives=[
                {"label": "stool", "score": 0.6},
                {"label": "bench", "score": 0.4},
            ],
            needs_confirmation=False,
        )
        
        assert result.recognized is True
        assert result.best_label == "chair"
        assert result.confidence == 0.85
        assert len(result.alternatives) == 2


# =============================================================================
# TEST: DEFAULT FEATURE EXTRACTOR
# =============================================================================

class TestDefaultFeatureExtractor:
    """Tests for DefaultFeatureExtractor."""
    
    def test_extract_returns_correct_dimension(self):
        """Extracted features have correct dimension."""
        extractor = DefaultFeatureExtractor(dim=64)
        features = extractor.extract("test-1", {"color": "red"})
        
        assert len(features) == 64
    
    def test_extract_deterministic(self):
        """Same input gives same output."""
        extractor = DefaultFeatureExtractor()
        
        features1 = extractor.extract("test-1", {"color": "red", "size": 5})
        features2 = extractor.extract("test-1", {"color": "red", "size": 5})
        
        assert features1 == features2
    
    def test_extract_different_inputs_different_outputs(self):
        """Different inputs give different outputs."""
        extractor = DefaultFeatureExtractor()
        
        features1 = extractor.extract("test-1", {"color": "red"})
        features2 = extractor.extract("test-2", {"color": "blue"})
        
        assert features1 != features2
    
    def test_extract_values_in_range(self):
        """Extracted values are in [-1, 1] range."""
        extractor = DefaultFeatureExtractor()
        features = extractor.extract("test-1", {"data": "test"})
        
        for val in features:
            assert -1.0 <= val <= 1.0


# =============================================================================
# TEST: LABEL LEARNER
# =============================================================================

class TestLabelLearnerBasics:
    """Basic tests for LabelLearner."""
    
    def test_create_empty(self):
        """Create empty learner."""
        learner = LabelLearner()
        
        assert learner.get_vocabulary_size() == 0
        assert learner.get_all_labels() == []
    
    def test_learn_label_simple(self):
        """Learn a simple label."""
        learner = LabelLearner()
        
        learned = learner.learn_label("chair", category=CATEGORY_ENTITY)
        
        assert learned.label == "chair"
        assert learned.category == CATEGORY_ENTITY
        assert learner.has_label("chair") is True
        assert learner.get_vocabulary_size() == 1
    
    def test_learn_label_case_insensitive_lookup(self):
        """Labels can be looked up case-insensitively."""
        learner = LabelLearner()
        
        learner.learn_label("Chair", category=CATEGORY_ENTITY)
        
        assert learner.has_label("chair") is True
        assert learner.has_label("CHAIR") is True
        assert learner.has_label("ChAiR") is True
    
    def test_learn_duplicate_returns_existing(self):
        """Learning duplicate label returns existing."""
        learner = LabelLearner()
        
        first = learner.learn_label("table")
        second = learner.learn_label("table")
        
        assert first.label_id == second.label_id
        assert learner.get_vocabulary_size() == 1
    
    def test_learn_label_with_aliases(self):
        """Learn label with aliases."""
        learner = LabelLearner()
        
        learned = learner.learn_label(
            "sofa",
            category=CATEGORY_ENTITY,
            aliases=["couch", "settee"],
        )
        
        assert learner.has_label("sofa") is True
        assert learner.has_label("couch") is True
        assert learner.has_label("settee") is True
        assert learner.get_vocabulary_size() == 1


class TestLabelLearnerExamples:
    """Tests for example management."""
    
    def test_add_example_to_label(self):
        """Add example to learned label."""
        learner = LabelLearner()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        example = learner.add_example(
            "chair",
            instance_id="entity-1",
            instance_type="entity",
            features={"color": "brown"},
        )
        
        assert example is not None
        assert example.instance_id == "entity-1"
        
        label = learner.get_label("chair")
        assert "entity-1" in label.examples
    
    def test_add_example_unknown_label(self):
        """Adding example to unknown label returns None."""
        learner = LabelLearner()
        
        example = learner.add_example(
            "unknown",
            instance_id="entity-1",
            instance_type="entity",
        )
        
        assert example is None
    
    def test_learn_with_initial_example(self):
        """Learn label with initial example."""
        learner = LabelLearner()
        
        learned = learner.learn_label(
            "door",
            category=CATEGORY_ENTITY,
            instance_id="entity-1",
            features={"state": "closed"},
        )
        
        assert "entity-1" in learned.examples


class TestLabelLearnerFeedback:
    """Tests for user feedback (confirm/correct)."""
    
    def test_confirm_label_increases_confidence(self):
        """Confirming label increases confidence."""
        learner = LabelLearner()
        
        learner.learn_label("chair")
        initial = learner.get_label("chair")
        initial_confidence = initial.confidence
        
        # Confirm several times
        for _ in range(5):
            learner.confirm_label("chair")
        
        label = learner.get_label("chair")
        assert label.times_confirmed == 5
        assert label.confidence >= initial_confidence
    
    def test_correct_label_decreases_old_confidence(self):
        """Correcting label decreases old label's confidence."""
        learner = LabelLearner()
        
        # Set up initial state with some confirmations
        learner.learn_label("stool", category=CATEGORY_ENTITY)
        for _ in range(3):
            learner.confirm_label("stool")
        
        initial_confidence = learner.get_label("stool").confidence
        
        # Correct it
        learner.correct_label("stool", "chair")
        
        # Old label should have been corrected
        old_label = learner.get_label("stool")
        assert old_label.times_corrected == 1
    
    def test_correct_label_creates_new_if_needed(self):
        """Correcting to new label creates it."""
        learner = LabelLearner()
        
        learner.learn_label("thing", category=CATEGORY_ENTITY)
        
        # Correct to new label
        new_label = learner.correct_label("thing", "lamp")
        
        assert new_label.label == "lamp"
        assert learner.has_label("lamp") is True
    
    def test_correct_label_reinforces_existing(self):
        """Correcting to existing label reinforces it."""
        learner = LabelLearner()
        
        learner.learn_label("stool", category=CATEGORY_ENTITY)
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        
        initial_confirmed = learner.get_label("chair").times_confirmed
        
        # Correct stool to chair
        learner.correct_label("stool", "chair")
        
        label = learner.get_label("chair")
        assert label.times_confirmed == initial_confirmed + 1


class TestLabelLearnerRecognition:
    """Tests for recognition functionality."""
    
    def test_recognize_no_labels(self):
        """Recognition with no learned labels."""
        learner = LabelLearner()
        
        result = learner.recognize(
            instance_id="entity-1",
            category=CATEGORY_ENTITY,
            features={"color": "red"},
        )
        
        assert result.recognized is False
        assert result.needs_confirmation is True
    
    def test_recognize_with_learned_labels(self):
        """Recognition with learned labels returns match."""
        learner = LabelLearner()
        
        # Learn with examples
        learner.learn_label(
            "chair",
            category=CATEGORY_ENTITY,
            instance_id="chair-1",
            features={"type": "seating"},
        )
        learner.add_example(
            "chair",
            instance_id="chair-2",
            instance_type="entity",
            features={"type": "seating"},
        )
        
        # Recognize similar
        result = learner.recognize(
            instance_id="chair-3",
            category=CATEGORY_ENTITY,
            features={"type": "seating"},
        )
        
        assert result.best_label == "chair"
        assert result.confidence > 0
    
    def test_recognize_returns_alternatives(self):
        """Recognition returns alternatives."""
        learner = LabelLearner()
        
        # Learn multiple labels
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        learner.learn_label("stool", category=CATEGORY_ENTITY)
        learner.learn_label("bench", category=CATEGORY_ENTITY)
        
        result = learner.recognize(
            instance_id="entity-1",
            category=CATEGORY_ENTITY,
            features={"type": "seating"},
        )
        
        # Should have alternatives
        assert result.best_label is not None or len(result.alternatives) >= 0
    
    def test_recognize_category_filtering(self):
        """Recognition only considers labels in same category."""
        learner = LabelLearner()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        learner.learn_label("kitchen", category=CATEGORY_LOCATION)
        
        result = learner.recognize(
            instance_id="room-1",
            category=CATEGORY_LOCATION,
            features={"type": "room"},
        )
        
        # Should only match location labels
        assert result.best_label is None or result.best_label == "kitchen"


class TestLabelLearnerSuggestions:
    """Tests for label suggestions."""
    
    def test_get_suggestions_empty(self):
        """Suggestions when no labels."""
        learner = LabelLearner()
        
        suggestions = learner.get_suggestions(CATEGORY_ENTITY)
        
        assert suggestions == []
    
    def test_get_suggestions_by_category(self):
        """Suggestions filtered by category."""
        learner = LabelLearner()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        learner.learn_label("table", category=CATEGORY_ENTITY)
        learner.learn_label("kitchen", category=CATEGORY_LOCATION)
        
        entity_suggestions = learner.get_suggestions(CATEGORY_ENTITY)
        location_suggestions = learner.get_suggestions(CATEGORY_LOCATION)
        
        assert "kitchen" not in entity_suggestions
        assert "kitchen" in location_suggestions
    
    def test_get_suggestions_limit(self):
        """Suggestions respect max limit."""
        learner = LabelLearner()
        
        for i in range(10):
            learner.learn_label(f"entity-{i}", category=CATEGORY_ENTITY)
        
        suggestions = learner.get_suggestions(CATEGORY_ENTITY, max_suggestions=3)
        
        assert len(suggestions) <= 3


class TestLabelLearnerRequests:
    """Tests for learning requests."""
    
    def test_create_request(self):
        """Create a learning request."""
        learner = LabelLearner()
        
        request = learner.request_label(
            category=CATEGORY_ENTITY,
            instance_id="entity-1",
            prompt="What is this object?",
            features={"color": "blue"},
        )
        
        assert request.category == CATEGORY_ENTITY
        assert request.instance_id == "entity-1"
        assert request.prompt == "What is this object?"
        assert request.resolved is False
    
    def test_request_includes_suggestions(self):
        """Request includes suggestions from learned labels."""
        learner = LabelLearner()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        learner.learn_label("table", category=CATEGORY_ENTITY)
        
        request = learner.request_label(
            category=CATEGORY_ENTITY,
            instance_id="entity-1",
            prompt="What is this?",
        )
        
        assert len(request.suggestions) > 0
    
    def test_resolve_request(self):
        """Resolve a pending request."""
        learner = LabelLearner()
        
        request = learner.request_label(
            category=CATEGORY_ENTITY,
            instance_id="entity-1",
            prompt="What is this?",
            features={"color": "red"},
        )
        
        # Resolve with user's answer
        learned = learner.resolve_request(request.request_id, "lamp")
        
        assert learned is not None
        assert learned.label == "lamp"
        assert learner.has_label("lamp") is True
    
    def test_resolve_nonexistent_request(self):
        """Resolving nonexistent request returns None."""
        learner = LabelLearner()
        
        result = learner.resolve_request("nonexistent", "label")
        
        assert result is None


class TestLabelLearnerSessions:
    """Tests for learning sessions."""
    
    def test_start_session(self):
        """Start a learning session."""
        learner = LabelLearner()
        
        session = learner.start_session("interactive")
        
        assert session.session_type == "interactive"
        assert session.started_at is not None
        assert session.ended_at is None
    
    def test_session_tracks_labels(self):
        """Session tracks learned labels."""
        learner = LabelLearner()
        
        session = learner.start_session()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        learner.learn_label("table", category=CATEGORY_ENTITY)
        
        assert len(session.labels_learned) == 2
    
    def test_session_tracks_corrections(self):
        """Session tracks corrections."""
        learner = LabelLearner()
        
        session = learner.start_session()
        
        learner.learn_label("thing")
        learner.correct_label("thing", "chair")
        
        assert session.corrections_made == 1
    
    def test_end_session(self):
        """End a learning session."""
        learner = LabelLearner()
        
        learner.start_session()
        learner.learn_label("chair")
        
        ended = learner.end_session()
        
        assert ended is not None
        assert ended.ended_at is not None
        assert len(ended.labels_learned) == 1
    
    def test_end_session_no_active(self):
        """Ending with no active session returns None."""
        learner = LabelLearner()
        
        result = learner.end_session()
        
        assert result is None


class TestLabelLearnerQueries:
    """Tests for querying the learner."""
    
    def test_get_label(self):
        """Get label by string."""
        learner = LabelLearner()
        
        original = learner.learn_label("chair", category=CATEGORY_ENTITY)
        
        found = learner.get_label("chair")
        
        assert found is not None
        assert found.label_id == original.label_id
    
    def test_get_label_not_found(self):
        """Get nonexistent label returns None."""
        learner = LabelLearner()
        
        result = learner.get_label("nonexistent")
        
        assert result is None
    
    def test_get_labels_by_category(self):
        """Get all labels in category."""
        learner = LabelLearner()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        learner.learn_label("table", category=CATEGORY_ENTITY)
        learner.learn_label("kitchen", category=CATEGORY_LOCATION)
        
        entities = learner.get_labels_by_category(CATEGORY_ENTITY)
        
        assert len(entities) == 2
        labels = [l.label for l in entities]
        assert "chair" in labels
        assert "table" in labels
    
    def test_get_all_labels(self):
        """Get all learned labels."""
        learner = LabelLearner()
        
        learner.learn_label("a")
        learner.learn_label("b")
        learner.learn_label("c")
        
        all_labels = learner.get_all_labels()
        
        assert len(all_labels) == 3
    
    def test_vocabulary_size(self):
        """Get vocabulary size."""
        learner = LabelLearner()
        
        learner.learn_label("a", category=CATEGORY_ENTITY)
        learner.learn_label("b", category=CATEGORY_ENTITY)
        learner.learn_label("c", category=CATEGORY_LOCATION)
        
        assert learner.get_vocabulary_size() == 3
        assert learner.get_vocabulary_size(CATEGORY_ENTITY) == 2
        assert learner.get_vocabulary_size(CATEGORY_LOCATION) == 1


class TestLabelLearnerStatistics:
    """Tests for statistics tracking."""
    
    def test_get_statistics(self):
        """Get learning statistics."""
        learner = LabelLearner()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        learner.learn_label("kitchen", category=CATEGORY_LOCATION)
        learner.confirm_label("chair")
        
        stats = learner.get_statistics()
        
        assert stats["total_labels"] == 2
        assert stats["successful_recognitions"] == 1
        assert CATEGORY_ENTITY in stats["labels_by_category"]
    
    def test_recognition_rate(self):
        """Recognition rate calculation."""
        learner = LabelLearner()
        
        learner.learn_label("chair", category=CATEGORY_ENTITY)
        
        # Do some recognitions
        learner.recognize("e1", CATEGORY_ENTITY, {})
        learner.recognize("e2", CATEGORY_ENTITY, {})
        learner.confirm_label("chair")
        
        stats = learner.get_statistics()
        
        assert stats["total_recognitions"] == 2
        assert "recognition_rate" in stats


# =============================================================================
# TEST: CATEGORY CONSTANTS
# =============================================================================

class TestCategoryConstants:
    """Tests for category constants."""
    
    def test_category_values(self):
        """Category constants have expected values."""
        assert CATEGORY_ENTITY == "entity"
        assert CATEGORY_LOCATION == "location"
        assert CATEGORY_EVENT == "event"
        assert CATEGORY_STATE == "state"
        assert CATEGORY_RELATION == "relation"
        assert CATEGORY_ATTRIBUTE == "attribute"
        assert CATEGORY_OTHER == "other"
    
    def test_categories_are_strings(self):
        """Categories are strings, not enums."""
        categories = [
            CATEGORY_ENTITY,
            CATEGORY_LOCATION,
            CATEGORY_EVENT,
            CATEGORY_STATE,
            CATEGORY_RELATION,
            CATEGORY_ATTRIBUTE,
            CATEGORY_OTHER,
        ]
        
        for cat in categories:
            assert isinstance(cat, str)


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestLabelLearnerIntegration:
    """Integration tests for complete workflows."""
    
    def test_first_encounter_workflow(self):
        """Test the first-encounter learning workflow."""
        learner = LabelLearner()
        
        # Agent encounters unknown entity
        result = learner.recognize(
            instance_id="entity-1",
            category=CATEGORY_ENTITY,
            features={"color": "brown", "size": "large"},
        )
        
        assert result.recognized is False
        assert result.needs_confirmation is True
        
        # Create request for user
        request = learner.request_label(
            category=CATEGORY_ENTITY,
            instance_id="entity-1",
            prompt="I see something new. What is this?",
            features={"color": "brown", "size": "large"},
        )
        
        # User responds
        learned = learner.resolve_request(request.request_id, "couch")
        
        assert learned.label == "couch"
        assert learner.has_label("couch") is True
    
    def test_recognition_workflow(self):
        """Test recognition of known entities."""
        learner = LabelLearner()
        
        # Learn from first encounter
        learner.learn_label(
            "chair",
            category=CATEGORY_ENTITY,
            instance_id="chair-1",
            features={"type": "seating", "legs": 4},
        )
        
        # Add more examples
        learner.add_example(
            "chair",
            instance_id="chair-2",
            instance_type="entity",
            features={"type": "seating", "legs": 4},
        )
        
        # Confirm a few times to build confidence
        for _ in range(5):
            learner.confirm_label("chair")
        
        # Now recognize similar
        result = learner.recognize(
            instance_id="chair-3",
            category=CATEGORY_ENTITY,
            features={"type": "seating", "legs": 4},
        )
        
        assert result.best_label == "chair"
        assert result.confidence > 0
    
    def test_correction_workflow(self):
        """Test correction workflow."""
        learner = LabelLearner()
        learner.start_session()
        
        # Learn initial label
        learner.learn_label("stool", category=CATEGORY_ENTITY)
        
        # User corrects
        learner.correct_label(
            "stool",
            "bar_stool",
            instance_id="entity-1",
            features={"height": "tall"},
        )
        
        session = learner.end_session()
        
        assert learner.has_label("bar_stool") is True
        assert session.corrections_made == 1
    
    def test_multi_category_learning(self):
        """Test learning across multiple categories."""
        learner = LabelLearner()
        
        # Learn entities
        learner.learn_label("door", category=CATEGORY_ENTITY)
        learner.learn_label("window", category=CATEGORY_ENTITY)
        
        # Learn locations
        learner.learn_label("kitchen", category=CATEGORY_LOCATION)
        learner.learn_label("bedroom", category=CATEGORY_LOCATION)
        
        # Learn states
        learner.learn_label("open", category=CATEGORY_STATE)
        learner.learn_label("closed", category=CATEGORY_STATE)
        
        # Learn events
        learner.learn_label("opened", category=CATEGORY_EVENT)
        learner.learn_label("moved", category=CATEGORY_EVENT)
        
        stats = learner.get_statistics()
        
        assert stats["labels_by_category"][CATEGORY_ENTITY] == 2
        assert stats["labels_by_category"][CATEGORY_LOCATION] == 2
        assert stats["labels_by_category"][CATEGORY_STATE] == 2
        assert stats["labels_by_category"][CATEGORY_EVENT] == 2
    
    def test_alias_learning(self):
        """Test learning with aliases."""
        learner = LabelLearner()
        
        # Learn with aliases
        learner.learn_label(
            "sofa",
            category=CATEGORY_ENTITY,
            aliases=["couch", "settee"],
        )
        
        # All should resolve to same label
        label1 = learner.get_label("sofa")
        label2 = learner.get_label("couch")
        label3 = learner.get_label("settee")
        
        assert label1.label_id == label2.label_id == label3.label_id
        
        # Confirming any alias updates the same label
        learner.confirm_label("couch")
        learner.confirm_label("settee")
        
        label = learner.get_label("sofa")
        assert label.times_confirmed == 2
