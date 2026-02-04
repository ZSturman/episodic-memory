"""Label learning module for emergent knowledge architecture.

ARCHITECTURAL INVARIANT: All labels emerge from user interaction.
The agent starts with NO predefined vocabulary and learns everything
through observation and user feedback.

This module provides:
- LabelLearner: Core class that manages learning from user interaction
- Recognition: Matching observations against learned labels
- Suggestions: Providing label suggestions based on similarity

Key concepts:
- First encounter: Agent sees something new, asks user for label
- Reinforcement: Agent recognizes something, user confirms or corrects
- Generalization: Agent uses examples to recognize similar things
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from episodic_agent.schemas.learning import (
    CATEGORY_ENTITY,
    CATEGORY_EVENT,
    CATEGORY_LOCATION,
    CATEGORY_OTHER,
    CATEGORY_STATE,
    LabelExample,
    LearnedLabel,
    LearningRequest,
    LearningSession,
    RecognitionResult,
)

if TYPE_CHECKING:
    from episodic_agent.modules.dialog import DialogManager


# =============================================================================
# CONFIGURATION
# =============================================================================

# Confidence thresholds for recognition
RECOGNITION_THRESHOLD = 0.7       # Minimum confidence to auto-recognize
CONFIRMATION_THRESHOLD = 0.5      # Below this, always ask for confirmation
SUGGESTION_THRESHOLD = 0.3        # Minimum similarity to suggest a label

# Learning parameters
MIN_EXAMPLES_FOR_CONFIDENCE = 3   # Need this many examples for high confidence
EMBEDDING_DIM = 128               # Default embedding dimension


# =============================================================================
# FEATURE EXTRACTION PROTOCOL
# =============================================================================

class FeatureExtractor(Protocol):
    """Protocol for extracting features from observations.
    
    Features are used to match new observations against learned labels.
    Different extractors can be used for different categories.
    """
    
    def extract(self, instance_id: str, data: dict[str, Any]) -> list[float]:
        """Extract feature embedding from instance data.
        
        Args:
            instance_id: ID of the instance
            data: Raw observation data
            
        Returns:
            Feature embedding as list of floats
        """
        ...


class DefaultFeatureExtractor:
    """Default feature extractor using basic hashing.
    
    This is a placeholder - real implementations would use
    embeddings from visual features, semantic models, etc.
    """
    
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
    
    def extract(self, instance_id: str, data: dict[str, Any]) -> list[float]:
        """Extract simple hash-based features."""
        # This is a placeholder - real implementation would use
        # visual embeddings, semantic features, etc.
        import hashlib
        
        # Create deterministic hash from data
        data_str = str(sorted(data.items()))
        hash_bytes = hashlib.sha256(data_str.encode()).digest()
        
        # Convert to float vector
        embedding = []
        for i in range(min(self.dim, len(hash_bytes))):
            embedding.append((hash_bytes[i] - 128) / 128.0)
        
        # Pad if needed
        while len(embedding) < self.dim:
            embedding.append(0.0)
        
        return embedding


# =============================================================================
# LABEL LEARNER
# =============================================================================

class LabelLearner:
    """Manages label learning from user interaction.
    
    Core component of the emergent knowledge architecture. Responsible for:
    
    1. Tracking learned labels and examples
    2. Recognizing observations against learned vocabulary
    3. Requesting labels from users for unknown things
    4. Building confidence through user feedback
    
    ARCHITECTURAL INVARIANT: Starts empty. All knowledge comes from users.
    """
    
    def __init__(
        self,
        dialog: DialogManager | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ):
        """Initialize the label learner.
        
        Args:
            dialog: Dialog manager for user interaction (optional for testing)
            feature_extractor: Feature extractor for embeddings
        """
        self._dialog = dialog
        self._feature_extractor = feature_extractor or DefaultFeatureExtractor()
        
        # Learned labels by ID
        self._labels: dict[str, LearnedLabel] = {}
        
        # Index: label string -> label ID
        self._label_index: dict[str, str] = {}
        
        # Index: alias -> label ID
        self._alias_index: dict[str, str] = {}
        
        # Index: category -> list of label IDs
        self._category_index: dict[str, list[str]] = {}
        
        # Examples by ID
        self._examples: dict[str, LabelExample] = {}
        
        # Index: label_id -> list of example IDs
        self._label_examples: dict[str, list[str]] = {}
        
        # Pending learning requests
        self._pending_requests: dict[str, LearningRequest] = {}
        
        # Current learning session
        self._current_session: LearningSession | None = None
        
        # Statistics
        self._total_recognitions = 0
        self._successful_recognitions = 0
        self._user_corrections = 0
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Learning
    # -------------------------------------------------------------------------
    
    def learn_label(
        self,
        label: str,
        category: str = CATEGORY_OTHER,
        instance_id: str | None = None,
        features: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
        description: str | None = None,
        learned_from: str = "user",
    ) -> LearnedLabel:
        """Learn a new label from user interaction.
        
        This is the primary way labels enter the system. When a user
        provides a label for something, we:
        1. Create a LearnedLabel entry
        2. Optionally add the instance as an example
        3. Index the label and aliases for lookup
        
        Args:
            label: The label string to learn
            category: What kind of thing this labels
            instance_id: Optional ID of instance that prompted learning
            features: Optional features of the instance
            aliases: Optional alternative names
            description: Optional description of what this label means
            learned_from: Source of the label
            
        Returns:
            The created LearnedLabel
        """
        # Check if we already have this label
        existing_id = self._label_index.get(label.lower())
        if existing_id:
            existing = self._labels[existing_id]
            # Add as alias if different case
            if label != existing.label and label not in existing.aliases:
                existing.aliases.append(label)
                self._alias_index[label.lower()] = existing_id
            # Maybe add example
            if instance_id and features:
                self._add_example(existing_id, instance_id, category, features)
            return existing
        
        # Create new learned label
        label_id = str(uuid.uuid4())
        learned = LearnedLabel(
            label_id=label_id,
            label=label,
            category=category,
            aliases=aliases or [],
            learned_at=datetime.now(),
            learned_from=learned_from,
            description=description,
            user_verified=learned_from == "user",
            confidence=0.5 if learned_from == "user" else 0.3,
        )
        
        # Store
        self._labels[label_id] = learned
        self._label_index[label.lower()] = label_id
        
        # Index aliases
        for alias in learned.aliases:
            self._alias_index[alias.lower()] = label_id
        
        # Index by category
        if category not in self._category_index:
            self._category_index[category] = []
        self._category_index[category].append(label_id)
        
        # Add example if provided
        if instance_id and features:
            self._add_example(label_id, instance_id, category, features)
        
        # Track in session
        if self._current_session:
            self._current_session.labels_learned.append(label_id)
        
        return learned
    
    def add_example(
        self,
        label: str,
        instance_id: str,
        instance_type: str,
        features: dict[str, Any] | None = None,
        is_canonical: bool = False,
    ) -> LabelExample | None:
        """Add an example instance for a label.
        
        Examples help the agent recognize similar things in the future.
        More examples = better generalization.
        
        Args:
            label: The label to add example for
            instance_id: ID of the example instance
            instance_type: Type of instance
            features: Features of the instance
            is_canonical: Whether this is a prototypical example
            
        Returns:
            The created example, or None if label not found
        """
        label_id = self._find_label_id(label)
        if not label_id:
            return None
        
        return self._add_example(
            label_id,
            instance_id,
            instance_type,
            features or {},
            is_canonical=is_canonical,
        )
    
    def confirm_label(self, label: str, instance_id: str | None = None) -> bool:
        """Confirm a label is correct (positive feedback).
        
        Called when user confirms the agent's recognition is correct.
        Increases confidence in the label.
        
        Args:
            label: The label that was confirmed
            instance_id: Optional ID of confirmed instance
            
        Returns:
            True if label was found and confirmed
        """
        label_id = self._find_label_id(label)
        if not label_id:
            return False
        
        learned = self._labels[label_id]
        learned.times_used += 1
        learned.times_confirmed += 1
        learned.update_confidence()
        
        self._successful_recognitions += 1
        
        return True
    
    def correct_label(
        self,
        old_label: str,
        new_label: str,
        instance_id: str | None = None,
        features: dict[str, Any] | None = None,
    ) -> LearnedLabel:
        """Correct a misrecognized label (negative feedback).
        
        Called when user corrects the agent's recognition.
        Decreases confidence in old label, learns/reinforces new label.
        
        Args:
            old_label: The incorrect label
            new_label: The correct label from user
            instance_id: ID of the misrecognized instance
            features: Features of the instance
            
        Returns:
            The correct label (created or updated)
        """
        # Decrease confidence in old label
        old_id = self._find_label_id(old_label)
        if old_id:
            old = self._labels[old_id]
            old.times_used += 1
            old.times_corrected += 1
            old.update_confidence()
        
        # Learn or reinforce the correct label
        new_id = self._find_label_id(new_label)
        if new_id:
            # Existing label - add example
            learned = self._labels[new_id]
            learned.times_confirmed += 1
            learned.update_confidence()
            if instance_id and features:
                category = learned.category
                self._add_example(new_id, instance_id, category, features)
        else:
            # New label
            category = self._labels[old_id].category if old_id else CATEGORY_OTHER
            learned = self.learn_label(
                new_label,
                category=category,
                instance_id=instance_id,
                features=features,
            )
        
        self._user_corrections += 1
        if self._current_session:
            self._current_session.corrections_made += 1
        
        return learned
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Recognition
    # -------------------------------------------------------------------------
    
    def recognize(
        self,
        instance_id: str,
        category: str,
        features: dict[str, Any],
    ) -> RecognitionResult:
        """Attempt to recognize/label something.
        
        The core recognition function. Given an observation:
        1. Extract features
        2. Compare against learned examples
        3. Return best match with confidence
        
        Args:
            instance_id: ID of the instance to recognize
            category: Category of thing being recognized
            features: Features of the instance
            
        Returns:
            RecognitionResult with match details
        """
        self._total_recognitions += 1
        
        # Get learned labels in this category
        candidate_ids = self._category_index.get(category, [])
        if not candidate_ids:
            return RecognitionResult(
                instance_id=instance_id,
                category=category,
                recognized=False,
                needs_confirmation=True,
            )
        
        # Extract embedding
        embedding = self._feature_extractor.extract(instance_id, features)
        
        # Find best matches
        matches = []
        for label_id in candidate_ids:
            label = self._labels[label_id]
            score = self._compute_similarity(embedding, label)
            matches.append({
                "label_id": label_id,
                "label": label.label,
                "score": score,
                "confidence": label.confidence,
            })
        
        # Sort by score
        matches.sort(key=lambda m: m["score"], reverse=True)
        
        # Best match
        if matches:
            best = matches[0]
            combined_confidence = best["score"] * best["confidence"]
            
            return RecognitionResult(
                instance_id=instance_id,
                category=category,
                recognized=combined_confidence >= RECOGNITION_THRESHOLD,
                best_label_id=best["label_id"],
                best_label=best["label"],
                confidence=combined_confidence,
                alternatives=matches[1:5],  # Top 5 alternatives
                needs_confirmation=combined_confidence < CONFIRMATION_THRESHOLD,
            )
        
        return RecognitionResult(
            instance_id=instance_id,
            category=category,
            recognized=False,
            needs_confirmation=True,
        )
    
    def get_suggestions(
        self,
        category: str,
        features: dict[str, Any] | None = None,
        max_suggestions: int = 5,
    ) -> list[str]:
        """Get label suggestions for a category.
        
        Used when asking user for a label - provides suggestions
        based on existing labels and similarity.
        
        Args:
            category: Category to get suggestions for
            features: Optional features to match against
            max_suggestions: Maximum suggestions to return
            
        Returns:
            List of suggested labels
        """
        # Get labels in category
        label_ids = self._category_index.get(category, [])
        
        if not features or not label_ids:
            # Just return most confident labels
            labels = [(self._labels[lid], lid) for lid in label_ids]
            labels.sort(key=lambda x: x[0].confidence, reverse=True)
            return [l[0].label for l in labels[:max_suggestions]]
        
        # Score by similarity
        embedding = self._feature_extractor.extract("temp", features)
        scored = []
        for label_id in label_ids:
            label = self._labels[label_id]
            score = self._compute_similarity(embedding, label)
            if score >= SUGGESTION_THRESHOLD:
                scored.append((label.label, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:max_suggestions]]
    
    # -------------------------------------------------------------------------
    # PUBLIC API: User Interaction
    # -------------------------------------------------------------------------
    
    def request_label(
        self,
        category: str,
        instance_id: str,
        prompt: str,
        features: dict[str, Any] | None = None,
    ) -> LearningRequest:
        """Create a request for user to provide a label.
        
        When the agent encounters something it doesn't recognize,
        it creates a learning request to ask the user.
        
        Args:
            category: Category of thing needing label
            instance_id: ID of the instance
            prompt: Question to ask user
            features: Features of the instance
            
        Returns:
            LearningRequest object
        """
        request_id = str(uuid.uuid4())
        
        # Get suggestions
        suggestions = self.get_suggestions(category, features)
        
        # Extract embedding
        embedding = None
        if features:
            embedding = self._feature_extractor.extract(instance_id, features)
        
        request = LearningRequest(
            request_id=request_id,
            category=category,
            instance_id=instance_id,
            prompt=prompt,
            suggestions=suggestions,
            embedding=embedding,
            features=features or {},
        )
        
        self._pending_requests[request_id] = request
        
        return request
    
    async def ask_user_for_label(
        self,
        category: str,
        instance_id: str,
        prompt: str,
        features: dict[str, Any] | None = None,
    ) -> str | None:
        """Ask user for a label interactively.
        
        Convenience method that creates request and waits for response.
        
        Args:
            category: Category of thing needing label
            instance_id: ID of the instance
            prompt: Question to ask user
            features: Features of the instance
            
        Returns:
            User's label response, or None if no dialog
        """
        if not self._dialog:
            return None
        
        # Get suggestions
        suggestions = self.get_suggestions(category, features)
        
        # Ask user
        response = await self._dialog.ask_label(prompt, suggestions)
        
        if response:
            # Learn the label
            self.learn_label(
                response,
                category=category,
                instance_id=instance_id,
                features=features,
            )
        
        return response
    
    def resolve_request(self, request_id: str, response: str) -> LearnedLabel | None:
        """Resolve a pending learning request with user's response.
        
        Args:
            request_id: ID of the request
            response: User's label response
            
        Returns:
            The learned label, or None if request not found
        """
        request = self._pending_requests.get(request_id)
        if not request:
            return None
        
        # Mark resolved
        request.resolved = True
        request.response = response
        request.resolved_at = datetime.now()
        
        # Learn the label
        learned = self.learn_label(
            response,
            category=request.category,
            instance_id=request.instance_id,
            features=request.features,
        )
        
        # Clean up
        del self._pending_requests[request_id]
        
        return learned
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Sessions
    # -------------------------------------------------------------------------
    
    def start_session(self, session_type: str = "interactive") -> LearningSession:
        """Start a learning session.
        
        Sessions track what was learned during a period of interaction,
        useful for reviewing and potentially undoing changes.
        
        Args:
            session_type: Type of session
            
        Returns:
            The new session
        """
        self._current_session = LearningSession(
            session_id=str(uuid.uuid4()),
            session_type=session_type,
        )
        return self._current_session
    
    def end_session(self) -> LearningSession | None:
        """End the current learning session.
        
        Returns:
            The completed session, or None if no session
        """
        if not self._current_session:
            return None
        
        session = self._current_session
        session.ended_at = datetime.now()
        self._current_session = None
        return session
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Queries
    # -------------------------------------------------------------------------
    
    def get_label(self, label: str) -> LearnedLabel | None:
        """Get a learned label by string.
        
        Args:
            label: Label string to look up
            
        Returns:
            LearnedLabel if found
        """
        label_id = self._find_label_id(label)
        if label_id:
            return self._labels[label_id]
        return None
    
    def get_labels_by_category(self, category: str) -> list[LearnedLabel]:
        """Get all labels in a category.
        
        Args:
            category: Category to query
            
        Returns:
            List of labels in that category
        """
        label_ids = self._category_index.get(category, [])
        return [self._labels[lid] for lid in label_ids]
    
    def get_all_labels(self) -> list[LearnedLabel]:
        """Get all learned labels.
        
        Returns:
            List of all labels
        """
        return list(self._labels.values())
    
    def has_label(self, label: str) -> bool:
        """Check if a label has been learned.
        
        Args:
            label: Label to check
            
        Returns:
            True if label is known
        """
        return self._find_label_id(label) is not None
    
    def get_vocabulary_size(self, category: str | None = None) -> int:
        """Get the size of learned vocabulary.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Number of learned labels
        """
        if category:
            return len(self._category_index.get(category, []))
        return len(self._labels)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get learning statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_labels": len(self._labels),
            "total_examples": len(self._examples),
            "total_recognitions": self._total_recognitions,
            "successful_recognitions": self._successful_recognitions,
            "recognition_rate": (
                self._successful_recognitions / self._total_recognitions
                if self._total_recognitions > 0 else 0.0
            ),
            "user_corrections": self._user_corrections,
            "labels_by_category": {
                cat: len(ids) for cat, ids in self._category_index.items()
            },
            "pending_requests": len(self._pending_requests),
        }
    
    # -------------------------------------------------------------------------
    # INTERNAL: Helpers
    # -------------------------------------------------------------------------
    
    def _find_label_id(self, label: str) -> str | None:
        """Find label ID by label string or alias."""
        lower = label.lower()
        return self._label_index.get(lower) or self._alias_index.get(lower)
    
    def _add_example(
        self,
        label_id: str,
        instance_id: str,
        instance_type: str,
        features: dict[str, Any],
        is_canonical: bool = False,
    ) -> LabelExample:
        """Add an example for a label."""
        example_id = str(uuid.uuid4())
        
        # Extract embedding
        embedding = self._feature_extractor.extract(instance_id, features)
        
        example = LabelExample(
            example_id=example_id,
            label_id=label_id,
            instance_id=instance_id,
            instance_type=instance_type,
            embedding=embedding,
            features=features,
            is_canonical=is_canonical,
        )
        
        # Store
        self._examples[example_id] = example
        
        # Index
        if label_id not in self._label_examples:
            self._label_examples[label_id] = []
        self._label_examples[label_id].append(example_id)
        
        # Update label embedding (average of examples)
        self._update_label_embedding(label_id)
        
        # Track in label
        label = self._labels[label_id]
        label.examples.append(instance_id)
        
        # Track in session
        if self._current_session:
            self._current_session.examples_added.append(example_id)
        
        return example
    
    def _update_label_embedding(self, label_id: str) -> None:
        """Update label embedding as average of examples."""
        example_ids = self._label_examples.get(label_id, [])
        if not example_ids:
            return
        
        # Average embeddings
        embeddings = [
            self._examples[eid].embedding
            for eid in example_ids
            if self._examples[eid].embedding
        ]
        
        if not embeddings:
            return
        
        # Compute mean
        dim = len(embeddings[0])
        mean_embedding = [0.0] * dim
        for emb in embeddings:
            for i, val in enumerate(emb):
                mean_embedding[i] += val
        for i in range(dim):
            mean_embedding[i] /= len(embeddings)
        
        self._labels[label_id].embedding = mean_embedding
    
    def _compute_similarity(
        self,
        embedding: list[float],
        label: LearnedLabel,
    ) -> float:
        """Compute similarity between embedding and label."""
        if not label.embedding:
            # No embedding - use number of examples as proxy
            num_examples = len(self._label_examples.get(label.label_id, []))
            return 0.3 if num_examples > 0 else 0.1
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding, label.embedding))
        
        norm_a = math.sqrt(sum(a * a for a in embedding))
        norm_b = math.sqrt(sum(b * b for b in label.embedding))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Normalize to 0-1
        return (similarity + 1) / 2
