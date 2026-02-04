"""Schemas for label learning and knowledge emergence.

ARCHITECTURAL INVARIANT: All labels emerge from user interaction.
The agent has no predefined knowledge - it learns from experience.

This module provides:
- LearnedLabel: A label learned from user interaction
- LabelExample: An example instance that helped learn the label
- LabelCategory: Types of things that can be labeled
- LearningContext: Context for when learning occurred
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# LABEL CATEGORIES
# =============================================================================
# 
# These describe WHAT kind of thing is being labeled, not the label itself.
# They help organize the learned vocabulary.
# =============================================================================

# Categories of things that can be labeled
CATEGORY_ENTITY = "entity"          # Physical objects (chair, door, ball)
CATEGORY_LOCATION = "location"      # Rooms/zones (kitchen, hallway)
CATEGORY_EVENT = "event"            # Events/actions (opened, moved)
CATEGORY_STATE = "state"            # States (open, closed, on, off)
CATEGORY_RELATION = "relation"      # Spatial relations (near, inside)
CATEGORY_ATTRIBUTE = "attribute"    # Properties (color, size)
CATEGORY_OTHER = "other"            # Uncategorized


class LearnedLabel(BaseModel):
    """A label learned from user interaction.
    
    Represents a piece of vocabulary the agent has learned through
    experience and user feedback. Includes:
    - The label itself
    - What category of thing it labels
    - Examples that established the label
    - Confidence and usage statistics
    
    ARCHITECTURAL INVARIANT: Labels come from users, not predefined lists.
    """
    
    label_id: str = Field(..., description="Unique identifier for this learned label")
    
    # The label
    label: str = Field(..., description="The learned label string")
    category: str = Field(
        default=CATEGORY_OTHER,
        description="What category this label belongs to",
    )
    
    # Alternative names/aliases
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative labels that map to this one",
    )
    
    # Learning provenance
    learned_at: datetime = Field(
        default_factory=datetime.now,
        description="When this label was first learned",
    )
    learned_from: str = Field(
        default="user",
        description="Source of the label (user, inference, etc.)",
    )
    
    # Example instances that established this label
    examples: list[str] = Field(
        default_factory=list,
        description="IDs of example instances (node IDs, entity GUIDs)",
    )
    
    # Embeddings for recognition (average of examples)
    embedding: list[float] | None = Field(
        default=None,
        description="Feature embedding for recognizing similar things",
    )
    
    # Usage statistics
    times_used: int = Field(default=0, ge=0, description="How often this label has been used")
    times_confirmed: int = Field(default=0, ge=0, description="How often user confirmed this label")
    times_corrected: int = Field(default=0, ge=0, description="How often user corrected this label")
    
    # Confidence (based on confirmation rate)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this label (higher = more confirmed)",
    )
    
    # User verification
    user_verified: bool = Field(
        default=False,
        description="Whether user has explicitly verified this label",
    )
    
    # Associated information
    description: str | None = Field(
        default=None,
        description="User-provided description of what this label means",
    )
    parent_label_id: str | None = Field(
        default=None,
        description="Parent label for hierarchical organization",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}
    
    def update_confidence(self) -> None:
        """Update confidence based on confirmation/correction rate."""
        total = self.times_confirmed + self.times_corrected
        if total > 0:
            # Confidence is confirmation rate, weighted by usage
            base_confidence = self.times_confirmed / total
            # Boost confidence with more usage (log scale)
            import math
            usage_factor = min(1.0, math.log10(total + 1) / 2)
            self.confidence = base_confidence * 0.7 + usage_factor * 0.3
        else:
            self.confidence = 0.5  # Default when no feedback


class LabelExample(BaseModel):
    """An example instance used to learn or reinforce a label.
    
    When the user labels something, we store the example to help
    recognize similar things in the future.
    """
    
    example_id: str = Field(..., description="Unique identifier")
    label_id: str = Field(..., description="ID of the learned label this exemplifies")
    
    # Reference to the actual instance
    instance_id: str = Field(..., description="ID of the instance (node, entity, etc.)")
    instance_type: str = Field(..., description="Type of instance (entity, location, event)")
    
    # When this example was added
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this example was recorded",
    )
    
    # Features for matching
    embedding: list[float] | None = Field(
        default=None,
        description="Feature embedding of this example",
    )
    features: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted features for matching",
    )
    
    # Context when this example was observed
    location_id: str | None = Field(default=None, description="Where this was observed")
    context_description: str | None = Field(default=None, description="Contextual notes")
    
    # Quality metrics
    is_canonical: bool = Field(
        default=False,
        description="Whether this is a canonical/prototypical example",
    )
    user_provided: bool = Field(
        default=True,
        description="Whether user explicitly provided this example",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class LearningRequest(BaseModel):
    """A request for the user to provide a label.
    
    Generated when the agent encounters something it doesn't recognize
    and wants to learn from the user.
    """
    
    request_id: str = Field(..., description="Unique identifier")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this request was created",
    )
    
    # What needs labeling
    category: str = Field(..., description="Category of thing needing a label")
    instance_id: str = Field(..., description="ID of the instance needing a label")
    
    # Context
    prompt: str = Field(..., description="Question to ask the user")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested labels based on similarity",
    )
    
    # Features of what needs labeling
    embedding: list[float] | None = Field(default=None)
    features: dict[str, Any] = Field(default_factory=dict)
    
    # Resolution
    resolved: bool = Field(default=False, description="Whether user has responded")
    response: str | None = Field(default=None, description="User's response")
    resolved_at: datetime | None = Field(default=None)
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class LearningSession(BaseModel):
    """A session of interactive learning.
    
    Tracks a period of user interaction where labels are being learned.
    Useful for reviewing what was learned and providing undo capability.
    """
    
    session_id: str = Field(..., description="Unique identifier")
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When the session started",
    )
    ended_at: datetime | None = Field(
        default=None,
        description="When the session ended",
    )
    
    # What was learned
    labels_learned: list[str] = Field(
        default_factory=list,
        description="IDs of labels learned this session",
    )
    examples_added: list[str] = Field(
        default_factory=list,
        description="IDs of examples added this session",
    )
    corrections_made: int = Field(
        default=0,
        ge=0,
        description="Number of corrections user made",
    )
    
    # Session type
    session_type: str = Field(
        default="interactive",
        description="Type of session (interactive, batch, correction)",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}


class RecognitionResult(BaseModel):
    """Result of attempting to recognize/label something.
    
    When the agent sees something, it tries to match it against
    learned labels. This stores the result of that matching.
    """
    
    # What was being recognized
    instance_id: str = Field(..., description="ID of the instance")
    category: str = Field(..., description="Category being recognized")
    
    # Recognition result
    recognized: bool = Field(
        default=False,
        description="Whether a confident match was found",
    )
    
    # Best matches
    best_label_id: str | None = Field(
        default=None,
        description="ID of best matching learned label",
    )
    best_label: str | None = Field(
        default=None,
        description="String of best matching label",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the recognition",
    )
    
    # Alternative matches
    alternatives: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative matches with scores",
    )
    
    # Whether user confirmation is recommended
    needs_confirmation: bool = Field(
        default=False,
        description="Whether user should confirm this recognition",
    )
    
    # Forward-compatible extras
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields for forward compatibility",
    )
    
    model_config = {"frozen": False}
