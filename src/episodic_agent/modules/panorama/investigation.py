"""Investigation state machine for panorama label gating.

Controls *when* the agent should request a label from the user.  The
key insight is that a label request should never fire on the first
ambiguous frame — the agent must first accumulate evidence across
multiple viewports, detect a confidence plateau, and only then surface
a request with a full evidence bundle attached.

States
------
- ``investigating_unknown`` — first encounter, gathering evidence
- ``matching_known`` — scene resembles a known location
- ``low_confidence_match`` — best candidate is weak
- ``confident_match`` — high confidence in identity
- ``novel_location_candidate`` — evidence points to a new location
- ``label_request`` — evidence bundle ready, requesting label

Transitions are adaptive: the investigation window length depends on
how quickly confidence stabilises, bounded by configurable min/max.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from episodic_agent.schemas.panorama_events import (
    EvidenceBundle,
    MatchCandidate,
    MatchEvaluation,
    PanoramaAgentState,
    PanoramaEventType,
    StateTransitionPayload,
)

logger = logging.getLogger(__name__)


class InvestigationStateMachine:
    """Adaptive investigation gate for label requests.

    The machine tracks confidence evolution and only permits a
    ``label_request`` when:

    1. The investigation has run for at least ``min_investigation_steps``.
    2. Confidence has plateaued (rolling std < ``plateau_threshold``).
    3. The best candidate's confidence is below ``label_request_ceiling``.

    If the best candidate rises above ``confident_match_threshold``,
    the machine transitions directly to ``confident_match`` without
    ever requesting a label.

    Parameters
    ----------
    plateau_threshold : float
        Standard deviation below which confidence is considered settled.
    min_investigation_steps : int
        Minimum steps before any label request can fire.
    max_investigation_steps : int
        Hard cap — force a decision after this many steps.
    label_request_ceiling : float
        Only request a label if best match confidence stays below this.
    confident_match_threshold : float
        Confidence above which we skip investigation and confirm.
    plateau_window : int
        How many recent confidence values to use for plateau detection.
    """

    def __init__(
        self,
        plateau_threshold: float = 0.05,
        min_investigation_steps: int = 5,
        max_investigation_steps: int = 20,
        label_request_ceiling: float = 0.4,
        confident_match_threshold: float = 0.7,
        plateau_window: int = 5,
        event_bus: Any = None,
    ) -> None:
        self._plateau_threshold = plateau_threshold
        self._min_steps = min_investigation_steps
        self._max_steps = max_investigation_steps
        self._label_request_ceiling = label_request_ceiling
        self._confident_threshold = confident_match_threshold
        self._plateau_window = plateau_window
        self._event_bus = event_bus

        # Internal state
        self._state = PanoramaAgentState.investigating_unknown
        self._previous_state = PanoramaAgentState.investigating_unknown
        self._steps_in_state: int = 0
        self._investigation_step_count: int = 0

        # Evidence accumulation
        self._confidence_history: deque[float] = deque(maxlen=200)
        self._viewport_images: deque[str] = deque(maxlen=12)
        self._feature_summaries: deque[dict[str, Any]] = deque(maxlen=20)
        self._match_scores: dict[str, float] = {}
        self._last_evaluation: MatchEvaluation | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> PanoramaAgentState:
        return self._state

    @property
    def previous_state(self) -> PanoramaAgentState:
        return self._previous_state

    @property
    def steps_in_state(self) -> int:
        return self._steps_in_state

    @property
    def investigation_steps(self) -> int:
        return self._investigation_step_count

    @property
    def confidence_history(self) -> list[float]:
        return list(self._confidence_history)

    # ------------------------------------------------------------------
    # Core transition logic
    # ------------------------------------------------------------------

    def update(
        self,
        evaluation: MatchEvaluation,
        viewport_b64: str | None = None,
        feature_summary: dict[str, Any] | None = None,
    ) -> PanoramaAgentState:
        """Feed a new match evaluation and compute the next state.

        Parameters
        ----------
        evaluation : MatchEvaluation
            Full ranked match result from the location resolver.
        viewport_b64 : str, optional
            Base64 JPEG of the current viewport (for evidence bundle).
        feature_summary : dict, optional
            Feature summary for the current step.

        Returns
        -------
        PanoramaAgentState
            The new agent state after this update.
        """
        self._last_evaluation = evaluation
        self._investigation_step_count += 1

        # Accumulate evidence
        top_confidence = (
            evaluation.candidates[0].confidence
            if evaluation.candidates
            else 0.0
        )
        self._confidence_history.append(top_confidence)

        if viewport_b64:
            self._viewport_images.append(viewport_b64)
        if feature_summary:
            self._feature_summaries.append(feature_summary)

        # Track best score per candidate
        for c in evaluation.candidates:
            prev = self._match_scores.get(c.location_id, 0.0)
            self._match_scores[c.location_id] = max(prev, c.confidence)

        # Compute new state
        new_state = self._compute_next_state(evaluation, top_confidence)

        if new_state != self._state:
            old_state = self._state
            self._previous_state = self._state
            logger.debug(
                "Investigation state: %s → %s (step=%d, conf=%.3f)",
                self._state.value,
                new_state.value,
                self._investigation_step_count,
                top_confidence,
            )
            old_steps_in_state = self._steps_in_state
            self._steps_in_state = 0
            self._state = new_state

            # Emit state_transition event
            self._emit_state_transition(
                old_state, new_state, top_confidence, old_steps_in_state,
            )

            # Emit investigation_window events on open/close
            if (
                old_state == PanoramaAgentState.confident_match
                and new_state == PanoramaAgentState.investigating_unknown
            ):
                self._emit_investigation_window(opened=True)
            elif (
                new_state in (
                    PanoramaAgentState.confident_match,
                    PanoramaAgentState.label_request,
                )
                and old_state not in (
                    PanoramaAgentState.confident_match,
                    PanoramaAgentState.label_request,
                )
            ):
                self._emit_investigation_window(opened=False)
        else:
            self._steps_in_state += 1

        return self._state

    def _compute_next_state(
        self,
        evaluation: MatchEvaluation,
        top_confidence: float,
    ) -> PanoramaAgentState:
        """Determine the next state based on evidence."""

        # Immediate confident match — skip investigation
        if top_confidence >= self._confident_threshold:
            return PanoramaAgentState.confident_match

        # During active investigation
        n = self._investigation_step_count

        # Not enough steps yet — stay investigating
        if n < self._min_steps:
            if top_confidence > self._label_request_ceiling:
                return PanoramaAgentState.matching_known
            return PanoramaAgentState.investigating_unknown

        # Check for confidence plateau
        is_plateaued = self._is_confidence_plateaued()

        # Hard cap — force a decision
        force_decision = n >= self._max_steps

        if is_plateaued or force_decision:
            if top_confidence <= self._label_request_ceiling:
                # Low confidence plateau → novel location candidate
                if self._state == PanoramaAgentState.novel_location_candidate:
                    # Already a candidate — emit label request
                    return PanoramaAgentState.label_request
                return PanoramaAgentState.novel_location_candidate
            else:
                # Moderate confidence plateau → matching known
                if evaluation.top_margin > 0.15:
                    return PanoramaAgentState.confident_match
                return PanoramaAgentState.low_confidence_match

        # Still evolving
        if top_confidence > self._label_request_ceiling:
            return PanoramaAgentState.matching_known

        return PanoramaAgentState.investigating_unknown

    def _is_confidence_plateaued(self) -> bool:
        """Check if recent confidence values have stabilised."""
        if len(self._confidence_history) < self._plateau_window:
            return False

        recent = list(self._confidence_history)[-self._plateau_window:]
        mean = sum(recent) / len(recent)
        variance = sum((v - mean) ** 2 for v in recent) / len(recent)
        std = math.sqrt(variance)

        return std < self._plateau_threshold

    # ------------------------------------------------------------------
    # Label gating
    # ------------------------------------------------------------------

    def should_request_label(self) -> bool:
        """Return True if the agent should request a label now."""
        return self._state == PanoramaAgentState.label_request

    # ------------------------------------------------------------------
    # Evidence bundle
    # ------------------------------------------------------------------

    def get_evidence_bundle(self) -> EvidenceBundle:
        """Build the evidence bundle for a label request.

        Aggregates images, features, scores, and confidence history
        accumulated during the investigation window.
        """
        best_label: str | None = None
        best_conf = 0.0
        if self._last_evaluation and self._last_evaluation.candidates:
            best = self._last_evaluation.candidates[0]
            best_label = best.label
            best_conf = best.confidence

        # Compute final margin
        margin = 0.0
        if self._last_evaluation and len(self._last_evaluation.candidates) >= 2:
            margin = (
                self._last_evaluation.candidates[0].confidence
                - self._last_evaluation.candidates[1].confidence
            )

        return EvidenceBundle(
            viewport_images_b64=list(self._viewport_images),
            feature_summaries=list(self._feature_summaries),
            match_scores=dict(self._match_scores),
            confidence_history=list(self._confidence_history),
            investigation_steps=self._investigation_step_count,
            margin=margin,
            best_candidate_label=best_label,
            best_candidate_confidence=best_conf,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the investigation state after a label is confirmed."""
        self._previous_state = self._state
        self._state = PanoramaAgentState.investigating_unknown
        self._steps_in_state = 0
        self._investigation_step_count = 0
        self._confidence_history.clear()
        self._viewport_images.clear()
        self._feature_summaries.clear()
        self._match_scores.clear()
        self._last_evaluation = None
        logger.debug("Investigation state machine reset")

    def reset_to_confident(self, location_label: str) -> None:
        """Reset into confident_match state (e.g. after receiving a label)."""
        self._previous_state = self._state
        self._state = PanoramaAgentState.confident_match
        self._steps_in_state = 0
        self._investigation_step_count = 0
        self._confidence_history.clear()
        self._viewport_images.clear()
        self._feature_summaries.clear()
        self._match_scores.clear()
        self._last_evaluation = None
        logger.debug("Investigation reset to confident_match: %s", location_label)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_state_transition(
        self,
        old_state: PanoramaAgentState,
        new_state: PanoramaAgentState,
        confidence: float,
        steps_in_previous: int,
    ) -> None:
        """Emit a state_transition event through the event bus."""
        if not self._event_bus:
            return
        from datetime import datetime as dt

        payload = StateTransitionPayload(
            previous_state=old_state.value,
            new_state=new_state.value,
            reason=self._transition_reason(old_state, new_state, confidence),
            confidence=confidence,
            steps_in_previous=steps_in_previous,
        )
        from episodic_agent.schemas.panorama_events import PanoramaEvent

        event = PanoramaEvent(
            event_type=PanoramaEventType.state_transition,
            timestamp=dt.now(),
            step=self._investigation_step_count,
            state=new_state,
            payload=payload.model_dump(),
        )
        self._event_bus.emit(event)

    def _emit_investigation_window(self, opened: bool) -> None:
        """Emit an investigation_window event (opened or closed)."""
        if not self._event_bus:
            return
        from datetime import datetime as dt
        from episodic_agent.schemas.panorama_events import PanoramaEvent

        payload = {
            "action": "opened" if opened else "closed",
            "investigation_steps": self._investigation_step_count,
            "confidence_history_length": len(self._confidence_history),
        }
        if not opened:
            payload["evidence_summary"] = {
                "total_viewports": len(self._viewport_images),
                "match_score_count": len(self._match_scores),
                "final_state": self._state.value,
            }

        event = PanoramaEvent(
            event_type=PanoramaEventType.investigation_window,
            timestamp=dt.now(),
            step=self._investigation_step_count,
            state=self._state,
            payload=payload,
        )
        self._event_bus.emit(event)

    @staticmethod
    def _transition_reason(
        old: PanoramaAgentState,
        new: PanoramaAgentState,
        confidence: float,
    ) -> str:
        """Generate a human-readable reason for a state transition."""
        if new == PanoramaAgentState.confident_match:
            return f"High confidence ({confidence:.3f}) exceeds threshold"
        if new == PanoramaAgentState.label_request:
            return "Novel location confirmed after investigation plateau"
        if new == PanoramaAgentState.novel_location_candidate:
            return f"Low confidence plateau ({confidence:.3f}), candidate for labeling"
        if new == PanoramaAgentState.matching_known:
            return f"Moderate confidence ({confidence:.3f}), matching known locations"
        if new == PanoramaAgentState.low_confidence_match:
            return f"Confidence settled at {confidence:.3f} with narrow margin"
        if new == PanoramaAgentState.investigating_unknown:
            return "New investigation window opened"
        return f"{old.value} → {new.value}"