"""Confidence calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from episodic_agent.utils.config import CONFIDENCE_T_HIGH, CONFIDENCE_T_LOW


@dataclass
class ConfidenceSignal:
    """A single confidence signal with weight."""
    
    name: str
    value: float  # Should be in [0, 1]
    weight: float = 1.0


class ConfidenceHelper:
    """Helper for computing aggregate confidence from multiple signals.
    
    Supports:
    - Weighted combination of signals
    - Configurable thresholds for low/high confidence
    - Various aggregation strategies
    """

    def __init__(
        self,
        t_low: float = CONFIDENCE_T_LOW,
        t_high: float = CONFIDENCE_T_HIGH,
    ) -> None:
        """Initialize the confidence helper.
        
        Args:
            t_low: Low confidence threshold (below = "unknown").
            t_high: High confidence threshold (above = "confident").
        """
        self.t_low = t_low
        self.t_high = t_high

    def combine_weighted(self, signals: list[ConfidenceSignal]) -> float:
        """Combine signals using weighted average.
        
        Args:
            signals: List of confidence signals with weights.
            
        Returns:
            Combined confidence in [0, 1].
        """
        if not signals:
            return 0.0
        
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(s.value * s.weight for s in signals)
        return max(0.0, min(1.0, weighted_sum / total_weight))

    def combine_max(self, signals: list[ConfidenceSignal]) -> float:
        """Combine signals using maximum value.
        
        Args:
            signals: List of confidence signals.
            
        Returns:
            Maximum confidence value.
        """
        if not signals:
            return 0.0
        return max(s.value for s in signals)

    def combine_min(self, signals: list[ConfidenceSignal]) -> float:
        """Combine signals using minimum value (conservative).
        
        Args:
            signals: List of confidence signals.
            
        Returns:
            Minimum confidence value.
        """
        if not signals:
            return 0.0
        return min(s.value for s in signals)

    def combine_product(self, signals: list[ConfidenceSignal]) -> float:
        """Combine signals using product (for independent probabilities).
        
        Args:
            signals: List of confidence signals.
            
        Returns:
            Product of confidence values.
        """
        if not signals:
            return 0.0
        
        result = 1.0
        for s in signals:
            result *= s.value
        return result

    def is_low(self, confidence: float) -> bool:
        """Check if confidence is below low threshold.
        
        Args:
            confidence: Confidence value to check.
            
        Returns:
            True if below t_low.
        """
        return confidence < self.t_low

    def is_high(self, confidence: float) -> bool:
        """Check if confidence is above high threshold.
        
        Args:
            confidence: Confidence value to check.
            
        Returns:
            True if above t_high.
        """
        return confidence > self.t_high

    def is_medium(self, confidence: float) -> bool:
        """Check if confidence is in the medium range.
        
        Args:
            confidence: Confidence value to check.
            
        Returns:
            True if between t_low and t_high (inclusive).
        """
        return self.t_low <= confidence <= self.t_high

    def categorize(self, confidence: float) -> str:
        """Categorize confidence level.
        
        Args:
            confidence: Confidence value to categorize.
            
        Returns:
            "low", "medium", or "high".
        """
        if self.is_low(confidence):
            return "low"
        elif self.is_high(confidence):
            return "high"
        return "medium"

    def decay(
        self,
        confidence: float,
        factor: float = 0.95,
        floor: float = 0.0,
    ) -> float:
        """Apply decay to a confidence value.
        
        Args:
            confidence: Current confidence.
            factor: Decay multiplier (< 1 for decay).
            floor: Minimum value after decay.
            
        Returns:
            Decayed confidence.
        """
        return max(floor, confidence * factor)

    def boost(
        self,
        confidence: float,
        amount: float = 0.1,
        ceiling: float = 1.0,
    ) -> float:
        """Boost a confidence value.
        
        Args:
            confidence: Current confidence.
            amount: Amount to add.
            ceiling: Maximum value after boost.
            
        Returns:
            Boosted confidence.
        """
        return min(ceiling, confidence + amount)

    def smooth_transition(
        self,
        old_confidence: float,
        new_confidence: float,
        alpha: float = 0.3,
    ) -> float:
        """Smooth transition between old and new confidence.
        
        Uses exponential moving average style smoothing.
        
        Args:
            old_confidence: Previous confidence value.
            new_confidence: New observed confidence.
            alpha: Smoothing factor (0 = keep old, 1 = use new).
            
        Returns:
            Smoothed confidence.
        """
        return old_confidence * (1 - alpha) + new_confidence * alpha


# Global instance with default thresholds
default_confidence_helper = ConfidenceHelper()


def combine_confidence(
    signals: list[tuple[str, float, float]] | list[ConfidenceSignal],
    method: str = "weighted",
) -> float:
    """Convenience function to combine confidence signals.
    
    Args:
        signals: Either list of ConfidenceSignal or tuples of (name, value, weight).
        method: Aggregation method ("weighted", "max", "min", "product").
        
    Returns:
        Combined confidence in [0, 1].
    """
    # Convert tuples to ConfidenceSignal if needed
    if signals and isinstance(signals[0], tuple):
        signals = [
            ConfidenceSignal(name=s[0], value=s[1], weight=s[2] if len(s) > 2 else 1.0)
            for s in signals
        ]
    
    helper = default_confidence_helper
    
    if method == "weighted":
        return helper.combine_weighted(signals)
    elif method == "max":
        return helper.combine_max(signals)
    elif method == "min":
        return helper.combine_min(signals)
    elif method == "product":
        return helper.combine_product(signals)
    else:
        return helper.combine_weighted(signals)
