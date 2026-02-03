"""Stub resolvers and related modules for Phase 1 testing."""

from __future__ import annotations

import uuid
from datetime import datetime

from episodic_agent.core.interfaces import (
    ACFBuilder,
    EntityResolver,
    EventResolver,
    LocationResolver,
    Retriever,
)
from episodic_agent.schemas import (
    ActiveContextFrame,
    ObjectCandidate,
    Percept,
    RetrievalResult,
)


class StubACFBuilder(ACFBuilder):
    """Stub ACF builder that creates minimal context frames."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the stub ACF builder.
        
        Args:
            seed: Random seed (unused in stub, but kept for interface consistency).
        """
        self._seed = seed

    def create_acf(self) -> ActiveContextFrame:
        """Create a new empty ACF.
        
        Returns:
            A fresh ActiveContextFrame with default values.
        """
        return ActiveContextFrame(
            acf_id=f"acf_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            step_count=0,
            location_label="unknown",
            location_confidence=0.0,
            location_embedding=None,
            entities=[],
            events=[],
            current_percept=None,
        )

    def update_acf(
        self,
        acf: ActiveContextFrame,
        percept: Percept,
    ) -> ActiveContextFrame:
        """Update ACF with new perception data.
        
        Args:
            acf: Current ACF to update.
            percept: New percept to incorporate.
            
        Returns:
            Updated ACF (same instance, mutated).
        """
        acf.current_percept = percept
        return acf


class StubLocationResolver(LocationResolver):
    """Stub location resolver that always returns 'unknown'."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the stub location resolver.
        
        Args:
            seed: Random seed (unused in stub).
        """
        self._seed = seed

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> tuple[str, float]:
        """Resolve current location - always returns unknown.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            Tuple of ("unknown", 0.0).
        """
        return ("unknown", 0.0)


class StubEntityResolver(EntityResolver):
    """Stub entity resolver that passes through perception candidates."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the stub entity resolver.
        
        Args:
            seed: Random seed (unused in stub).
        """
        self._seed = seed

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[ObjectCandidate]:
        """Resolve entities - passes through percept candidates.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            List of candidates from the percept.
        """
        return list(percept.candidates)


class StubEventResolver(EventResolver):
    """Stub event resolver that detects no events."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the stub event resolver.
        
        Args:
            seed: Random seed (unused in stub).
        """
        self._seed = seed

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> list[dict]:
        """Detect events - always returns empty list.
        
        Args:
            percept: Current perception data.
            acf: Current active context frame.
            
        Returns:
            Empty list (no events detected).
        """
        return []


class StubRetriever(Retriever):
    """Stub retriever that returns empty results."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the stub retriever.
        
        Args:
            seed: Random seed (unused in stub).
        """
        self._seed = seed

    def retrieve(
        self,
        acf: ActiveContextFrame,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve relevant memories - returns empty results.
        
        Args:
            acf: Current active context frame as query.
            top_k: Maximum number of results to return.
            
        Returns:
            Empty RetrievalResult.
        """
        return RetrievalResult(
            query_id=f"q_{uuid.uuid4().hex[:8]}",
            episodes=[],
            episode_scores=[],
            nodes=[],
            node_scores=[],
            retrieval_method="stub",
            retrieval_time_ms=0.0,
        )
