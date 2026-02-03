"""Stub perception module for Phase 1 testing."""

from __future__ import annotations

import random
import uuid

from episodic_agent.core.interfaces import PerceptionModule
from episodic_agent.schemas import ObjectCandidate, Percept, SensorFrame
from episodic_agent.utils.config import DEFAULT_EMBEDDING_DIM


class StubPerception(PerceptionModule):
    """Stub perception that generates deterministic embeddings.
    
    Produces percepts with fixed-length embeddings generated from
    a seeded RNG for reproducibility.
    """

    def __init__(
        self,
        seed: int = 42,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        """Initialize the stub perception module.
        
        Args:
            seed: Random seed for deterministic embeddings.
            embedding_dim: Dimensionality of generated embeddings.
        """
        self._seed = seed
        self._embedding_dim = embedding_dim
        self._rng = random.Random(seed)

    def process(self, frame: SensorFrame) -> Percept:
        """Process a sensor frame into a percept.
        
        Generates deterministic embeddings based on frame_id and seed.
        
        Args:
            frame: Raw sensor frame to process.
            
        Returns:
            Percept with synthetic scene embedding and no candidates.
        """
        # Reset RNG state based on frame_id for determinism
        self._rng.seed(self._seed + frame.frame_id)
        
        # Generate scene embedding
        scene_embedding = [
            self._rng.gauss(0.0, 1.0) for _ in range(self._embedding_dim)
        ]
        
        # Normalize embedding
        norm = sum(x * x for x in scene_embedding) ** 0.5
        if norm > 0:
            scene_embedding = [x / norm for x in scene_embedding]
        
        # Generate a small number of synthetic candidates
        num_candidates = self._rng.randint(0, 3)
        candidates = []
        
        for i in range(num_candidates):
            candidate_embedding = [
                self._rng.gauss(0.0, 1.0) for _ in range(self._embedding_dim)
            ]
            norm = sum(x * x for x in candidate_embedding) ** 0.5
            if norm > 0:
                candidate_embedding = [x / norm for x in candidate_embedding]
            
            candidates.append(
                ObjectCandidate(
                    candidate_id=f"cand_{uuid.uuid4().hex[:8]}",
                    label="unknown",
                    labels=[],
                    confidence=self._rng.uniform(0.3, 0.7),
                    embedding=candidate_embedding,
                    position=None,
                    bounding_box=None,
                )
            )
        
        return Percept(
            percept_id=f"perc_{uuid.uuid4().hex[:8]}",
            source_frame_id=frame.frame_id,
            scene_embedding=scene_embedding,
            candidates=candidates,
            confidence=self._rng.uniform(0.5, 0.9),
        )
