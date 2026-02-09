"""Real location resolver using scene fingerprinting.

Discovers location boundaries from statistical regularities in the
perception stream — no Unity GUIDs required. Works identically with
real-world sensors or simulated environments.

Algorithm:
1. Each frame produces a scene_embedding (from any PerceptionModule)
2. Cosine distance to the current location's centroid is tracked
3. When distance exceeds a threshold for N consecutive frames
   (hysteresis), a location transition is declared
4. The new scene fingerprint is matched against known locations
   or a new location is created
5. User is prompted for labels via dialog_manager

ARCHITECTURAL INVARIANT: No pre-wired semantics. The resolver never
reads Unity GUIDs, room names, or any ground-truth labels. It only
sees perception embeddings and agent-relative positions.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from episodic_agent.core.interfaces import LocationResolver
from episodic_agent.schemas import (
    ActiveContextFrame,
    EdgeType,
    GraphEdge,
    GraphNode,
    LocationFingerprint,
    NodeType,
    Percept,
)
from episodic_agent.schemas.panorama_events import (
    MatchCandidate,
    MatchEvaluation,
    MemoryWritePayload,
    PanoramaAgentState,
    PanoramaEvent,
    PanoramaEventType,
    StateTransitionPayload,
)
from episodic_agent.utils.confidence import ConfidenceHelper, ConfidenceSignal
from episodic_agent.utils.config import (
    CONFIDENCE_T_HIGH,
    CONFIDENCE_T_LOW,
    DEFAULT_EMBEDDING_DIM,
)

if TYPE_CHECKING:
    from episodic_agent.memory.graph_store import LabeledGraphStore
    from episodic_agent.modules.dialog import DialogManager
    from episodic_agent.modules.panorama.event_bus import PanoramaEventBus
    from episodic_agent.modules.panorama.investigation import InvestigationStateMachine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: cosine distance
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Returns value in [-1, 1]; 1 = identical directions, 0 = orthogonal.
    """
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return dot / (norm_a * norm_b)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance (0 = identical, 2 = opposite)."""
    return 1.0 - _cosine_similarity(a, b)


def _running_average(
    centroid: list[float],
    new_vec: list[float],
    count: int,
) -> list[float]:
    """Compute an incremental running average of embedding vectors.

    centroid = ((centroid * count) + new_vec) / (count + 1)
    """
    if not centroid:
        return list(new_vec)
    n = count + 1
    return [(c * count + v) / n for c, v in zip(centroid, new_vec)]


# ---------------------------------------------------------------------------
# LocationResolverReal
# ---------------------------------------------------------------------------

class LocationResolverReal(LocationResolver):
    """Location resolver using perception-based scene fingerprinting.

    Instead of relying on Unity room GUIDs (like LocationResolverCheat)
    this resolver discovers location boundaries from the scene embedding
    stream that any PerceptionModule produces.

    Features:
    - Scene fingerprinting via centroid embeddings
    - Transition detection with cosine distance + hysteresis
    - Location matching against known fingerprints
    - User label prompting via dialog_manager
    - Graph persistence of discovered locations
    - Tracks transition positions for boundary estimation

    Constructor parameters:
        graph_store:        Graph store for persistence.
        dialog_manager:     For label prompting.
        transition_threshold:   Cosine distance to trigger a candidate
                                transition (default 0.40).
        hysteresis_frames:  Consecutive frames above threshold before
                            accepting a transition (default 5).
        match_threshold:    Max cosine distance to consider a fingerprint
                            match to a known location (default 0.35).
        embedding_dim:      Dimensionality of scene embeddings.
        auto_label:         If True, auto-generates labels without prompt.
    """

    def __init__(
        self,
        graph_store: "LabeledGraphStore",
        dialog_manager: "DialogManager",
        transition_threshold: float = 0.40,
        hysteresis_frames: int = 5,
        match_threshold: float = 0.35,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        auto_label: bool = False,
        event_bus: "PanoramaEventBus | None" = None,
        investigation_sm: "InvestigationStateMachine | None" = None,
    ) -> None:
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        self._transition_threshold = transition_threshold
        self._hysteresis_frames = hysteresis_frames
        self._match_threshold = match_threshold
        self._embedding_dim = embedding_dim
        self._auto_label = auto_label

        # Observability hooks (optional — backward-compatible)
        self._event_bus: PanoramaEventBus | None = event_bus
        self._investigation_sm: InvestigationStateMachine | None = investigation_sm
        self._step_counter: int = 0

        # Per-fingerprint match history: location_id → [(step, confidence)]
        self._match_history: dict[str, list[tuple[int, float]]] = {}
        # Per-fingerprint running variance tracking
        self._embedding_sum_sq: dict[str, list[float]] = {}
        # Step tracking per location
        self._first_seen_step: dict[str, int] = {}
        self._last_seen_step: dict[str, int] = {}
        # Aggregated feature summaries per location
        self._aggregated_features: dict[str, dict[str, Any]] = {}
        # Label callback (set from API server for dashboard labelling)
        self._label_callback: Any = None

        self._confidence_helper = ConfidenceHelper()

        # ── internal state ──────────────────────────────────────────────
        self._current_location_id: str | None = None
        self._fingerprints: dict[str, LocationFingerprint] = {}

        # Hysteresis counters
        self._transition_counter: int = 0
        self._candidate_embedding_buffer: list[list[float]] = []

        # Pending label request tracking
        self._pending_label_request: str | None = None

        # Load any persisted fingerprints from the graph store
        self._load_persisted_locations()

    # ------------------------------------------------------------------
    # LocationResolver interface
    # ------------------------------------------------------------------

    def resolve(
        self,
        percept: Percept,
        acf: ActiveContextFrame,
    ) -> tuple[str, float]:
        """Resolve current location from the scene embedding.

        Args:
            percept: Current perception (must have scene_embedding).
            acf:     Active context frame.

        Returns:
            (location_label, confidence)
        """
        self._step_counter += 1

        embedding = percept.scene_embedding
        if not embedding:
            # No embedding available — return current or unknown
            if self._current_location_id:
                fp = self._fingerprints.get(self._current_location_id)
                node = self._find_node_for_location(self._current_location_id)
                label = node.label if node else "unknown"
                conf = self._compute_confidence(fp, 0.0) if fp else 0.0
                return (label, conf)
            return ("unknown", 0.0)

        # Extract agent position from extras for boundary tracking
        agent_pos = self._extract_agent_position(percept)

        # ── Panorama-aware: only evaluate transitions at image boundaries ──
        # Intermediate viewports of the same panorama can differ greatly;
        # we only check for location changes on the last heading when the
        # scene_embedding is the full panoramic average.
        extras = percept.extras or {}
        is_last_heading = extras.get("is_last_heading", True)  # default True for non-panorama

        # ── first-ever frame: bootstrap a location ──────────────────
        if self._current_location_id is None:
            if not is_last_heading:
                # Wait until we have the panoramic embedding to bootstrap
                return ("unknown", 0.0)
            return self._bootstrap_first_location(embedding, agent_pos)

        # ── check distance to current centroid ──────────────────────
        current_fp = self._fingerprints[self._current_location_id]
        distance = _cosine_distance(embedding, current_fp.centroid_embedding)

        if distance < self._transition_threshold:
            # Still in the same location — update centroid
            self._transition_counter = 0
            self._candidate_embedding_buffer.clear()
            current_fp.centroid_embedding = _running_average(
                current_fp.centroid_embedding,
                embedding,
                current_fp.observation_count,
            )
            current_fp.observation_count += 1
            current_fp.last_visited = datetime.now()

            # Track embedding variance
            self._update_embedding_variance(self._current_location_id, embedding)

            # Update entity co-occurrence
            self._update_entity_cooccurrence(current_fp, percept)

            node = self._find_node_for_location(self._current_location_id)
            label = node.label if node else "unknown"
            confidence = self._compute_confidence(current_fp, distance)

            # Emit match evaluation event
            self._emit_match_evaluation(embedding, label, confidence, distance)

            return (label, confidence)

        # ── candidate transition ────────────────────────────────────
        # Only count toward transition on image boundaries (last heading).
        # Mid-sweep viewports of the same panorama may legitimately differ
        # from the centroid without implying a location change.
        if not is_last_heading:
            node = self._find_node_for_location(self._current_location_id)
            label = node.label if node else "unknown"
            confidence = self._compute_confidence(current_fp, distance)
            # Still emit match evaluation for dashboard visibility
            self._emit_match_evaluation(embedding, label, confidence, distance)
            return (label, confidence)

        self._transition_counter += 1
        self._candidate_embedding_buffer.append(embedding)

        if self._transition_counter < self._hysteresis_frames:
            # Not yet enough consecutive frames — stay put
            node = self._find_node_for_location(self._current_location_id)
            label = node.label if node else "unknown"
            # Return lower confidence to signal instability
            confidence = max(0.0, self._compute_confidence(current_fp, distance) - 0.2)
            return (label, confidence)

        # ── transition confirmed ────────────────────────────────────
        logger.info(
            "Location transition detected (distance=%.3f, frames=%d)",
            distance,
            self._transition_counter,
        )

        # Record transition position on the old location
        if agent_pos:
            current_fp.transition_positions.append(agent_pos)

        # Average the buffer into a candidate centroid
        candidate_centroid = self._average_buffer()
        self._transition_counter = 0
        self._candidate_embedding_buffer.clear()

        # Try to match candidate against known locations
        match_id, match_dist = self._find_best_match(candidate_centroid)

        if match_id and match_id != self._current_location_id:
            # Revisiting a known location
            return self._handle_revisit(match_id, candidate_centroid, agent_pos)
        else:
            # New location discovered
            return self._handle_new_location(candidate_centroid, agent_pos, percept)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_location_fingerprint(self, location_id: str) -> LocationFingerprint | None:
        """Get the fingerprint for a discovered location."""
        return self._fingerprints.get(location_id)

    def get_all_fingerprints(self) -> dict[str, LocationFingerprint]:
        """Get all discovered location fingerprints."""
        return dict(self._fingerprints)

    def get_location_node(self, location_id: str) -> GraphNode | None:
        """Get the graph node for a location ID (public accessor)."""
        return self._find_node_for_location(location_id)

    def get_all_match_scores(
        self, embedding: list[float]
    ) -> list[MatchCandidate]:
        """Compute match scores against all known fingerprints.

        Returns a ranked list (best match first) of MatchCandidate
        objects with confidence and distance for every known location.
        """
        candidates: list[MatchCandidate] = []
        for lid, fp in self._fingerprints.items():
            if not fp.centroid_embedding:
                continue
            dist = _cosine_distance(embedding, fp.centroid_embedding)
            conf = self._compute_confidence(fp, dist)
            node = self._find_node_for_location(lid)
            label = node.label if node else f"location_{lid[:8]}"
            candidates.append(
                MatchCandidate(
                    location_id=lid,
                    label=label,
                    confidence=conf,
                    distance=dist,
                )
            )
        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def get_match_history(self, location_id: str) -> list[tuple[int, float]]:
        """Return the match confidence history for a location."""
        return list(self._match_history.get(location_id, []))

    def get_embedding_variance(self, location_id: str) -> float:
        """Return the running embedding variance for a location."""
        fp = self._fingerprints.get(location_id)
        if not fp or fp.observation_count < 2:
            return 0.0
        sum_sq = self._embedding_sum_sq.get(location_id)
        if not sum_sq:
            return 0.0
        n = fp.observation_count
        centroid = fp.centroid_embedding
        # Var = E[X^2] - (E[X])^2, averaged across dimensions
        variance_per_dim = [
            (sq / n) - (c ** 2)
            for sq, c in zip(sum_sq, centroid)
        ]
        return sum(max(0.0, v) for v in variance_per_dim) / len(variance_per_dim)

    def get_first_seen_step(self, location_id: str) -> int:
        """Return the step number when a location was first discovered."""
        return self._first_seen_step.get(location_id, 0)

    def get_last_seen_step(self, location_id: str) -> int:
        """Return the step number when a location was last observed."""
        return self._last_seen_step.get(location_id, 0)

    def get_aggregated_features(self, location_id: str) -> dict[str, Any]:
        """Return the aggregated feature summary for a location."""
        return dict(self._aggregated_features.get(location_id, {}))

    def update_aggregated_features(
        self, location_id: str, feature_summary: dict[str, Any]
    ) -> None:
        """Update the running aggregated feature summary for a location.

        Merges numeric fields using running average, keeps latest for others.
        """
        if not feature_summary:
            return
        existing = self._aggregated_features.get(location_id, {})
        fp = self._fingerprints.get(location_id)
        n = fp.observation_count if fp else 1

        merged: dict[str, Any] = dict(existing)
        for key, value in feature_summary.items():
            if isinstance(value, (int, float)) and key in existing:
                prev = existing[key]
                if isinstance(prev, (int, float)) and n > 1:
                    # Running average
                    merged[key] = prev + (value - prev) / n
                else:
                    merged[key] = value
            else:
                merged[key] = value
        self._aggregated_features[location_id] = merged

    def apply_dashboard_label(self, label: str) -> None:
        """Apply a label from the dashboard (via API server POST /api/label).

        Updates the current location's graph node and resets the
        investigation state machine if present.
        """
        if not self._current_location_id:
            logger.warning("apply_dashboard_label: no current location")
            return

        node = self._find_node_for_location(self._current_location_id)
        if node:
            old_label = node.label
            if label != old_label:
                # Check for label collision — auto-merge if another location
                # already has this name (dashboard relabeling scenario).
                existing_node, existing_lid = self._find_node_by_label(label)
                if (
                    existing_node
                    and existing_lid
                    and existing_lid != self._current_location_id
                ):
                    current_fp = self._fingerprints.get(self._current_location_id)
                    if current_fp:
                        self._merge_fingerprints(
                            existing_lid,
                            self._current_location_id,
                            current_fp.centroid_embedding,
                            current_fp.approximate_center,
                        )
                        self._current_location_id = existing_lid
                        logger.info(
                            "Dashboard label merge: %s into %s (%s)",
                            old_label, label, existing_lid,
                        )
                        if self._investigation_sm:
                            self._investigation_sm.reset_to_confident(label)
                        return

                node.labels.append(old_label)
            node.label = label
            logger.info("Dashboard label applied: %s → %s", old_label, label)
        else:
            logger.warning("apply_dashboard_label: no node for %s",
                           self._current_location_id)

        if self._investigation_sm:
            self._investigation_sm.reset_to_confident(label)

    # ------------------------------------------------------------------
    # Bootstrap / first frame
    # ------------------------------------------------------------------

    def _bootstrap_first_location(
        self,
        embedding: list[float],
        agent_pos: tuple[float, float, float] | None,
    ) -> tuple[str, float]:
        """Create the very first location from the first frame."""
        location_id = f"loc_fp_{uuid.uuid4().hex[:12]}"

        fp = LocationFingerprint(
            location_id=location_id,
            centroid_embedding=list(embedding),
            observation_count=1,
            approximate_center=agent_pos,
            first_visited=datetime.now(),
            last_visited=datetime.now(),
        )
        self._fingerprints[location_id] = fp
        self._current_location_id = location_id
        self._first_seen_step[location_id] = self._step_counter
        self._last_seen_step[location_id] = self._step_counter

        # Create graph node
        node = self._create_location_node(location_id, embedding, agent_pos)
        label = node.label

        self._dialog_manager.notify(f"📍 First location established: {label}")
        return (label, CONFIDENCE_T_LOW)

    # ------------------------------------------------------------------
    # Revisit a known location
    # ------------------------------------------------------------------

    def _handle_revisit(
        self,
        location_id: str,
        candidate_centroid: list[float],
        agent_pos: tuple[float, float, float] | None,
    ) -> tuple[str, float]:
        """Handle returning to a previously seen location."""
        fp = self._fingerprints[location_id]
        fp.centroid_embedding = _running_average(
            fp.centroid_embedding,
            candidate_centroid,
            fp.observation_count,
        )
        fp.observation_count += 1
        fp.last_visited = datetime.now()
        if agent_pos:
            fp.transition_positions.append(agent_pos)

        self._current_location_id = location_id
        self._last_seen_step[location_id] = self._step_counter
        if location_id not in self._first_seen_step:
            self._first_seen_step[location_id] = self._step_counter

        node = self._find_node_for_location(location_id)
        if node:
            node.last_accessed = datetime.now()
            node.access_count += 1
            label = node.label
        else:
            label = f"location_{location_id[:8]}"

        distance = _cosine_distance(candidate_centroid, fp.centroid_embedding)
        confidence = self._compute_confidence(fp, distance)

        self._dialog_manager.notify(f"📍 Returned to: {label}")
        logger.info("Revisiting location: %s (visits=%d)", label, fp.observation_count)

        # Emit memory_write event for revisit
        self._emit_memory_write(location_id, label, is_new=False, observation_count=fp.observation_count)

        return (label, confidence)

    # ------------------------------------------------------------------
    # New location discovered
    # ------------------------------------------------------------------

    def _handle_new_location(
        self,
        candidate_centroid: list[float],
        agent_pos: tuple[float, float, float] | None,
        percept: Percept,
    ) -> tuple[str, float]:
        """Handle discovery of a new location."""
        location_id = f"loc_fp_{uuid.uuid4().hex[:12]}"

        fp = LocationFingerprint(
            location_id=location_id,
            centroid_embedding=list(candidate_centroid),
            observation_count=1,
            approximate_center=agent_pos,
            first_visited=datetime.now(),
            last_visited=datetime.now(),
        )

        # Capture entities currently visible
        for candidate in percept.candidates:
            guid = candidate.extras.get("guid", candidate.candidate_id) if candidate.extras else candidate.candidate_id
            if guid not in fp.entity_guids_seen:
                fp.entity_guids_seen.append(guid)
            fp.entity_cooccurrence_counts[guid] = fp.entity_cooccurrence_counts.get(guid, 0) + 1

        self._fingerprints[location_id] = fp
        self._current_location_id = location_id
        self._first_seen_step[location_id] = self._step_counter
        self._last_seen_step[location_id] = self._step_counter

        node = self._create_location_node(location_id, candidate_centroid, agent_pos)
        label = node.label

        self._dialog_manager.notify(f"🆕 New location discovered: {label}")
        logger.info("New location: %s", label)

        # Emit memory_write event
        self._emit_memory_write(location_id, label, is_new=True)

        return (label, CONFIDENCE_T_LOW)

    # ------------------------------------------------------------------
    # Graph persistence helpers
    # ------------------------------------------------------------------

    def _create_location_node(
        self,
        location_id: str,
        embedding: list[float],
        agent_pos: tuple[float, float, float] | None,
    ) -> GraphNode:
        """Create a new graph node for a discovered location."""
        if self._auto_label:
            label = f"location_{location_id[-8:]}"
        else:
            suggestions = [f"location_{location_id[-8:]}"]
            self._dialog_manager.notify("🆕 New location detected!")
            label = self._dialog_manager.ask_label(
                "What should this location be called?",
                suggestions=suggestions,
            )

        # Check if a location with this label already exists —
        # offer to merge fingerprints if so (handles duplicate names).
        existing_node, existing_lid = self._find_node_by_label(label)
        if existing_node and existing_lid and existing_lid != location_id:
            self._dialog_manager.notify(
                f"⚠️  A location named '{label}' already exists "
                f"({self._fingerprints[existing_lid].observation_count} visits)."
            )
            merge = self._dialog_manager.ask_label(
                f"Merge this with existing '{label}'? (yes/no)",
                suggestions=["yes", "no"],
            )
            if merge.lower().strip() in ("yes", "y"):
                self._merge_fingerprints(existing_lid, location_id, embedding, agent_pos)
                self._current_location_id = existing_lid
                # Reset investigation SM
                if self._investigation_sm:
                    self._investigation_sm.reset_to_confident(label)
                return existing_node

        # Reset investigation SM now that we have a label —
        # no need to keep investigating this location.
        if self._investigation_sm:
            self._investigation_sm.reset_to_confident(label)

        node = GraphNode(
            node_id=location_id,
            node_type=NodeType.LOCATION,
            label=label,
            labels=[],
            embedding=embedding[:self._embedding_dim] if len(embedding) > self._embedding_dim else embedding,
            source_id=location_id,
            confidence=CONFIDENCE_T_LOW,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            extras={
                "source": "fingerprint",
                "agent_position": list(agent_pos) if agent_pos else None,
            },
        )

        self._graph_store.add_node(node)
        return node

    def _find_node_for_location(self, location_id: str) -> GraphNode | None:
        """Find the graph node associated with a location ID."""
        location_nodes = self._graph_store.get_nodes_by_type(NodeType.LOCATION)
        for node in location_nodes:
            if node.source_id == location_id or node.node_id == location_id:
                return node
        return None

    def _find_node_by_label(self, label: str) -> tuple[GraphNode | None, str | None]:
        """Find a location node by its label.

        Returns (node, location_id) or (None, None) if not found.
        """
        location_nodes = self._graph_store.get_nodes_by_type(NodeType.LOCATION)
        for node in location_nodes:
            if node.label == label:
                lid = node.source_id or node.node_id
                return (node, lid)
        return (None, None)

    def _merge_fingerprints(
        self,
        keep_id: str,
        merge_id: str,
        merge_embedding: list[float],
        merge_pos: tuple[float, float, float] | None,
    ) -> None:
        """Merge a new fingerprint into an existing one.

        Combines centroids via weighted average and accumulates
        observation counts, entity data, and transition positions.
        """
        keep_fp = self._fingerprints.get(keep_id)
        if not keep_fp:
            logger.warning("merge_fingerprints: keep_id %s not found", keep_id)
            return

        # Weighted average of centroids
        total = keep_fp.observation_count + 1
        keep_fp.centroid_embedding = _running_average(
            keep_fp.centroid_embedding,
            merge_embedding,
            keep_fp.observation_count,
        )
        keep_fp.observation_count = total
        keep_fp.last_visited = datetime.now()

        if merge_pos:
            keep_fp.transition_positions.append(merge_pos)

        # If the merge_id had a fingerprint, fold its data too
        merge_fp = self._fingerprints.pop(merge_id, None)
        if merge_fp:
            # Fold observation count
            keep_fp.observation_count += merge_fp.observation_count - 1  # -1 since we already added 1
            # Merge entity data
            for guid in merge_fp.entity_guids_seen:
                if guid not in keep_fp.entity_guids_seen:
                    keep_fp.entity_guids_seen.append(guid)
            for guid, count in merge_fp.entity_cooccurrence_counts.items():
                keep_fp.entity_cooccurrence_counts[guid] = (
                    keep_fp.entity_cooccurrence_counts.get(guid, 0) + count
                )
            # Merge transition positions
            keep_fp.transition_positions.extend(merge_fp.transition_positions)

        # Update graph node
        keep_node = self._find_node_for_location(keep_id)
        if keep_node:
            keep_node.access_count = keep_fp.observation_count
            keep_node.last_accessed = datetime.now()

        # Clean up merge_id node from graph
        merge_node = self._find_node_for_location(merge_id)
        if merge_node:
            # Add as alias
            if keep_node and merge_id not in keep_node.labels:
                keep_node.labels.append(merge_id)

        self._dialog_manager.notify(
            f"✅ Merged into existing location "
            f"({keep_fp.observation_count} total visits)"
        )
        logger.info(
            "Merged fingerprint %s into %s (total visits: %d)",
            merge_id, keep_id, keep_fp.observation_count,
        )

    def _load_persisted_locations(self) -> None:
        """Load previously persisted location nodes into fingerprints.

        On startup, reconstruct fingerprint state from graph nodes that
        were created by a prior run (source == 'fingerprint').
        """
        try:
            location_nodes = self._graph_store.get_nodes_by_type(NodeType.LOCATION)
        except Exception:
            return

        for node in location_nodes:
            if node.extras.get("source") != "fingerprint":
                continue
            lid = node.source_id or node.node_id
            if lid in self._fingerprints:
                continue

            fp = LocationFingerprint(
                location_id=lid,
                centroid_embedding=node.embedding or [],
                observation_count=node.access_count,
                first_visited=node.created_at,
                last_visited=node.last_accessed,
            )
            self._fingerprints[lid] = fp
            logger.debug("Loaded persisted fingerprint: %s (%s)", node.label, lid)

    # ------------------------------------------------------------------
    # Matching & similarity
    # ------------------------------------------------------------------

    def _find_best_match(
        self,
        candidate_centroid: list[float],
    ) -> tuple[str | None, float]:
        """Find the best matching known location for a candidate centroid.

        Returns (location_id, cosine_distance) or (None, inf) if no match.
        """
        best_id: str | None = None
        best_dist = float("inf")

        for lid, fp in self._fingerprints.items():
            if lid == self._current_location_id:
                continue  # skip the location we're leaving
            if not fp.centroid_embedding:
                continue
            dist = _cosine_distance(candidate_centroid, fp.centroid_embedding)
            if dist < best_dist:
                best_dist = dist
                best_id = lid

        if best_dist <= self._match_threshold:
            return (best_id, best_dist)
        return (None, best_dist)

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        fp: LocationFingerprint,
        distance: float,
    ) -> float:
        """Compute confidence for current location assignment."""
        signals = [
            ConfidenceSignal(
                "embedding_closeness",
                max(0.0, 1.0 - distance),
                weight=2.0,
            ),
            ConfidenceSignal(
                "visit_count",
                min(1.0, fp.observation_count / 20.0),
                weight=1.0,
            ),
        ]
        return self._confidence_helper.combine_weighted(signals)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _average_buffer(self) -> list[float]:
        """Average the hysteresis embedding buffer into a single centroid."""
        if not self._candidate_embedding_buffer:
            return []
        dim = len(self._candidate_embedding_buffer[0])
        avg = [0.0] * dim
        for vec in self._candidate_embedding_buffer:
            for i, v in enumerate(vec):
                avg[i] += v
        n = len(self._candidate_embedding_buffer)
        return [x / n for x in avg]

    @staticmethod
    def _extract_agent_position(
        percept: Percept,
    ) -> tuple[float, float, float] | None:
        """Extract agent position from percept extras (if available)."""
        extras = percept.extras or {}
        camera_pose = extras.get("camera_pose")
        if camera_pose and isinstance(camera_pose, dict):
            pos = camera_pose.get("position", {})
            x = pos.get("x")
            y = pos.get("y")
            z = pos.get("z")
            if x is not None and y is not None and z is not None:
                return (float(x), float(y), float(z))
        return None

    def _update_entity_cooccurrence(
        self,
        fp: LocationFingerprint,
        percept: Percept,
    ) -> None:
        """Update entity co-occurrence counts for the current location."""
        for candidate in percept.candidates:
            guid = (
                candidate.extras.get("guid", candidate.candidate_id)
                if candidate.extras
                else candidate.candidate_id
            )
            if guid not in fp.entity_guids_seen:
                fp.entity_guids_seen.append(guid)
            fp.entity_cooccurrence_counts[guid] = (
                fp.entity_cooccurrence_counts.get(guid, 0) + 1
            )

    # ------------------------------------------------------------------
    # Event emission helpers
    # ------------------------------------------------------------------

    def _emit_match_evaluation(
        self,
        embedding: list[float],
        current_label: str,
        current_confidence: float,
        current_distance: float,
    ) -> None:
        """Emit a match_evaluation event with full ranked candidates."""
        if not self._event_bus:
            return

        candidates = self.get_all_match_scores(embedding)

        # Compute margin between top-2
        margin = 0.0
        if len(candidates) >= 2:
            margin = candidates[0].confidence - candidates[1].confidence

        evaluation = MatchEvaluation(
            candidates=candidates,
            top_margin=margin,
            hysteresis_active=self._transition_counter > 0,
            stabilization_frames=self._transition_counter,
            current_location_id=self._current_location_id,
            current_distance=current_distance,
        )

        # Record match history for each candidate
        for c in candidates:
            hist = self._match_history.setdefault(c.location_id, [])
            hist.append((self._step_counter, c.confidence))
            # Keep bounded
            if len(hist) > 100:
                self._match_history[c.location_id] = hist[-100:]

        # Feed investigation state machine if present
        sm_state = PanoramaAgentState.investigating_unknown
        if self._investigation_sm:
            sm_state = self._investigation_sm.state

        event = PanoramaEvent(
            event_type=PanoramaEventType.match_evaluation,
            timestamp=datetime.now(),
            step=self._step_counter,
            state=sm_state,
            payload=evaluation.model_dump(),
        )
        self._event_bus.emit(event)

    def _emit_memory_write(
        self,
        location_id: str,
        label: str,
        is_new: bool,
        observation_count: int = 1,
    ) -> None:
        """Emit a memory_write event when a fingerprint is created or updated."""
        if not self._event_bus:
            return

        fp = self._fingerprints.get(location_id)
        emb_norm = 0.0
        if fp and fp.centroid_embedding:
            emb_norm = math.sqrt(sum(v * v for v in fp.centroid_embedding))

        payload = MemoryWritePayload(
            location_id=location_id,
            label=label,
            is_new=is_new,
            observation_count=observation_count,
            embedding_norm=emb_norm,
        )

        sm_state = PanoramaAgentState.investigating_unknown
        if self._investigation_sm:
            sm_state = self._investigation_sm.state

        event = PanoramaEvent(
            event_type=PanoramaEventType.memory_write,
            timestamp=datetime.now(),
            step=self._step_counter,
            state=sm_state,
            payload=payload.model_dump(),
        )
        self._event_bus.emit(event)

    def _update_embedding_variance(
        self,
        location_id: str,
        embedding: list[float],
    ) -> None:
        """Update running sum-of-squares for embedding variance tracking."""
        if location_id not in self._embedding_sum_sq:
            self._embedding_sum_sq[location_id] = [v * v for v in embedding]
        else:
            sq = self._embedding_sum_sq[location_id]
            for i, v in enumerate(embedding):
                if i < len(sq):
                    sq[i] += v * v
                else:
                    sq.append(v * v)
