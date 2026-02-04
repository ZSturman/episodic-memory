"""Label management with conflict detection and resolution."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from episodic_agent.schemas import (
    ConflictResolutionType,
    EdgeType,
    GraphEdge,
    LabelAssignment,
    LabelConflict,
)

if TYPE_CHECKING:
    from episodic_agent.core.interfaces import DialogManager
    from episodic_agent.memory.graph_store import LabeledGraphStore
    from episodic_agent.schemas import GraphNode


class LabelManager:
    """Manages label assignment and conflict resolution for graph nodes.
    
    Handles:
    - Assigning labels to nodes (primary and alternative)
    - Detecting label conflicts (same label on different nodes)
    - Routing conflicts through dialog manager for resolution
    - Creating merge/alias edges to preserve history
    """

    def __init__(
        self,
        graph_store: LabeledGraphStore,
        dialog_manager: DialogManager,
    ) -> None:
        """Initialize the label manager.
        
        Args:
            graph_store: The graph store to manage labels in.
            dialog_manager: Dialog manager for conflict resolution.
        """
        self._graph_store = graph_store
        self._dialog_manager = dialog_manager
        
        # Track conflicts and assignments for logging
        self._pending_conflicts: dict[str, LabelConflict] = {}
        self._resolved_conflicts: list[LabelConflict] = []
        self._assignments: list[LabelAssignment] = []

    def assign_label(
        self,
        node_id: str,
        label: str,
        is_primary: bool = False,
        source: str = "system",
    ) -> tuple[bool, LabelConflict | None]:
        """Attempt to assign a label to a node.
        
        If the label already exists on another node, creates a conflict
        but does not resolve it yet.
        
        Args:
            node_id: ID of the node to label.
            label: Label to assign.
            is_primary: Whether this should be the primary label.
            source: Source of the assignment (user, system, etc.).
            
        Returns:
            Tuple of (success, conflict_or_none).
        """
        node = self._graph_store.get_node(node_id)
        if not node:
            return (False, None)
        
        # Check if label already exists on another node
        existing_nodes = self._graph_store.get_nodes_by_label(label)
        conflicting_nodes = [n for n in existing_nodes if n.node_id != node_id]
        
        if conflicting_nodes:
            # Create a conflict
            # node_type is now a string, not enum with .value
            existing_type = conflicting_nodes[0].node_type
            new_type = node.node_type
            conflict = LabelConflict(
                conflict_id=f"conflict_{uuid.uuid4().hex[:12]}",
                label=label,
                existing_node_id=conflicting_nodes[0].node_id,
                new_node_id=node_id,
                existing_node_type=existing_type if isinstance(existing_type, str) else str(existing_type),
                new_node_type=new_type if isinstance(new_type, str) else str(new_type),
            )
            self._pending_conflicts[conflict.conflict_id] = conflict
            return (False, conflict)
        
        # No conflict - assign the label
        self._do_assign_label(node_id, label, is_primary, source)
        return (True, None)

    def _do_assign_label(
        self,
        node_id: str,
        label: str,
        is_primary: bool,
        source: str,
        conflict_id: str | None = None,
    ) -> None:
        """Actually assign the label (internal)."""
        node = self._graph_store.get_node(node_id)
        if not node:
            return
        
        if is_primary:
            # Move old primary to alternatives if different
            if node.label and node.label != label and node.label != "unknown":
                if node.label not in node.labels:
                    node.labels.append(node.label)
            node.label = label
            # Also add to label index
            self._graph_store._label_to_nodes[label.lower()].add(node_id)
        else:
            self._graph_store.add_label_to_node(node_id, label)
        
        # Record the assignment
        assignment = LabelAssignment(
            assignment_id=f"assign_{uuid.uuid4().hex[:12]}",
            node_id=node_id,
            label=label,
            is_primary=is_primary,
            source=source,
            conflict_id=conflict_id,
        )
        self._assignments.append(assignment)

    def resolve_conflict(
        self,
        conflict_id: str,
        interactive: bool = True,
    ) -> ConflictResolutionType:
        """Resolve a label conflict.
        
        Args:
            conflict_id: ID of the conflict to resolve.
            interactive: Whether to use dialog manager interactively.
            
        Returns:
            The resolution type chosen.
        """
        conflict = self._pending_conflicts.get(conflict_id)
        if not conflict:
            return ConflictResolutionType.PENDING
        
        existing_node = self._graph_store.get_node(conflict.existing_node_id)
        new_node = self._graph_store.get_node(conflict.new_node_id)
        
        if not existing_node or not new_node:
            return ConflictResolutionType.PENDING
        
        # Build resolution options
        options = [
            f"Same thing: merge '{new_node.label}' into '{existing_node.label}'",
            f"Different things: keep both with sublabels",
            f"Rename: choose a new label for one",
        ]
        
        # node_type is now a string, not enum with .value
        existing_type = existing_node.node_type if isinstance(existing_node.node_type, str) else str(existing_node.node_type)
        new_type = new_node.node_type if isinstance(new_node.node_type, str) else str(new_node.node_type)
        prompt = (
            f"Label conflict: '{conflict.label}' already exists.\n"
            f"  Existing: {existing_type} '{existing_node.label}' ({existing_node.node_id})\n"
            f"  New: {new_type} '{new_node.label}' ({new_node.node_id})\n"
            "How should this be resolved?"
        )
        
        if interactive:
            choice = self._dialog_manager.resolve_conflict(prompt, options)
        else:
            choice = 0  # Default to merge in auto mode
        
        resolution = self._apply_resolution(conflict, choice)
        
        # Mark as resolved
        conflict.resolution = resolution
        conflict.resolved_at = datetime.now()
        self._resolved_conflicts.append(conflict)
        del self._pending_conflicts[conflict_id]
        
        return resolution

    def _apply_resolution(
        self,
        conflict: LabelConflict,
        choice: int,
    ) -> ConflictResolutionType:
        """Apply the chosen resolution to a conflict."""
        existing_node = self._graph_store.get_node(conflict.existing_node_id)
        new_node = self._graph_store.get_node(conflict.new_node_id)
        
        if not existing_node or not new_node:
            return ConflictResolutionType.PENDING
        
        if choice == 0:
            # MERGE: new node becomes alias of existing
            self._graph_store.create_merge_edge(
                merged_node_id=conflict.new_node_id,
                target_node_id=conflict.existing_node_id,
            )
            # Copy labels from new to existing
            for lbl in new_node.labels:
                self._graph_store.add_label_to_node(conflict.existing_node_id, lbl)
            if new_node.label and new_node.label != "unknown":
                self._graph_store.add_label_to_node(
                    conflict.existing_node_id, new_node.label
                )
            conflict.resolution_details = {
                "merged_into": conflict.existing_node_id,
            }
            return ConflictResolutionType.MERGE
            
        elif choice == 1:
            # DISAMBIGUATE: add sublabels
            # Ask for sublabels
            # node_type is now a string, not enum with .value
            existing_type = existing_node.node_type if isinstance(existing_node.node_type, str) else str(existing_node.node_type)
            new_type = new_node.node_type if isinstance(new_node.node_type, str) else str(new_node.node_type)
            existing_sublabel = self._dialog_manager.request_label(
                f"Enter sublabel for existing '{conflict.label}' ({existing_type}):",
                [f"{conflict.label}_1", f"{conflict.label}_existing"],
            )
            new_sublabel = self._dialog_manager.request_label(
                f"Enter sublabel for new '{conflict.label}' ({new_type}):",
                [f"{conflict.label}_2", f"{conflict.label}_new"],
            )
            
            if existing_sublabel:
                self._do_assign_label(
                    conflict.existing_node_id,
                    existing_sublabel,
                    is_primary=True,
                    source="disambiguate",
                    conflict_id=conflict.conflict_id,
                )
            if new_sublabel:
                self._do_assign_label(
                    conflict.new_node_id,
                    new_sublabel,
                    is_primary=True,
                    source="disambiguate",
                    conflict_id=conflict.conflict_id,
                )
            
            conflict.resolution_details = {
                "existing_sublabel": existing_sublabel,
                "new_sublabel": new_sublabel,
            }
            return ConflictResolutionType.DISAMBIGUATE
            
        elif choice == 2:
            # RENAME: ask which to rename
            new_label = self._dialog_manager.request_label(
                f"Enter new label for '{new_node.label}':",
                [f"{conflict.label}_alt", f"new_{conflict.label}"],
            )
            if new_label:
                self._do_assign_label(
                    conflict.new_node_id,
                    new_label,
                    is_primary=True,
                    source="rename",
                    conflict_id=conflict.conflict_id,
                )
            conflict.resolution_details = {"renamed_to": new_label}
            return ConflictResolutionType.RENAME
        
        return ConflictResolutionType.KEEP_BOTH

    def auto_resolve_all(self) -> list[str]:
        """Auto-resolve all pending conflicts using defaults.
        
        Returns:
            List of resolved conflict IDs.
        """
        resolved_ids = []
        for conflict_id in list(self._pending_conflicts.keys()):
            self.resolve_conflict(conflict_id, interactive=False)
            resolved_ids.append(conflict_id)
        return resolved_ids

    def get_pending_conflicts(self) -> list[LabelConflict]:
        """Get all pending (unresolved) conflicts."""
        return list(self._pending_conflicts.values())

    def get_resolved_conflicts(self) -> list[LabelConflict]:
        """Get all resolved conflicts."""
        return list(self._resolved_conflicts)

    def get_recent_assignments(self, n: int = 10) -> list[LabelAssignment]:
        """Get the N most recent label assignments."""
        return self._assignments[-n:]

    def clear_assignment_log(self) -> None:
        """Clear the assignment log (for new run)."""
        self._assignments.clear()

    def pop_assignments(self) -> list[LabelAssignment]:
        """Pop and return all assignments since last call."""
        assignments = self._assignments.copy()
        self._assignments.clear()
        return assignments

    def pop_resolved_conflicts(self) -> list[LabelConflict]:
        """Pop and return all resolved conflicts since last call."""
        resolved = self._resolved_conflicts.copy()
        self._resolved_conflicts.clear()
        return resolved
