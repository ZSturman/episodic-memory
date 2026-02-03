"""Dialog manager for agent interactions."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from episodic_agent.schemas.labels import LabelConflict


class DialogManager(ABC):
    """Abstract base for dialog managers."""

    @abstractmethod
    def confirm(self, message: str, default: bool = True) -> bool:
        """Ask for yes/no confirmation.
        
        Args:
            message: The question to ask.
            default: Default response if auto_accept.
            
        Returns:
            True for yes, False for no.
        """
        ...

    @abstractmethod
    def ask_label(self, prompt: str, suggestions: list[str] | None = None) -> str:
        """Ask user to provide a label.
        
        Args:
            prompt: The prompt to display.
            suggestions: Optional list of suggested labels.
            
        Returns:
            The label string provided.
        """
        ...

    @abstractmethod
    def resolve_conflict(
        self, conflict: "LabelConflict | str", options: list[str]
    ) -> int:
        """Present a conflict and ask user to select resolution.
        
        Args:
            conflict: The label conflict or prompt string.
            options: List of resolution options.
            
        Returns:
            Index of selected option (0-based).
        """
        ...

    @abstractmethod
    def notify(self, message: str) -> None:
        """Display a notification to the user.
        
        Args:
            message: The message to display.
        """
        ...


class AutoAcceptDialogManager(DialogManager):
    """Dialog manager that auto-accepts all prompts.
    
    Used for automated runs, tests, and non-interactive modes.
    Records all interactions for later inspection.
    """

    def __init__(self, default_label: str = "auto_label") -> None:
        """Initialize with default responses.
        
        Args:
            default_label: Default label to return for ask_label.
        """
        self._default_label = default_label
        self._label_counter = 0
        self._interactions: list[dict] = []

    def confirm(self, message: str, default: bool = True) -> bool:
        """Auto-accept confirmation with default."""
        self._interactions.append({
            "type": "confirm",
            "message": message,
            "response": default,
        })
        return default

    def ask_label(self, prompt: str, suggestions: list[str] | None = None) -> str:
        """Return first suggestion or generate a label."""
        if suggestions:
            label = suggestions[0]
        else:
            self._label_counter += 1
            label = f"{self._default_label}_{self._label_counter}"
        
        self._interactions.append({
            "type": "ask_label",
            "prompt": prompt,
            "suggestions": suggestions,
            "response": label,
        })
        return label

    def resolve_conflict(
        self, conflict: "LabelConflict | str", options: list[str]
    ) -> int:
        """Always select first option."""
        # Handle both LabelConflict objects and string prompts
        conflict_label = conflict.label if hasattr(conflict, "label") else str(conflict)[:50]
        self._interactions.append({
            "type": "resolve_conflict",
            "conflict_label": conflict_label,
            "options": options,
            "response": 0,
        })
        return 0

    def notify(self, message: str) -> None:
        """Record notification silently."""
        self._interactions.append({
            "type": "notify",
            "message": message,
        })

    @property
    def interactions(self) -> list[dict]:
        """Get recorded interactions."""
        return list(self._interactions)

    def clear_interactions(self) -> None:
        """Clear recorded interactions."""
        self._interactions.clear()


class CLIDialogManager(DialogManager):
    """Interactive CLI dialog manager.
    
    Prompts user via stdin/stdout for all interactions.
    """

    def __init__(
        self,
        input_stream=None,
        output_stream=None,
        error_stream=None,
    ) -> None:
        """Initialize with optional custom streams.
        
        Args:
            input_stream: Input stream (default: sys.stdin).
            output_stream: Output stream (default: sys.stdout).
            error_stream: Error stream (default: sys.stderr).
        """
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout
        self._error = error_stream or sys.stderr

    def _print(self, message: str, end: str = "\n") -> None:
        """Print to output stream."""
        self._output.write(f"{message}{end}")
        self._output.flush()

    def _read(self, prompt: str = "") -> str:
        """Read from input stream."""
        if prompt:
            self._output.write(prompt)
            self._output.flush()
        return self._input.readline().strip()

    def confirm(self, message: str, default: bool = True) -> bool:
        """Ask for yes/no confirmation."""
        default_str = "Y/n" if default else "y/N"
        prompt = f"{message} [{default_str}]: "
        
        while True:
            response = self._read(prompt).lower()
            
            if response == "":
                return default
            elif response in ("y", "yes"):
                return True
            elif response in ("n", "no"):
                return False
            else:
                self._print("Please enter 'y' or 'n'")

    def ask_label(self, prompt: str, suggestions: list[str] | None = None) -> str:
        """Ask user to provide a label."""
        self._print(prompt)
        
        if suggestions:
            self._print("Suggestions:")
            for i, s in enumerate(suggestions, 1):
                self._print(f"  {i}. {s}")
            self._print(f"Enter a number (1-{len(suggestions)}) or type a new label:")
        
        while True:
            response = self._read("> ")
            
            if not response:
                if suggestions:
                    return suggestions[0]
                self._print("Please enter a label")
                continue
            
            # Check if it's a number selecting a suggestion
            if suggestions and response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(suggestions):
                    return suggestions[idx]
            
            return response

    def resolve_conflict(
        self, conflict: "LabelConflict | str", options: list[str]
    ) -> int:
        """Present a conflict and ask user to select resolution."""
        # Handle both LabelConflict objects and string prompts
        if hasattr(conflict, "label"):
            self._print(f"\n⚠️  Label Conflict: '{conflict.label}'")
            self._print(f"   Existing node: {conflict.existing_node_id}")
            self._print(f"   New node: {conflict.new_node_id}")
            self._print(f"   Reason: {conflict.reason}")
        else:
            self._print(f"\n{conflict}")
        self._print("")
        self._print("Resolution options:")
        
        for i, opt in enumerate(options, 1):
            self._print(f"  {i}. {opt}")
        
        while True:
            response = self._read(f"Select option (1-{len(options)}): ")
            
            if response.isdigit():
                idx = int(response) - 1
                if 0 <= idx < len(options):
                    return idx
            
            self._print(f"Please enter a number between 1 and {len(options)}")

    def notify(self, message: str) -> None:
        """Display a notification."""
        self._print(f"ℹ️  {message}")


class NonBlockingDialogManager(DialogManager):
    """Dialog manager that queues prompts for later resolution.
    
    Useful when the agent should continue running while
    conflicts are resolved asynchronously.
    """

    def __init__(self, fallback: DialogManager | None = None) -> None:
        """Initialize with optional fallback manager.
        
        Args:
            fallback: Manager to use for immediate resolution.
                     If None, uses AutoAcceptDialogManager defaults.
        """
        self._fallback = fallback or AutoAcceptDialogManager()
        self._pending_confirmations: list[tuple[str, bool]] = []
        self._pending_labels: list[tuple[str, list[str] | None]] = []
        self._pending_conflicts: list[tuple["LabelConflict", list[str]]] = []
        self._notifications: list[str] = []

    def confirm(self, message: str, default: bool = True) -> bool:
        """Queue confirmation and return default."""
        self._pending_confirmations.append((message, default))
        return default

    def ask_label(self, prompt: str, suggestions: list[str] | None = None) -> str:
        """Queue label request and return fallback."""
        self._pending_labels.append((prompt, suggestions))
        return self._fallback.ask_label(prompt, suggestions)

    def resolve_conflict(
        self, conflict: "LabelConflict", options: list[str]
    ) -> int:
        """Queue conflict and return fallback resolution."""
        self._pending_conflicts.append((conflict, options))
        return self._fallback.resolve_conflict(conflict, options)

    def notify(self, message: str) -> None:
        """Queue notification."""
        self._notifications.append(message)

    @property
    def has_pending(self) -> bool:
        """Check if there are pending items."""
        return bool(
            self._pending_confirmations or
            self._pending_labels or
            self._pending_conflicts
        )

    def get_pending_confirmations(self) -> list[tuple[str, bool]]:
        """Get pending confirmations."""
        return list(self._pending_confirmations)

    def get_pending_labels(self) -> list[tuple[str, list[str] | None]]:
        """Get pending label requests."""
        return list(self._pending_labels)

    def get_pending_conflicts(self) -> list[tuple["LabelConflict", list[str]]]:
        """Get pending conflicts."""
        return list(self._pending_conflicts)

    def get_notifications(self) -> list[str]:
        """Get notifications."""
        return list(self._notifications)

    def clear_all(self) -> None:
        """Clear all pending items."""
        self._pending_confirmations.clear()
        self._pending_labels.clear()
        self._pending_conflicts.clear()
        self._notifications.clear()
