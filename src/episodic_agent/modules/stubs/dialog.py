"""Stub dialog manager for Phase 1 testing."""

from __future__ import annotations

from episodic_agent.core.interfaces import DialogManager


class StubDialogManager(DialogManager):
    """Non-blocking stub dialog manager.
    
    Returns default values immediately without user interaction.
    Useful for automated testing and demos.
    """

    def __init__(
        self,
        auto_accept: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize the stub dialog manager.
        
        Args:
            auto_accept: Whether to auto-accept all prompts.
            seed: Random seed (unused in stub).
        """
        self._auto_accept = auto_accept
        self._seed = seed

    def request_label(
        self,
        prompt: str,
        candidates: list[str],
    ) -> str | None:
        """Request a label - returns first candidate or None.
        
        Args:
            prompt: Description of what needs labeling.
            candidates: Suggested label options.
            
        Returns:
            First candidate if available, None otherwise.
        """
        if self._auto_accept and candidates:
            return candidates[0]
        return None

    def confirm(self, prompt: str, default: bool = True) -> bool:
        """Request confirmation - returns default value.
        
        Args:
            prompt: Yes/no question to ask.
            default: Default response.
            
        Returns:
            The default value.
        """
        return default if self._auto_accept else False

    def resolve_conflict(
        self,
        prompt: str,
        options: list[str],
    ) -> int:
        """Resolve a conflict - returns first option.
        
        Args:
            prompt: Description of the conflict.
            options: Available resolution options.
            
        Returns:
            Index 0 (first option).
        """
        return 0
