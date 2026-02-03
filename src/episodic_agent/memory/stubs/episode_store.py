"""In-memory episode store for Phase 1 testing."""

from __future__ import annotations

from episodic_agent.core.interfaces import EpisodeStore
from episodic_agent.schemas import Episode


class InMemoryEpisodeStore(EpisodeStore):
    """Simple in-memory episode storage.
    
    Stores episodes in a dictionary. Not persisted across runs.
    Suitable for testing and Phase 1 demonstrations.
    """

    def __init__(self) -> None:
        """Initialize an empty episode store."""
        self._episodes: dict[str, Episode] = {}

    def store(self, episode: Episode) -> None:
        """Store a frozen episode.
        
        Args:
            episode: Episode to persist.
        """
        self._episodes[episode.episode_id] = episode

    def get(self, episode_id: str) -> Episode | None:
        """Retrieve an episode by ID.
        
        Args:
            episode_id: ID of the episode to retrieve.
            
        Returns:
            The Episode if found, None otherwise.
        """
        return self._episodes.get(episode_id)

    def get_all(self) -> list[Episode]:
        """Retrieve all stored episodes.
        
        Returns:
            List of all episodes in storage order.
        """
        return list(self._episodes.values())

    def count(self) -> int:
        """Get the number of stored episodes.
        
        Returns:
            Count of episodes in storage.
        """
        return len(self._episodes)

    def clear(self) -> None:
        """Clear all episodes from storage."""
        self._episodes.clear()
