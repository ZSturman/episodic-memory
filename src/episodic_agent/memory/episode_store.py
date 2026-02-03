"""Persistent episode store using JSONL files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from episodic_agent.core.interfaces import EpisodeStore
from episodic_agent.schemas import Episode


def _episode_to_dict(episode: Episode) -> dict[str, Any]:
    """Convert an Episode to a JSON-serializable dictionary."""
    return {
        "episode_id": episode.episode_id,
        "created_at": episode.created_at.isoformat(),
        "start_time": episode.start_time.isoformat(),
        "end_time": episode.end_time.isoformat(),
        "step_count": episode.step_count,
        "location_label": episode.location_label,
        "location_confidence": episode.location_confidence,
        "location_embedding": episode.location_embedding,
        "entities": [e.model_dump() for e in episode.entities],
        "events": episode.events,
        "episode_embedding": episode.episode_embedding,
        "source_acf_id": episode.source_acf_id,
        "boundary_reason": episode.boundary_reason,
        "extras": episode.extras,
    }


def _dict_to_episode(data: dict[str, Any]) -> Episode:
    """Convert a dictionary back to an Episode."""
    from episodic_agent.schemas import ObjectCandidate
    
    return Episode(
        episode_id=data["episode_id"],
        created_at=datetime.fromisoformat(data["created_at"]),
        start_time=datetime.fromisoformat(data["start_time"]),
        end_time=datetime.fromisoformat(data["end_time"]),
        step_count=data["step_count"],
        location_label=data["location_label"],
        location_confidence=data["location_confidence"],
        location_embedding=data.get("location_embedding"),
        entities=[ObjectCandidate(**e) for e in data.get("entities", [])],
        events=data.get("events", []),
        episode_embedding=data.get("episode_embedding"),
        source_acf_id=data["source_acf_id"],
        boundary_reason=data.get("boundary_reason", "unknown"),
        extras=data.get("extras", {}),
    )


class PersistentEpisodeStore(EpisodeStore):
    """Append-only episode store persisted to JSONL files.
    
    Episodes are written as immutable records to a JSONL file.
    The store also maintains an in-memory index for fast lookups.
    """

    def __init__(self, storage_path: Path) -> None:
        """Initialize the persistent episode store.
        
        Args:
            storage_path: Path to the JSONL file for episode storage.
        """
        self._storage_path = storage_path
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory index for fast lookups
        self._episodes: dict[str, Episode] = {}
        
        # Load existing episodes if file exists
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing episodes from the storage file."""
        if not self._storage_path.exists():
            return
            
        with open(self._storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    episode = _dict_to_episode(data)
                    self._episodes[episode.episode_id] = episode
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Skip malformed lines
                    continue

    def store(self, episode: Episode) -> None:
        """Store a frozen episode (append-only).
        
        Args:
            episode: Episode to persist.
        """
        # Write to file first (append-only)
        with open(self._storage_path, "a", encoding="utf-8") as f:
            line = json.dumps(_episode_to_dict(episode), separators=(",", ":"))
            f.write(line + "\n")
        
        # Update in-memory index
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

    def get_by_location(self, location_label: str) -> list[Episode]:
        """Get all episodes at a specific location.
        
        Args:
            location_label: Location to filter by.
            
        Returns:
            List of episodes at that location.
        """
        return [
            ep for ep in self._episodes.values()
            if ep.location_label == location_label
        ]

    def get_recent(self, n: int = 10) -> list[Episode]:
        """Get the N most recent episodes.
        
        Args:
            n: Number of episodes to return.
            
        Returns:
            List of recent episodes, newest first.
        """
        sorted_episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.end_time,
            reverse=True,
        )
        return sorted_episodes[:n]
