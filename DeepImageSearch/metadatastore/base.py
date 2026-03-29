"""
Abstract metadata store interface and ImageRecord dataclass.

The ImageRecord schema is designed to map directly to a SQL table,
enabling future database backends (PostgreSQL, SQLite, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ImageRecord:
    """
    A single image record in the metadata store.

    Maps directly to a database row:
        image_id    TEXT PRIMARY KEY  — deterministic hash of path
        image_index INTEGER UNIQUE    — auto-incrementing position
        image_name  TEXT              — filename (basename)
        image_path  TEXT              — full absolute path
        caption     TEXT NULL         — LLM-generated caption
        indexed_at  TEXT              — ISO 8601 timestamp
        extra       JSON NULL         — additional user metadata
    """

    image_id: str
    image_index: int
    image_name: str
    image_path: str
    caption: Optional[str] = None
    indexed_at: str = ""
    extra: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=data["image_id"],
            image_index=data["image_index"],
            image_name=data["image_name"],
            image_path=data["image_path"],
            caption=data.get("caption"),
            indexed_at=data.get("indexed_at", ""),
            extra=data.get("extra", {}),
        )


class BaseMetadataStore(ABC):
    """
    Abstract interface for image record storage.

    Implementations: JsonMetadataStore (local), PostgresMetadataStore (future).
    """

    @abstractmethod
    def add(self, records: List[ImageRecord]) -> None:
        """Add image records to the store."""

    @abstractmethod
    def get(self, image_id: str) -> Optional[ImageRecord]:
        """Get a record by image_id."""

    @abstractmethod
    def get_by_index(self, image_index: int) -> Optional[ImageRecord]:
        """Get a record by image_index."""

    @abstractmethod
    def list_all(self) -> List[ImageRecord]:
        """Return all records."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of records."""

    @abstractmethod
    def delete(self, image_ids: List[str]) -> None:
        """Delete records by image_id."""

    @abstractmethod
    def next_index(self) -> int:
        """Return the next available image_index value."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist records to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load records from disk."""
