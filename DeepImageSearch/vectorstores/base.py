"""Abstract vector store interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseVectorStore(ABC):
    """
    Abstract interface for vector storage backends.

    All stores must support:
    - Adding vectors with IDs and optional metadata
    - Searching by vector with optional metadata filters
    - Persistence (save/load)
    """

    @abstractmethod
    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add vectors to the store.

        Parameters
        ----------
        ids : list[str]
            Unique identifiers for each vector.
        vectors : np.ndarray
            (N, D) float32 array of vectors.
        metadata : list[dict] or None
            Optional metadata per vector.
        """

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Parameters
        ----------
        query_vector : np.ndarray
            1-D query vector.
        k : int
            Number of results.
        filters : dict or None
            Metadata filters (store-specific).

        Returns
        -------
        list[dict]
            Each dict has 'id', 'score', and 'metadata'.
        """

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the store to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the store from disk."""
