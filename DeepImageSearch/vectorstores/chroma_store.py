# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""ChromaDB-based vector store."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from DeepImageSearch.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """
    ChromaDB vector store backend.

    Requires: pip install chromadb

    Parameters
    ----------
    collection_name : str
        Name of the Chroma collection.
    persist_directory : str or None
        Directory for persistent storage. None for in-memory.
    """

    def __init__(self, collection_name: str = "deep_image_search", persist_directory: Optional[str] = None):
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb is required. Install with: uv pip install 'DeepImageSearch[chroma]'")

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.persist_directory = persist_directory
        logger.info(f"ChromaDB collection '{collection_name}' ready ({self.count()} vectors)")

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        vectors = vectors.astype(np.float32)
        # Chroma needs metadata values to be str, int, float, or bool, and it
        # rejects empty dicts outright — so absent metadata must be sent as None.
        clean_metadata = None
        if metadata:
            clean_metadata = []
            for m in metadata:
                clean = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
                clean_metadata.append(clean or None)

        # ChromaDB has a batch limit, chunk if needed
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=vectors[i:end].tolist(),
                metadatas=clean_metadata[i:end] if clean_metadata else None,
            )
        logger.info(f"Added {len(ids)} vectors to ChromaDB")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        where = filters if filters else None
        results = self.collection.query(
            query_embeddings=[query_vector.astype(np.float32).tolist()],
            n_results=min(k, self.count()) if self.count() > 0 else k,
            where=where,
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for idx, (id_, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
                # Chroma hands back None for records stored without metadata;
                # the BaseVectorStore contract promises a dict.
                meta = (results["metadatas"][0][idx] if results["metadatas"] else None) or {}
                output.append({
                    "id": id_,
                    "score": 1.0 - dist,  # Chroma returns distance, convert to similarity
                    "metadata": meta,
                })
        return output

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()

    def save(self, path: str) -> None:
        # ChromaDB PersistentClient auto-persists
        logger.info("ChromaDB auto-persists (no manual save needed)")

    def load(self, path: str) -> None:
        # Re-init with the persist directory
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
