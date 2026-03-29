"""FAISS-based vector store with metadata support."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from DeepImageSearch.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSStore(BaseVectorStore):
    """
    FAISS vector store with sidecar metadata.

    Parameters
    ----------
    dimension : int
        Vector dimension.
    index_type : str
        'flat' (exact), 'ivf' (approximate), or 'hnsw' (approximate).
    """

    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index: Optional[faiss.Index] = None
        self._ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._build_index(dimension)

    def _build_index(self, dimension: int) -> None:
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # inner product for cosine sim on normalised vectors
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")
        logger.info(f"Created FAISS {self.index_type} index (dim={dimension})")

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(vectors)

        self.index.add(vectors)
        self._ids.extend(ids)

        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in ids])

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        if self.index_type == "ivf":
            self.index.nprobe = min(20, self.index.nlist)

        # If filters are set, search more candidates and post-filter
        search_k = k * 5 if filters else k
        search_k = min(search_k, self.index.ntotal)

        if search_k == 0:
            return []

        scores, indices = self.index.search(query, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            if filters and not self._matches_filters(meta, filters):
                continue
            results.append({
                "id": self._ids[idx],
                "score": float(score),
                "metadata": meta,
            })
            if len(results) >= k:
                break

        return results

    @staticmethod
    def _matches_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in meta:
                return False
            if isinstance(value, list):
                if meta[key] not in value:
                    return False
            elif meta[key] != value:
                return False
        return True

    def delete(self, ids: List[str]) -> None:
        # FAISS doesn't support deletion natively on all index types
        # Rebuild index without the deleted IDs
        ids_set = set(ids)
        keep_mask = [i for i, _id in enumerate(self._ids) if _id not in ids_set]

        if len(keep_mask) == len(self._ids):
            return

        # Reconstruct all vectors
        all_vectors = np.vstack([self.index.reconstruct(i) for i in keep_mask]).astype(np.float32)
        new_ids = [self._ids[i] for i in keep_mask]
        new_meta = [self._metadata[i] for i in keep_mask]

        self._build_index(self.dimension)
        self._ids = []
        self._metadata = []
        self.add(new_ids, all_vectors, new_meta)

    def count(self) -> int:
        return self.index.ntotal

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({"ids": self._ids, "metadata": self._metadata, "index_type": self.index_type}, f)
        logger.info(f"FAISS store saved to {path} ({self.count()} vectors)")

    def load(self, path: str) -> None:
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.json")) as f:
            data = json.load(f)
        self._ids = data["ids"]
        self._metadata = data["metadata"]
        self.index_type = data.get("index_type", "flat")
        self.dimension = self.index.d
        logger.info(f"FAISS store loaded from {path} ({self.count()} vectors)")
