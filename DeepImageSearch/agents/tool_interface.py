"""
Generic tool interface for DeepImageSearch that can be used by any agent framework.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from DeepImageSearch.core.embeddings import EmbeddingManager
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.vectorstores.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class ImageSearchTool:
    """
    A ready-to-use image search tool for LLM agents.

    Wraps the full DeepImageSearch pipeline into a single callable
    that agents can invoke for text-to-image and image-to-image search.

    Parameters
    ----------
    index_path : str
        Path to a saved vector store index.
    model_name : str
        Embedding model preset name.
    vector_store_type : str
        'faiss', 'chroma', or 'qdrant'.
    device : str or None
        Device for model inference.
    """

    def __init__(
        self,
        index_path: str,
        model_name: str = "clip-vit-b-32",
        vector_store_type: str = "faiss",
        device: Optional[str] = None,
    ):
        self.embedding = EmbeddingManager.create(model_name, device=device)
        self.vector_store = self._load_store(vector_store_type, index_path, self.embedding.dimension)
        self.searcher = Searcher(self.embedding, self.vector_store)

    @staticmethod
    def _load_store(store_type: str, path: str, dimension: int):
        if store_type == "faiss":
            store = FAISSStore(dimension=dimension)
            store.load(path)
            return store
        elif store_type == "chroma":
            from DeepImageSearch.vectorstores.chroma_store import ChromaStore
            return ChromaStore(persist_directory=path)
        elif store_type == "qdrant":
            from DeepImageSearch.vectorstores.qdrant_store import QdrantStore
            return QdrantStore(path=path, dimension=dimension)
        raise ValueError(f"Unknown store type: {store_type}")

    def __call__(
        self,
        query: str,
        k: int = 5,
        mode: str = "auto",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for images matching a query.

        Parameters
        ----------
        query : str
            Natural language description or image file path.
        k : int
            Number of results.
        mode : str
            'auto', 'text', or 'image'.
        filters : dict or None
            Metadata filters.

        Returns
        -------
        list[dict]
            Search results with 'id', 'score', 'metadata'.
        """
        return self.searcher.search(query, k=k, filters=filters, mode=mode)

    @property
    def tool_definition(self) -> Dict[str, Any]:
        """Return a tool definition compatible with Claude/OpenAI function calling."""
        return {
            "name": "search_images",
            "description": (
                "Search an indexed image collection using natural language text queries "
                "or image file paths. Returns the most similar images with scores and metadata."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query or path to a query image",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "text", "image"],
                        "description": "Search mode: 'text' for semantic search, 'image' for visual similarity, 'auto' to detect",
                        "default": "auto",
                    },
                },
                "required": ["query"],
            },
        }
