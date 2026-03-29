"""
Unified search engine supporting text, image, and hybrid queries.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from PIL import Image

from DeepImageSearch.core.embeddings import BaseEmbedding
from DeepImageSearch.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class Searcher:
    """
    Multi-modal search over an indexed image collection.

    Supports:
    - Image-to-image search (visual similarity)
    - Text-to-image search (semantic, requires CLIP-family embedding)
    - Hybrid search (combine text and image queries)
    - Metadata filtering
    - LLM re-ranking (optional)

    Parameters
    ----------
    embedding : BaseEmbedding
        Embedding backend.
    vector_store : BaseVectorStore
        Vector store with indexed images.
    """

    def __init__(self, embedding: BaseEmbedding, vector_store: BaseVectorStore):
        self.embedding = embedding
        self.vector_store = vector_store

    def search(
        self,
        query: Union[str, Image.Image, np.ndarray],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        mode: Literal["auto", "text", "image", "hybrid"] = "auto",
        text_weight: float = 0.5,
        image_query: Optional[Union[str, Image.Image]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images.

        Parameters
        ----------
        query : str or PIL.Image or np.ndarray
            - str: text query (requires CLIP-family embedding) or image path
            - PIL.Image: query image
            - np.ndarray: pre-computed query vector
        k : int
            Number of results to return.
        filters : dict or None
            Metadata filters passed to the vector store.
        mode : str
            'auto' detects query type, 'text' forces text search,
            'image' forces image search, 'hybrid' combines both.
        text_weight : float
            Weight for text query in hybrid mode (0-1). Image weight = 1 - text_weight.
        image_query : str or PIL.Image or None
            Secondary image query for hybrid mode.

        Returns
        -------
        list[dict]
            Each dict has 'id', 'score', 'metadata' (including 'image_path').
        """
        if mode == "hybrid":
            return self._hybrid_search(query, image_query, k, filters, text_weight)

        query_vector = self._resolve_query(query, mode)
        return self.vector_store.search(query_vector, k=k, filters=filters)

    def search_by_image(
        self,
        image_path: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using an image file path."""
        img = Image.open(image_path)
        vector = self.embedding.embed_image(img)
        img.close()
        return self.vector_store.search(vector, k=k, filters=filters)

    def search_by_text(
        self,
        text: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using a text query (requires CLIP-family embedding)."""
        if not self.embedding.supports_text:
            raise ValueError(
                "Text search requires a CLIP-family embedding model. "
                "Use EmbeddingManager.create('clip-vit-b-32') or similar."
            )
        vector = self.embedding.embed_text(text)
        return self.vector_store.search(vector, k=k, filters=filters)

    def _hybrid_search(
        self,
        text_query: Union[str, Any],
        image_query: Optional[Union[str, Image.Image]],
        k: int,
        filters: Optional[Dict[str, Any]],
        text_weight: float,
    ) -> List[Dict[str, Any]]:
        """Combine text and image queries with weighted fusion."""
        if not self.embedding.supports_text:
            raise ValueError("Hybrid search requires a CLIP-family embedding model")

        # Get text vector
        if isinstance(text_query, str):
            text_vector = self.embedding.embed_text(text_query)
        else:
            raise ValueError("In hybrid mode, the primary query must be a text string")

        # Get image vector
        if image_query is None:
            raise ValueError("In hybrid mode, image_query must be provided")
        if isinstance(image_query, str):
            img = Image.open(image_query)
            image_vector = self.embedding.embed_image(img)
            img.close()
        elif isinstance(image_query, Image.Image):
            image_vector = self.embedding.embed_image(image_query)
        else:
            raise ValueError("image_query must be a file path or PIL Image")

        # Weighted combination
        combined = text_weight * text_vector + (1.0 - text_weight) * image_vector
        combined = combined / np.linalg.norm(combined)

        return self.vector_store.search(combined, k=k, filters=filters)

    def _resolve_query(self, query: Union[str, Image.Image, np.ndarray], mode: str) -> np.ndarray:
        """Convert a query into a vector."""
        if isinstance(query, np.ndarray):
            return query

        if isinstance(query, Image.Image):
            return self.embedding.embed_image(query)

        if isinstance(query, str):
            if mode == "text" or (mode == "auto" and not self._looks_like_path(query)):
                if not self.embedding.supports_text:
                    raise ValueError(
                        f"Text search not supported by {type(self.embedding).__name__}. "
                        "Use a CLIP model for text queries."
                    )
                return self.embedding.embed_text(query)
            else:
                # Treat as image path
                img = Image.open(query)
                vector = self.embedding.embed_image(img)
                img.close()
                return vector

        raise TypeError(f"Unsupported query type: {type(query)}")

    @staticmethod
    def _looks_like_path(s: str) -> bool:
        """Heuristic: does this string look like a file path?"""
        import os
        if os.sep in s or s.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")):
            return True
        if os.path.exists(s):
            return True
        return False

    def get_similar_images(self, image_path: str, number_of_images: int = 10) -> Dict[int, str]:
        """
        Backward-compatible method returning {index: path} dict.

        Parameters
        ----------
        image_path : str
            Path to query image.
        number_of_images : int
            Number of results.

        Returns
        -------
        dict[int, str]
        """
        results = self.search_by_image(image_path, k=number_of_images)
        return {i: r["metadata"].get("image_path", r["id"]) for i, r in enumerate(results)}

    def plot_similar_images(self, image_path: str, number_of_images: int = 6) -> None:
        """Plot query image and similar results using matplotlib."""
        import math
        import matplotlib.pyplot as plt
        from PIL import ImageOps

        results = self.search_by_image(image_path, k=number_of_images)

        # Show input
        input_img = Image.open(image_path)
        input_resized = ImageOps.fit(input_img, (224, 224), Image.LANCZOS)
        plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.title("Input Image", fontsize=18)
        plt.imshow(input_resized)
        plt.show()

        # Show results
        grid = math.ceil(math.sqrt(number_of_images))
        fig = plt.figure(figsize=(20, 15))
        for i, r in enumerate(results):
            path = r["metadata"].get("image_path", "")
            if not path:
                continue
            try:
                ax = fig.add_subplot(grid, grid, i + 1)
                ax.axis("off")
                img = Image.open(path)
                img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
                ax.imshow(img_resized)
                score = r.get("score", 0)
                ax.set_title(f"Score: {score:.3f}", fontsize=10)
            except Exception as e:
                logger.error(f"Error displaying {path}: {e}")

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle("Similar Results", fontsize=22)
        plt.show()
