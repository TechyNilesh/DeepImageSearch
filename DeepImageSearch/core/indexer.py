"""
Indexing pipeline: extract embeddings, optionally caption, store in vector DB + metadata store.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from DeepImageSearch.core.embeddings import BaseEmbedding
from DeepImageSearch.core.captioner import Captioner
from DeepImageSearch.vectorstores.base import BaseVectorStore
from DeepImageSearch.metadatastore.base import BaseMetadataStore, ImageRecord

logger = logging.getLogger(__name__)


def _path_to_id(path: str) -> str:
    """Deterministic ID from file path."""
    return hashlib.md5(path.encode()).hexdigest()


class Indexer:
    """
    Index images into a vector store and metadata store.

    Parameters
    ----------
    embedding : BaseEmbedding
        Embedding backend (CLIP, timm, etc.).
    vector_store : BaseVectorStore
        Vector store backend (FAISS, Chroma, Qdrant).
    metadata_store : BaseMetadataStore or None
        Metadata store for image records. None to skip record tracking.
    captioner : Captioner or None
        Optional LLM captioner for generating captions during indexing.
    batch_size : int
        Number of images to process per batch.
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        metadata_store: Optional[BaseMetadataStore] = None,
        captioner: Optional[Captioner] = None,
        batch_size: int = 64,
    ):
        self.embedding = embedding
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.captioner = captioner
        self.batch_size = batch_size

    def index(
        self,
        image_paths: List[str],
        extra_metadata: Optional[List[Dict[str, Any]]] = None,
        generate_captions: bool = False,
        caption_prompt: Optional[str] = None,
    ) -> int:
        """
        Index a list of images.

        Parameters
        ----------
        image_paths : list[str]
            Paths to images to index.
        extra_metadata : list[dict] or None
            Additional metadata per image (must match length of image_paths).
        generate_captions : bool
            If True and a captioner is set, generate captions for each image.
        caption_prompt : str or None
            Custom captioning prompt.

        Returns
        -------
        int
            Number of images successfully indexed.
        """
        if not image_paths:
            logger.warning("No images to index")
            return 0

        if extra_metadata and len(extra_metadata) != len(image_paths):
            raise ValueError("extra_metadata length must match image_paths length")

        # Generate captions if requested
        captions: Dict[str, str] = {}
        if generate_captions and self.captioner:
            logger.info("Generating captions with LLM...")
            captions = self.captioner.caption_batch(image_paths, prompt=caption_prompt)

        # Determine starting index for new images
        start_index = 0
        if self.metadata_store:
            start_index = self.metadata_store.next_index()

        total_indexed = 0
        global_valid_count = 0

        for batch_start in tqdm(range(0, len(image_paths), self.batch_size), desc="Indexing"):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            # Load images
            images = []
            valid_indices = []
            for i, path in enumerate(batch_paths):
                try:
                    img = Image.open(path)
                    img.load()
                    images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")

            if not images:
                continue

            # Extract embeddings
            vectors = self.embedding.embed_images(images)

            # Close images
            for img in images:
                img.close()

            # Build IDs, metadata, and records
            ids = []
            metadata_list = []
            records = []
            now = datetime.now(timezone.utc).isoformat()

            for idx in valid_indices:
                abs_idx = batch_start + idx
                path = batch_paths[idx]
                image_id = _path_to_id(path)
                image_index = start_index + global_valid_count
                image_name = os.path.basename(path)
                caption = captions.get(path, "")

                ids.append(image_id)

                # Enriched metadata for vector store
                meta: Dict[str, Any] = {
                    "image_id": image_id,
                    "image_index": image_index,
                    "image_name": image_name,
                    "image_path": path,
                    "indexed_at": now,
                }

                if caption:
                    meta["caption"] = caption

                if extra_metadata and abs_idx < len(extra_metadata):
                    meta.update(extra_metadata[abs_idx])

                metadata_list.append(meta)

                # Build ImageRecord for metadata store
                extra = {}
                if extra_metadata and abs_idx < len(extra_metadata):
                    extra = extra_metadata[abs_idx]

                records.append(ImageRecord(
                    image_id=image_id,
                    image_index=image_index,
                    image_name=image_name,
                    image_path=path,
                    caption=caption or None,
                    indexed_at=now,
                    extra=extra,
                ))

                global_valid_count += 1

            # Store vectors
            self.vector_store.add(ids, vectors, metadata_list)

            # Store records
            if self.metadata_store and records:
                self.metadata_store.add(records)

            total_indexed += len(ids)

        logger.info(f"Indexed {total_indexed} images")
        return total_indexed

    def add_images(
        self,
        image_paths: List[str],
        extra_metadata: Optional[List[Dict[str, Any]]] = None,
        generate_captions: bool = False,
    ) -> int:
        """Alias for index() — add new images to an existing index."""
        return self.index(image_paths, extra_metadata, generate_captions)
