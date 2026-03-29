"""
Backward-compatible shim for DeepImageSearch v2 API.

Delegates to v3 modules. Existing code using Load_Data and Search_Setup
will continue to work without changes.
"""

import logging
import os
from typing import Dict, List, Optional, Union

from DeepImageSearch.data.loader import Load_Data
from DeepImageSearch.core.embeddings import EmbeddingManager
from DeepImageSearch.core.indexer import Indexer
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.vectorstores.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

# Re-export Load_Data so old imports still work
__all__ = ["Load_Data", "Search_Setup"]


class Search_Setup:
    """
    v2-compatible search setup class.

    Wraps v3 modules (EmbeddingManager, FAISSStore, Indexer, Searcher)
    behind the original Search_Setup interface.

    Parameters
    ----------
    image_list : list
        A list of image paths to be indexed and searched.
    model_name : str, optional (default='vgg19')
        Model name. Use CLIP presets for text+image search
        (e.g. 'clip-vit-b-32') or timm names for image-only.
    pretrained : bool, optional (default=True)
        Whether to use pretrained weights.
    image_count : int, optional (default=None)
        Limit number of images. None uses all.
    image_size : int, optional (default=224)
        Input image resolution (only for timm models).
    metadata_dir : str, optional (default='metadata-files')
        Directory to store index files.
    use_gpu : bool, optional (default=False)
        Whether to use GPU.
    index_type : str, optional (default='flat')
        FAISS index type: 'flat', 'ivf', 'hnsw'.
    """

    def __init__(
        self,
        image_list: List[str],
        model_name: str = "vgg19",
        pretrained: bool = True,
        image_count: Optional[int] = None,
        image_size: int = 224,
        metadata_dir: str = "metadata-files",
        use_gpu: bool = False,
        index_type: str = "flat",
    ):
        if not image_list:
            raise ValueError("image_list cannot be empty")
        if not isinstance(image_list, list):
            raise TypeError("image_list must be a list")

        self.model_name = model_name
        self.metadata_dir = metadata_dir
        self.index_type = index_type

        if image_count is not None:
            self.image_list = image_list[:image_count]
        else:
            self.image_list = image_list

        # Detect device
        device = None
        if use_gpu:
            import torch
            if torch.cuda.is_available():
                device = "cuda"

        # Create v3 components
        self.embedding = EmbeddingManager.create(
            model_name,
            device=device,
            pretrained=pretrained,
            image_size=image_size,
        )

        index_dir = os.path.join(metadata_dir, model_name)
        os.makedirs(index_dir, exist_ok=True)

        self.vector_store = FAISSStore(
            dimension=self.embedding.dimension,
            index_type=index_type,
        )

        # Try loading existing index
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            self.vector_store.load(index_dir)

        self.indexer = Indexer(
            embedding=self.embedding,
            vector_store=self.vector_store,
        )
        self.searcher = Searcher(
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        self._index_dir = index_dir
        logger.info(f"Search_Setup initialized: {len(self.image_list)} images, model={model_name}")

    def run_index(self, force_reindex: bool = False):
        """Index the images. Skips if index already exists unless force_reindex=True."""
        if self.vector_store.count() > 0 and not force_reindex:
            logger.info("Index already exists. Use force_reindex=True to rebuild.")
            return

        if force_reindex and self.vector_store.count() > 0:
            # Rebuild: create fresh store
            self.vector_store = FAISSStore(
                dimension=self.embedding.dimension,
                index_type=self.index_type,
            )
            self.indexer = Indexer(
                embedding=self.embedding,
                vector_store=self.vector_store,
            )
            self.searcher = Searcher(
                embedding=self.embedding,
                vector_store=self.vector_store,
            )

        self.indexer.index(self.image_list)
        self.vector_store.save(self._index_dir)
        logger.info(f"Indexed {self.vector_store.count()} images")

    def add_images_to_index(self, new_image_paths: List[str], batch_size: int = 100):
        """Add new images to the existing index."""
        self.indexer.add_images(new_image_paths)
        self.vector_store.save(self._index_dir)

    def get_similar_images(self, image_path: str, number_of_images: int = 10) -> Dict[int, str]:
        """Return {index: image_path} dict of similar images."""
        return self.searcher.get_similar_images(image_path, number_of_images)

    def plot_similar_images(self, image_path: str, number_of_images: int = 6):
        """Plot query image and similar results."""
        self.searcher.plot_similar_images(image_path, number_of_images)

    def get_image_metadata_file(self):
        """Return index info (replaces old DataFrame metadata)."""
        return {
            "total_images": self.vector_store.count(),
            "model": self.model_name,
            "index_dir": self._index_dir,
        }
