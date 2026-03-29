"""
Embedding backends for DeepImageSearch.

Supports:
- CLIP / SigLIP / EVA-CLIP via open_clip  (text + image)
- timm CNN/ViT models                     (image only, legacy)
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    """Abstract embedding backend."""

    supports_text: bool = False
    dimension: int = 0

    @abstractmethod
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Return (N, D) float32 array of normalised image embeddings."""

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) float32 array of normalised text embeddings."""
        raise NotImplementedError("This embedding model does not support text queries")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Convenience: embed a single image and return 1-D vector."""
        return self.embed_images([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Convenience: embed a single text and return 1-D vector."""
        return self.embed_texts([text])[0]


class CLIPEmbedding(BaseEmbedding):
    """
    CLIP / SigLIP / EVA-CLIP embedding via open_clip.

    Supports both image and text embedding in a shared space.

    Parameters
    ----------
    model_name : str
        open_clip model name (e.g. 'ViT-B-32', 'ViT-L-14', 'ViT-bigG-14').
    pretrained : str
        Pretrained weights tag (e.g. 'laion2b_s34b_b79k', 'openai').
    device : str or None
        'cuda', 'mps', 'cpu', or None for auto-detect.
    batch_size : int
        Batch size for embedding extraction.
    """

    supports_text = True

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        import open_clip

        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading CLIP model: {model_name} (pretrained={pretrained}) on {device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            self.dimension = self.model.encode_image(dummy).shape[-1]

        logger.info(f"CLIP model loaded — dimension={self.dimension}")

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
            features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features).astype(np.float32)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        all_features = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features).astype(np.float32)


class TimmEmbedding(BaseEmbedding):
    """
    Legacy timm-based image embedding (image-only).

    Parameters
    ----------
    model_name : str
        timm model name (e.g. 'vgg19', 'resnet50', 'vit_base_patch16_224').
    pretrained : bool
        Use pretrained weights.
    image_size : int
        Input image resolution.
    device : str or None
        Device for inference.
    batch_size : int
        Batch size for embedding extraction.
    """

    supports_text = False

    def __init__(
        self,
        model_name: str = "vgg19",
        pretrained: bool = True,
        image_size: int = 224,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        import timm
        from torchvision import transforms

        self.image_size = image_size
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        logger.info(f"Loading timm model: {model_name} on {device}")
        base_model = timm.create_model(model_name, pretrained=pretrained)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        self.model.to(device)

        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=device)
            self.dimension = self.model(dummy).flatten(1).shape[-1]

        logger.info(f"timm model loaded — dimension={self.dimension}")

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
            features = self.model(tensors).flatten(1)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features).astype(np.float32)


class CustomEmbedding(BaseEmbedding):
    """
    Custom embedding using any user-provided feature extractor.

    Accepts a callable that takes a PIL Image and returns a 1-D numpy array
    (or list/tensor). Works with ONNX, TorchScript, TensorFlow, sklearn,
    or any custom model.

    Parameters
    ----------
    extract_fn : callable
        Function that takes a PIL.Image and returns a feature vector
        (numpy array, list, or tensor). Must return the same dimension
        for every image.
    dimension : int
        Output dimension of the feature vector.
    name : str
        Name for this embedding (used in logging/metadata).

    Examples
    --------
    # ONNX model
    import onnxruntime as ort
    session = ort.InferenceSession("model.onnx")

    def my_extractor(img):
        arr = np.array(img.resize((224, 224))).astype(np.float32)
        arr = arr.transpose(2, 0, 1)[None] / 255.0
        return session.run(None, {"input": arr})[0].flatten()

    emb = CustomEmbedding(extract_fn=my_extractor, dimension=512)
    engine = SearchEngine(model_name="clip-vit-b-32")  # ignored
    engine.embedding = emb  # override

    # Or use directly with core modules
    from DeepImageSearch.core.indexer import Indexer
    from DeepImageSearch.vectorstores.faiss_store import FAISSStore

    store = FAISSStore(dimension=512)
    indexer = Indexer(embedding=emb, vector_store=store)
    indexer.index(image_paths)
    """

    supports_text = False

    def __init__(self, extract_fn: callable, dimension: int, name: str = "custom"):
        self.extract_fn = extract_fn
        self.dimension = dimension
        self.name = name
        logger.info(f"CustomEmbedding loaded — name={name}, dimension={dimension}")

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        features = []
        for img in images:
            feat = self.extract_fn(img.convert("RGB"))
            feat = np.asarray(feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            features.append(feat)
        return np.vstack(features).astype(np.float32)


# ── Factory ────────────────────────────────────────────────────────────────────

# Common presets mapping friendly names to open_clip (model, pretrained) tuples
CLIP_PRESETS = {
    "clip-vit-b-32": ("ViT-B-32", "openai"),
    "clip-vit-b-16": ("ViT-B-16", "openai"),
    "clip-vit-l-14": ("ViT-L-14", "openai"),
    "clip-vit-l-14-336": ("ViT-L-14-336", "openai"),
    "clip-vit-bigg-14": ("ViT-bigG-14", "laion2b_s39b_b160k"),
    "eva-clip-vit-b-16": ("EVA02-B-16", "merged2b_s8b_b131k"),
    "siglip-vit-b-16": ("ViT-B-16-SigLIP", "webli"),
    "siglip-vit-l-16": ("ViT-L-16-SigLIP-256", "webli"),
}


class EmbeddingManager:
    """
    Factory that creates the right embedding backend from a simple name.

    Usage
    -----
    >>> emb = EmbeddingManager.create("clip-vit-b-32")          # CLIP
    >>> emb = EmbeddingManager.create("vgg19")                  # timm legacy
    >>> emb = EmbeddingManager.create("clip-vit-l-14", device="cuda")
    """

    @staticmethod
    def create(
        model_name: str = "clip-vit-b-32",
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> BaseEmbedding:
        """
        Create an embedding backend.

        Parameters
        ----------
        model_name : str
            A preset name (e.g. 'clip-vit-b-32') or a raw timm model name.
        device : str or None
            Target device.
        batch_size : int
            Batch size for extraction.

        Returns
        -------
        BaseEmbedding
        """
        name_lower = model_name.lower().strip()

        # Check CLIP presets
        if name_lower in CLIP_PRESETS:
            clip_model, clip_pretrained = CLIP_PRESETS[name_lower]
            return CLIPEmbedding(
                model_name=clip_model,
                pretrained=clip_pretrained,
                device=device,
                batch_size=batch_size,
            )

        # Check if it looks like an open_clip model name (contains 'ViT' or 'clip' or 'siglip')
        if any(k in name_lower for k in ("vit-", "clip", "siglip", "eva")):
            pretrained = kwargs.get("pretrained", "openai")
            return CLIPEmbedding(
                model_name=model_name,
                pretrained=pretrained,
                device=device,
                batch_size=batch_size,
            )

        # Fall back to timm
        return TimmEmbedding(
            model_name=model_name,
            pretrained=kwargs.get("pretrained", True),
            image_size=kwargs.get("image_size", 224),
            device=device,
            batch_size=batch_size,
        )

    @staticmethod
    def list_presets() -> dict:
        """Return available CLIP preset names and their model details."""
        return dict(CLIP_PRESETS)
