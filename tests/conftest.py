# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Shared fixtures.

Tests must never download model weights — CI runs on four Python versions and
three operating systems. Everything here is deterministic and CPU-only:
`DummyEmbedding` produces fixed vectors so search results are exactly
predictable, and images are tiny solid-colour PNGs generated on the fly.
"""

from typing import List

import numpy as np
import pytest
from PIL import Image

from DeepImageSearch.core.embeddings import BaseEmbedding

DIM = 8


def _unit(vector: List[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    return arr / np.linalg.norm(arr)


class DummyEmbedding(BaseEmbedding):
    """
    Deterministic stand-in for CLIP.

    An image embeds to a unit vector derived from its dominant colour channel;
    a text embeds to a unit vector derived from the hash of its first word.
    Both land in the same 8-D space, so text/image/hybrid paths are exercisable
    without any model download.
    """

    supports_text = True
    dimension = DIM

    def __init__(self, supports_text: bool = True):
        self.supports_text = supports_text
        self.embed_images_calls = 0
        self.embed_texts_calls = 0

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        self.embed_images_calls += 1
        vectors = []
        for img in images:
            r, g, b = img.convert("RGB").resize((1, 1)).getpixel((0, 0))
            base = [r, g, b, 1.0, 0.0, 0.0, 0.0, 0.0]
            vectors.append(_unit(base))
        return np.vstack(vectors).astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self.supports_text:
            raise NotImplementedError("This embedding model does not support text queries")
        self.embed_texts_calls += 1
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 255
            base = [seed, 255 - seed, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            vectors.append(_unit(base))
        return np.vstack(vectors).astype(np.float32)


@pytest.fixture
def embedding():
    return DummyEmbedding()


@pytest.fixture
def image_only_embedding():
    return DummyEmbedding(supports_text=False)


def make_image(path, colour=(255, 0, 0), size=(16, 16)):
    """Write a tiny solid-colour PNG and return its path as a string."""
    Image.new("RGB", size, colour).save(path)
    return str(path)


@pytest.fixture
def image_dir(tmp_path):
    """A folder of three distinctly-coloured images plus a nested one."""
    root = tmp_path / "images"
    root.mkdir()
    make_image(root / "red.png", (255, 0, 0))
    make_image(root / "green.png", (0, 255, 0))
    make_image(root / "blue.jpg", (0, 0, 255))
    nested = root / "nested"
    nested.mkdir()
    make_image(nested / "yellow.png", (255, 255, 0))
    return root


@pytest.fixture
def image_paths(image_dir):
    """Deterministically ordered paths of the top-level images."""
    return sorted(str(p) for p in image_dir.glob("*.???") if p.suffix in {".png", ".jpg"})
