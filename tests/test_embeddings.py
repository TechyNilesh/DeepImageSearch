# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for the embedding layer.

CLIPEmbedding and TimmEmbedding download weights, so they are not instantiated
here — the factory's routing logic is tested by monkeypatching the backends.
"""

import numpy as np
import pytest
from PIL import Image

from DeepImageSearch.core import embeddings as emb_module
from DeepImageSearch.core.embeddings import (
    CLIP_PRESETS,
    BaseEmbedding,
    CustomEmbedding,
    EmbeddingManager,
)


class TestCustomEmbedding:
    def test_normalises_output_to_unit_length(self):
        emb = CustomEmbedding(extract_fn=lambda img: [3.0, 4.0], dimension=2)
        vectors = emb.embed_images([Image.new("RGB", (4, 4))])
        assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-6)
        assert vectors[0].tolist() == pytest.approx([0.6, 0.8])

    def test_zero_vector_is_left_alone_rather_than_dividing_by_zero(self):
        emb = CustomEmbedding(extract_fn=lambda img: [0.0, 0.0], dimension=2)
        vectors = emb.embed_images([Image.new("RGB", (4, 4))])
        assert vectors[0].tolist() == [0.0, 0.0]
        assert not np.isnan(vectors).any()

    def test_flattens_multidimensional_output(self):
        emb = CustomEmbedding(extract_fn=lambda img: np.ones((1, 4)), dimension=4)
        assert emb.embed_images([Image.new("RGB", (4, 4))]).shape == (1, 4)

    def test_returns_float32(self):
        emb = CustomEmbedding(extract_fn=lambda img: np.ones(4, dtype=np.float64), dimension=4)
        assert emb.embed_images([Image.new("RGB", (4, 4))]).dtype == np.float32

    def test_stacks_a_batch(self):
        emb = CustomEmbedding(extract_fn=lambda img: [1.0, 0.0], dimension=2)
        assert emb.embed_images([Image.new("RGB", (4, 4))] * 3).shape == (3, 2)

    def test_extractor_receives_rgb_even_for_greyscale_input(self):
        seen = {}

        def extractor(img):
            seen["mode"] = img.mode
            return [1.0, 0.0]

        CustomEmbedding(extract_fn=extractor, dimension=2).embed_images([Image.new("L", (4, 4))])
        assert seen["mode"] == "RGB"

    def test_does_not_claim_text_support(self):
        emb = CustomEmbedding(extract_fn=lambda img: [1.0], dimension=1)
        assert emb.supports_text is False
        with pytest.raises(NotImplementedError):
            emb.embed_text("a cat")


class TestBaseEmbeddingConvenienceMethods:
    def test_embed_image_returns_a_1d_vector(self, embedding):
        vector = embedding.embed_image(Image.new("RGB", (4, 4), (255, 0, 0)))
        assert vector.ndim == 1
        assert vector.shape == (embedding.dimension,)

    def test_embed_text_returns_a_1d_vector(self, embedding):
        vector = embedding.embed_text("a red square")
        assert vector.ndim == 1
        assert vector.shape == (embedding.dimension,)

    def test_text_embedding_raises_by_default(self):
        class ImageOnly(BaseEmbedding):
            dimension = 2

            def embed_images(self, images):
                return np.zeros((len(images), 2), dtype=np.float32)

        with pytest.raises(NotImplementedError, match="does not support text"):
            ImageOnly().embed_texts(["a cat"])

    def test_base_embedding_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseEmbedding()


class TestPresets:
    def test_list_presets_matches_the_preset_table(self):
        assert EmbeddingManager.list_presets() == CLIP_PRESETS

    def test_list_presets_returns_a_defensive_copy(self):
        presets = EmbeddingManager.list_presets()
        presets["clip-vit-b-32"] = ("tampered", "tampered")
        assert CLIP_PRESETS["clip-vit-b-32"] == ("ViT-B-32", "openai")

    def test_documented_presets_are_present(self):
        # These names appear in the README and Documents/; removing one breaks users.
        for name in ["clip-vit-b-32", "clip-vit-b-16", "clip-vit-l-14", "siglip-vit-b-16"]:
            assert name in CLIP_PRESETS

    def test_every_preset_is_a_model_pretrained_pair(self):
        for name, value in CLIP_PRESETS.items():
            assert name == name.lower()
            assert isinstance(value, tuple) and len(value) == 2
            assert all(isinstance(part, str) and part for part in value)


class TestFactoryRouting:
    """EmbeddingManager.create() picks a backend from the name — verify the routing only."""

    @pytest.fixture
    def spy(self, monkeypatch):
        calls = {}

        class FakeCLIP:
            def __init__(self, **kwargs):
                calls["backend"] = "clip"
                calls["kwargs"] = kwargs

        class FakeTimm:
            def __init__(self, **kwargs):
                calls["backend"] = "timm"
                calls["kwargs"] = kwargs

        monkeypatch.setattr(emb_module, "CLIPEmbedding", FakeCLIP)
        monkeypatch.setattr(emb_module, "TimmEmbedding", FakeTimm)
        return calls

    def test_preset_name_resolves_to_clip_with_preset_weights(self, spy):
        EmbeddingManager.create("clip-vit-b-32")
        assert spy["backend"] == "clip"
        assert spy["kwargs"]["model_name"] == "ViT-B-32"
        assert spy["kwargs"]["pretrained"] == "openai"

    def test_preset_lookup_is_case_and_whitespace_insensitive(self, spy):
        EmbeddingManager.create("  CLIP-ViT-B-32  ")
        assert spy["kwargs"]["model_name"] == "ViT-B-32"

    @pytest.mark.parametrize("name", ["ViT-B-32", "siglip-custom", "eva-something"])
    def test_clip_like_names_route_to_clip(self, spy, name):
        EmbeddingManager.create(name)
        assert spy["backend"] == "clip"
        assert spy["kwargs"]["model_name"] == name

    @pytest.mark.parametrize("name", ["vgg19", "resnet50", "efficientnet_b0"])
    def test_other_names_fall_back_to_timm(self, spy, name):
        EmbeddingManager.create(name)
        assert spy["backend"] == "timm"
        assert spy["kwargs"]["model_name"] == name

    def test_device_and_batch_size_are_forwarded(self, spy):
        EmbeddingManager.create("clip-vit-b-32", device="cpu", batch_size=8)
        assert spy["kwargs"]["device"] == "cpu"
        assert spy["kwargs"]["batch_size"] == 8

    def test_timm_specific_kwargs_are_forwarded(self, spy):
        EmbeddingManager.create("resnet50", image_size=384)
        assert spy["kwargs"]["image_size"] == 384

    def test_default_model_is_a_known_preset(self, spy):
        EmbeddingManager.create()
        assert spy["kwargs"]["model_name"] == CLIP_PRESETS["clip-vit-b-32"][0]
