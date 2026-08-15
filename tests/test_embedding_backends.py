# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for the CLIP and timm embedding backends.

Both normally download weights. Here `open_clip` and `timm` are replaced with
fakes returning fixed tensors, so the code the backends actually own — device
selection, batching, L2 normalisation, dtype, dimension probing — is exercised
without touching the network.
"""

import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from DeepImageSearch.core.embeddings import CLIPEmbedding, TimmEmbedding

CLIP_DIM = 6
TIMM_DIM = 5


class FakeClipModel:
    def __init__(self):
        self.eval_called = False
        self.image_batches = []
        self.text_batches = []

    def eval(self):
        self.eval_called = True
        return self

    def encode_image(self, tensors):
        self.image_batches.append(len(tensors))
        # Rows of ascending magnitude, so normalisation is observable.
        scale = torch.arange(1, len(tensors) + 1, dtype=torch.float32).unsqueeze(1)
        return torch.ones(len(tensors), CLIP_DIM) * scale

    def encode_text(self, tokens):
        self.text_batches.append(len(tokens))
        return torch.ones(len(tokens), CLIP_DIM) * 3.0


@pytest.fixture
def fake_open_clip(monkeypatch):
    state = {"model": FakeClipModel(), "created_with": None, "tokenizer_for": None}

    def create_model_and_transforms(model_name, pretrained=None, device=None):
        state["created_with"] = {"model_name": model_name, "pretrained": pretrained, "device": device}
        return state["model"], None, lambda img: torch.zeros(3, 4, 4)

    def get_tokenizer(model_name):
        state["tokenizer_for"] = model_name
        return lambda batch: torch.zeros(len(batch), 3)

    module = types.ModuleType("open_clip")
    module.create_model_and_transforms = create_model_and_transforms
    module.get_tokenizer = get_tokenizer
    monkeypatch.setitem(sys.modules, "open_clip", module)
    return state


@pytest.fixture
def fake_timm(monkeypatch):
    state = {"created_with": None}

    class FakeChild(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], TIMM_DIM) * 2.0

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = FakeChild()
            self.classifier = torch.nn.Identity()  # dropped by children()[:-1]

    def create_model(model_name, pretrained=True):
        state["created_with"] = {"model_name": model_name, "pretrained": pretrained}
        return FakeBackbone()

    module = types.ModuleType("timm")
    module.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", module)
    return state


def _ignore_device_placement(monkeypatch):
    """Let the backends "allocate" on cuda/mps on a machine that has neither."""
    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: real_zeros(*a, **{**kw, "device": "cpu"}))


def images(count=1):
    return [Image.new("RGB", (8, 8), (255, 0, 0)) for _ in range(count)]


class TestClipEmbedding:
    def test_probes_its_dimension_at_load_time(self, fake_open_clip):
        assert CLIPEmbedding(device="cpu").dimension == CLIP_DIM

    def test_puts_the_model_in_eval_mode(self, fake_open_clip):
        CLIPEmbedding(device="cpu")
        assert fake_open_clip["model"].eval_called is True

    def test_forwards_model_name_and_weights_tag(self, fake_open_clip):
        CLIPEmbedding(model_name="ViT-L-14", pretrained="laion2b", device="cpu")
        assert fake_open_clip["created_with"]["model_name"] == "ViT-L-14"
        assert fake_open_clip["created_with"]["pretrained"] == "laion2b"
        assert fake_open_clip["tokenizer_for"] == "ViT-L-14"

    def test_declares_text_support(self, fake_open_clip):
        assert CLIPEmbedding(device="cpu").supports_text is True

    def test_image_embeddings_are_unit_length_float32(self, fake_open_clip):
        vectors = CLIPEmbedding(device="cpu").embed_images(images(3))
        assert vectors.shape == (3, CLIP_DIM)
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    def test_text_embeddings_are_unit_length_float32(self, fake_open_clip):
        vectors = CLIPEmbedding(device="cpu").embed_texts(["a cat", "a dog"])
        assert vectors.shape == (2, CLIP_DIM)
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    def test_images_are_processed_in_batches(self, fake_open_clip):
        embedding = CLIPEmbedding(device="cpu", batch_size=2)
        probe_calls = len(fake_open_clip["model"].image_batches)  # dimension probe
        embedding.embed_images(images(5))
        assert fake_open_clip["model"].image_batches[probe_calls:] == [2, 2, 1]

    def test_texts_are_processed_in_batches(self, fake_open_clip):
        CLIPEmbedding(device="cpu", batch_size=2).embed_texts(["a", "b", "c"])
        assert fake_open_clip["model"].text_batches == [2, 1]

    def test_batching_does_not_change_the_result(self, fake_open_clip):
        one_shot = CLIPEmbedding(device="cpu", batch_size=64).embed_images(images(4))
        batched = CLIPEmbedding(device="cpu", batch_size=2).embed_images(images(4))
        assert np.allclose(one_shot, batched)

    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
    )
    def test_device_auto_detection(self, fake_open_clip, monkeypatch, cuda, mps, expected):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        _ignore_device_placement(monkeypatch)
        # Only the device *decision* is under test — no tensor really moves.
        assert CLIPEmbedding().device == expected

    def test_explicit_device_wins_over_detection(self, fake_open_clip, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert CLIPEmbedding(device="cpu").device == "cpu"


class TestTimmEmbedding:
    def test_probes_its_dimension_at_load_time(self, fake_timm):
        assert TimmEmbedding(device="cpu").dimension == TIMM_DIM

    def test_does_not_claim_text_support(self, fake_timm):
        embedding = TimmEmbedding(device="cpu")
        assert embedding.supports_text is False
        with pytest.raises(NotImplementedError):
            embedding.embed_texts(["a cat"])

    def test_forwards_model_name_and_pretrained_flag(self, fake_timm):
        TimmEmbedding(model_name="resnet50", pretrained=False, device="cpu")
        assert fake_timm["created_with"] == {"model_name": "resnet50", "pretrained": False}

    def test_drops_the_classifier_head(self, fake_timm):
        # children()[:-1] must leave the feature body only.
        assert len(list(TimmEmbedding(device="cpu").model.children())) == 1

    def test_embeddings_are_unit_length_float32(self, fake_timm):
        vectors = TimmEmbedding(device="cpu").embed_images(images(2))
        assert vectors.shape == (2, TIMM_DIM)
        assert vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    def test_images_are_processed_in_batches(self, fake_timm):
        vectors = TimmEmbedding(device="cpu", batch_size=2).embed_images(images(5))
        assert vectors.shape == (5, TIMM_DIM)

    def test_image_size_is_honoured(self, fake_timm):
        assert TimmEmbedding(device="cpu", image_size=384).image_size == 384

    @pytest.mark.parametrize(
        ("cuda", "mps", "expected"),
        [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
    )
    def test_device_auto_detection(self, fake_timm, monkeypatch, cuda, mps, expected):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        monkeypatch.setattr(torch.nn.Module, "to", lambda self, device: self)
        _ignore_device_placement(monkeypatch)
        assert TimmEmbedding().device == expected
