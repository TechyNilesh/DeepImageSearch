# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for the LLM captioner.

No network access: a fake `openai` module is installed in sys.modules so the
OpenAI-SDK-shaped call chain (client.chat.completions.create) is exercised
against canned responses.
"""

import base64
import io
import sys
import types

import pytest
from PIL import Image

from DeepImageSearch.core.captioner import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_METADATA_PROMPT,
    Captioner,
    _image_to_base64,
)
from tests.conftest import make_image


class FakeCompletions:
    def __init__(self, responses, errors=None):
        self.responses = list(responses)
        self.errors = errors or {}
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index in self.errors:
            raise self.errors[index]
        content = self.responses[index % len(self.responses)]
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


@pytest.fixture
def fake_openai(monkeypatch):
    """Install a fake `openai` module; returns a factory for building captioners."""
    state = {}

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            state["api_key"] = api_key
            state["base_url"] = base_url
            self.chat = types.SimpleNamespace(completions=state["completions"])

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)

    def build(responses=("a caption",), errors=None, **kwargs):
        state["completions"] = FakeCompletions(responses, errors)
        captioner = Captioner(model="vision-model", api_key="secret",
                              base_url="https://example.invalid/v1", **kwargs)
        return captioner, state

    return build


class TestImageEncoding:
    def test_returns_decodable_base64_jpeg(self, tmp_path):
        path = make_image(tmp_path / "a.png", (10, 20, 30))
        encoded = _image_to_base64(path)
        with Image.open(io.BytesIO(base64.standard_b64decode(encoded))) as img:
            assert img.format == "JPEG"

    def test_large_images_are_downscaled(self, tmp_path):
        path = make_image(tmp_path / "big.png", size=(2048, 1024))
        encoded = _image_to_base64(path, max_size=256)
        with Image.open(io.BytesIO(base64.standard_b64decode(encoded))) as img:
            assert max(img.size) == 256
            assert img.size == (256, 128)  # aspect ratio preserved

    def test_small_images_are_left_alone(self, tmp_path):
        path = make_image(tmp_path / "small.png", size=(64, 32))
        encoded = _image_to_base64(path, max_size=1024)
        with Image.open(io.BytesIO(base64.standard_b64decode(encoded))) as img:
            assert img.size == (64, 32)

    def test_greyscale_and_rgba_inputs_are_converted(self, tmp_path):
        for mode in ("L", "RGBA"):
            path = tmp_path / f"{mode}.png"
            Image.new(mode, (32, 32)).save(path)
            encoded = _image_to_base64(str(path))
            with Image.open(io.BytesIO(base64.standard_b64decode(encoded))) as img:
                assert img.mode == "RGB"


class TestCaption:
    def test_returns_the_model_response(self, fake_openai, tmp_path):
        captioner, _ = fake_openai(responses=["a red square on white"])
        assert captioner.caption(make_image(tmp_path / "a.png")) == "a red square on white"

    def test_sends_model_and_token_limit(self, fake_openai, tmp_path):
        captioner, state = fake_openai(max_tokens=123)
        captioner.caption(make_image(tmp_path / "a.png"))
        call = state["completions"].calls[0]
        assert call["model"] == "vision-model"
        assert call["max_tokens"] == 123

    def test_sends_the_image_as_a_data_url_and_the_default_prompt(self, fake_openai, tmp_path):
        captioner, state = fake_openai()
        captioner.caption(make_image(tmp_path / "a.png"))
        content = state["completions"].calls[0]["messages"][0]["content"]
        image_part = next(p for p in content if p["type"] == "image_url")
        text_part = next(p for p in content if p["type"] == "text")
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert text_part["text"] == DEFAULT_CAPTION_PROMPT

    def test_custom_prompt_overrides_the_default(self, fake_openai, tmp_path):
        captioner, state = fake_openai()
        captioner.caption(make_image(tmp_path / "a.png"), prompt="just the colours")
        content = state["completions"].calls[0]["messages"][0]["content"]
        assert next(p for p in content if p["type"] == "text")["text"] == "just the colours"

    def test_credentials_are_passed_to_the_client(self, fake_openai, tmp_path):
        _, state = fake_openai()
        assert state["api_key"] == "secret"
        assert state["base_url"] == "https://example.invalid/v1"


class TestCaptionBatch:
    def test_maps_every_path_to_a_caption(self, fake_openai, tmp_path):
        captioner, _ = fake_openai(responses=["one", "two"])
        paths = [make_image(tmp_path / "a.png"), make_image(tmp_path / "b.png")]
        assert captioner.caption_batch(paths) == {paths[0]: "one", paths[1]: "two"}

    def test_skips_failures_by_default(self, fake_openai, tmp_path):
        captioner, _ = fake_openai(responses=["ok", "ok"], errors={0: RuntimeError("boom")})
        paths = [make_image(tmp_path / "a.png"), make_image(tmp_path / "b.png")]
        result = captioner.caption_batch(paths)
        assert result[paths[0]] == ""
        assert result[paths[1]] == "ok"

    def test_raises_when_asked_to(self, fake_openai, tmp_path):
        captioner, _ = fake_openai(responses=["ok"], errors={0: RuntimeError("boom")})
        with pytest.raises(RuntimeError, match="boom"):
            captioner.caption_batch([make_image(tmp_path / "a.png")], on_error="raise")

    def test_empty_input_makes_no_calls(self, fake_openai):
        captioner, state = fake_openai()
        assert captioner.caption_batch([]) == {}
        assert state["completions"].calls == []


class TestExtractMetadata:
    def test_parses_a_json_response(self, fake_openai, tmp_path):
        captioner, _ = fake_openai(responses=['{"caption": "a cat", "tags": ["pet"]}'])
        assert captioner.extract_metadata(make_image(tmp_path / "a.png")) == {
            "caption": "a cat", "tags": ["pet"],
        }

    def test_strips_a_markdown_code_fence(self, fake_openai, tmp_path):
        captioner, _ = fake_openai(responses=['```json\n{"caption": "a cat"}\n```'])
        assert captioner.extract_metadata(make_image(tmp_path / "a.png")) == {"caption": "a cat"}

    def test_falls_back_to_raw_text_on_invalid_json(self, fake_openai, tmp_path):
        captioner, _ = fake_openai(responses=["not json at all"])
        result = captioner.extract_metadata(make_image(tmp_path / "a.png"))
        assert result == {"caption": "not json at all", "raw_response": True}

    def test_uses_the_metadata_prompt(self, fake_openai, tmp_path):
        captioner, state = fake_openai(responses=["{}"])
        captioner.extract_metadata(make_image(tmp_path / "a.png"))
        content = state["completions"].calls[0]["messages"][0]["content"]
        assert next(p for p in content if p["type"] == "text")["text"] == DEFAULT_METADATA_PROMPT


def test_missing_openai_dependency_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # forces ImportError on `from openai import ...`
    with pytest.raises(ImportError, match=r"DeepImageSearch\[llm\]"):
        Captioner(model="m", api_key="k", base_url="u")
