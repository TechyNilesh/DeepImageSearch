# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for the high-level SearchEngine facade.

SearchEngine builds a real embedding backend in __init__, so every test here
patches EmbeddingManager.create to hand back the deterministic DummyEmbedding.
"""

import numpy as np
import pytest

from DeepImageSearch import search_engine as se_module
from DeepImageSearch.metadatastore.json_store import JsonMetadataStore
from DeepImageSearch.search_engine import SearchEngine
from DeepImageSearch.vectorstores.base import BaseVectorStore
from DeepImageSearch.vectorstores.faiss_store import FAISSStore
from tests.conftest import DIM, DummyEmbedding, make_image


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    """Every SearchEngine in this module gets the dummy embedding."""
    monkeypatch.setattr(
        se_module.EmbeddingManager, "create",
        staticmethod(lambda *args, **kwargs: DummyEmbedding()),
    )


@pytest.fixture
def engine(tmp_path):
    return SearchEngine(index_dir=str(tmp_path / "index"))


class TestConstruction:
    def test_defaults_to_faiss_and_json_stores(self, engine):
        assert isinstance(engine.vector_store, FAISSStore)
        assert isinstance(engine.metadata_store, JsonMetadataStore)
        assert engine.count == 0

    def test_vector_store_dimension_follows_the_embedding(self, engine):
        assert engine.vector_store.dimension == DIM

    def test_accepts_an_injected_vector_store(self, tmp_path):
        store = FAISSStore(dimension=DIM, index_type="hnsw")
        engine = SearchEngine(vector_store=store, index_dir=str(tmp_path))
        assert engine.vector_store is store

    def test_accepts_an_injected_metadata_store(self, tmp_path):
        records = JsonMetadataStore()
        engine = SearchEngine(metadata_store=records, index_dir=str(tmp_path))
        assert engine.metadata_store is records

    def test_unknown_vector_store_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown vector_store"):
            SearchEngine(vector_store="pinecone", index_dir=str(tmp_path))

    def test_reopening_an_index_dir_loads_the_existing_index(self, tmp_path, image_paths):
        index_dir = str(tmp_path / "index")
        first = SearchEngine(index_dir=index_dir)
        first.index(image_paths)

        reopened = SearchEngine(index_dir=index_dir)
        assert reopened.count == len(image_paths)
        assert len(reopened.get_records()) == len(image_paths)

    def test_no_captioner_without_full_credentials(self, tmp_path):
        assert SearchEngine(index_dir=str(tmp_path)).captioner is None

    def test_partial_captioner_credentials_are_ignored(self, tmp_path):
        engine = SearchEngine(index_dir=str(tmp_path), captioner_model="m", captioner_api_key="k")
        assert engine.captioner is None

    def test_captioner_built_when_all_credentials_present(self, tmp_path, monkeypatch):
        built = {}

        class FakeCaptioner:
            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr(se_module, "Captioner", FakeCaptioner)
        engine = SearchEngine(
            index_dir=str(tmp_path),
            captioner_model="vision-model",
            captioner_api_key="secret",
            captioner_base_url="https://example.invalid/v1",
        )
        assert isinstance(engine.captioner, FakeCaptioner)
        assert built == {
            "model": "vision-model",
            "api_key": "secret",
            "base_url": "https://example.invalid/v1",
        }
        assert engine.indexer.captioner is engine.captioner


class TestPathResolution:
    def test_a_folder_is_expanded_to_its_images(self, engine, image_dir):
        resolved = engine._resolve_image_paths(str(image_dir))
        assert len(resolved) == 4

    def test_a_single_path_is_wrapped_in_a_list(self, engine, image_paths):
        assert engine._resolve_image_paths(image_paths[0]) == [image_paths[0]]

    def test_a_list_is_passed_through(self, engine, image_paths):
        assert engine._resolve_image_paths(image_paths) == image_paths


class TestIndexing:
    def test_index_accepts_a_folder(self, engine, image_dir):
        assert engine.index(str(image_dir)) == 4
        assert engine.count == 4

    def test_index_accepts_a_list(self, engine, image_paths):
        assert engine.index(image_paths) == len(image_paths)

    def test_index_persists_by_default(self, tmp_path, image_paths):
        index_dir = tmp_path / "index"
        SearchEngine(index_dir=str(index_dir)).index(image_paths)
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "image_records.json").exists()

    def test_save_can_be_deferred(self, tmp_path, image_paths):
        index_dir = tmp_path / "index"
        SearchEngine(index_dir=str(index_dir)).index(image_paths, save=False)
        assert not (index_dir / "index.faiss").exists()

    def test_metadata_is_attached_to_records(self, engine, image_paths):
        engine.index(image_paths, metadata=[{"album": "trip"} for _ in image_paths])
        assert all(r["extra"] == {"album": "trip"} for r in engine.get_records())

    def test_add_images_extends_the_index(self, engine, image_paths, tmp_path):
        engine.index(image_paths)
        later = make_image(tmp_path / "later.png", (7, 7, 7))
        assert engine.add_images([later]) == 1
        assert engine.count == len(image_paths) + 1

    def test_captions_are_requested_when_asked(self, tmp_path, image_paths, monkeypatch):
        class FakeCaptioner:
            def __init__(self, **kwargs):
                self.seen = None

            def caption_batch(self, paths, prompt=None):
                self.seen = (list(paths), prompt)
                return {p: "a caption" for p in paths}

        monkeypatch.setattr(se_module, "Captioner", FakeCaptioner)
        engine = SearchEngine(
            index_dir=str(tmp_path),
            captioner_model="m", captioner_api_key="k", captioner_base_url="u",
        )
        engine.index(image_paths, generate_captions=True, caption_prompt="describe it")

        assert engine.captioner.seen == (image_paths, "describe it")
        assert all(r["caption"] == "a caption" for r in engine.get_records())


class TestSearch:
    def test_search_by_image_finds_the_query(self, engine, image_paths):
        engine.index(image_paths)
        assert engine.search_by_image(image_paths[0], k=1)[0]["metadata"]["image_path"] == image_paths[0]

    def test_search_by_text_returns_results(self, engine, image_paths):
        engine.index(image_paths)
        assert len(engine.search_by_text("a red square", k=2)) == 2

    def test_search_dispatches_by_query_type(self, engine, image_paths):
        engine.index(image_paths)
        assert engine.search(image_paths[0], k=1)[0]["metadata"]["image_path"] == image_paths[0]
        assert len(engine.search("a red square", k=2)) == 2
        assert len(engine.search(np.ones(DIM, dtype=np.float32), k=2)) == 2

    def test_hybrid_search_through_the_facade(self, engine, image_paths):
        engine.index(image_paths)
        results = engine.search("a red square", k=2, mode="hybrid", image_query=image_paths[0])
        assert len(results) == 2

    def test_filters_are_forwarded(self, engine, image_paths):
        tags = [{"tag": "keep"}] + [{"tag": "drop"}] * (len(image_paths) - 1)
        engine.index(image_paths, metadata=tags)
        assert len(engine.search_by_image(image_paths[0], k=5, filters={"tag": "keep"})) == 1

    def test_get_similar_images_v2_shape(self, engine, image_paths):
        engine.index(image_paths)
        similar = engine.get_similar_images(image_paths[0], number_of_images=2)
        assert list(similar) == [0, 1]
        assert similar[0] == image_paths[0]

    def test_plot_similar_images_delegates_to_the_searcher(self, engine, image_paths, monkeypatch):
        engine.index(image_paths)
        calls = []
        monkeypatch.setattr(engine.searcher, "plot_similar_images",
                            lambda path, n: calls.append((path, n)))
        engine.plot_similar_images(image_paths[0], number_of_images=3)
        assert calls == [(image_paths[0], 3)]


class TestRecords:
    def test_get_records_returns_dicts(self, engine, image_paths):
        engine.index(image_paths)
        records = engine.get_records()
        assert len(records) == len(image_paths)
        assert set(records[0]) >= {"image_id", "image_index", "image_name", "image_path"}

    def test_get_record_by_id(self, engine, image_paths):
        engine.index(image_paths)
        image_id = engine.get_records()[0]["image_id"]
        assert engine.get_record(image_id)["image_id"] == image_id

    def test_get_record_returns_none_for_unknown_id(self, engine, image_paths):
        engine.index(image_paths)
        assert engine.get_record("no-such-id") is None


class TestPersistence:
    def test_explicit_save_then_load_round_trip(self, tmp_path, image_paths):
        index_dir = str(tmp_path / "index")
        engine = SearchEngine(index_dir=index_dir)
        engine.index(image_paths, save=False)
        engine.save()

        fresh = SearchEngine(index_dir=index_dir)
        fresh.load()
        assert fresh.count == len(image_paths)
        assert fresh.search_by_image(image_paths[0], k=1)[0]["metadata"]["image_path"] == image_paths[0]


class TestIntrospection:
    def test_info_reports_the_configuration(self, engine, image_paths):
        engine.index(image_paths)
        info = engine.info()
        assert info["indexed_images"] == len(image_paths)
        assert info["dimension"] == DIM
        assert info["vector_store"] == "FAISSStore"
        assert info["metadata_store"] == "JsonMetadataStore"
        assert info["supports_text_search"] is True

    def test_supports_text_search_follows_the_backend(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            se_module.EmbeddingManager, "create",
            staticmethod(lambda *a, **k: DummyEmbedding(supports_text=False)),
        )
        assert SearchEngine(index_dir=str(tmp_path)).supports_text_search is False

    def test_repr_mentions_model_and_count(self, engine, image_paths):
        engine.index(image_paths)
        text = repr(engine)
        assert "SearchEngine(" in text
        assert f"images={len(image_paths)}" in text


def test_custom_store_subclass_is_accepted_without_a_name_lookup(tmp_path):
    """A user-supplied BaseVectorStore must bypass _create_store entirely."""

    class RecordingStore(BaseVectorStore):
        def __init__(self):
            self.added = []

        def add(self, ids, vectors, metadata=None):
            self.added.append(ids)

        def search(self, query_vector, k=10, filters=None):
            return []

        def delete(self, ids):
            pass

        def count(self):
            return len(self.added)

        def save(self, path):
            pass

        def load(self, path):
            pass

    store = RecordingStore()
    engine = SearchEngine(vector_store=store, index_dir=str(tmp_path))
    engine.index([], save=False)
    assert engine.vector_store is store
