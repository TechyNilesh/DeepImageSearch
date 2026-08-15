# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for the v2-compatible Search_Setup shim.

This is the API v2 users still call, so its surface is pinned here: constructor
validation, index-on-disk layout, and the {index: path} return shape.
"""

import os

import pytest

from DeepImageSearch import DeepImageSearch as v2_module
from DeepImageSearch.DeepImageSearch import Search_Setup
from tests.conftest import DummyEmbedding, make_image


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    monkeypatch.setattr(
        v2_module.EmbeddingManager, "create",
        staticmethod(lambda *args, **kwargs: DummyEmbedding()),
    )


@pytest.fixture
def setup(tmp_path, image_paths):
    return Search_Setup(image_list=image_paths, metadata_dir=str(tmp_path / "metadata-files"))


class TestConstruction:
    def test_empty_image_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be empty"):
            Search_Setup(image_list=[], metadata_dir=str(tmp_path))

    def test_non_list_raises(self, tmp_path, image_paths):
        with pytest.raises(TypeError, match="must be a list"):
            Search_Setup(image_list=image_paths[0], metadata_dir=str(tmp_path))

    def test_image_count_truncates_the_list(self, tmp_path, image_paths):
        setup = Search_Setup(image_list=image_paths, image_count=2, metadata_dir=str(tmp_path))
        assert setup.image_list == image_paths[:2]

    def test_image_count_none_keeps_everything(self, setup, image_paths):
        assert setup.image_list == image_paths

    def test_index_dir_is_namespaced_by_model(self, tmp_path, image_paths):
        metadata_dir = tmp_path / "metadata-files"
        setup = Search_Setup(image_list=image_paths, model_name="vgg19", metadata_dir=str(metadata_dir))
        assert setup._index_dir == os.path.join(str(metadata_dir), "vgg19")
        assert os.path.isdir(setup._index_dir)


class TestIndexing:
    def test_run_index_writes_the_index(self, setup, image_paths):
        setup.run_index()
        assert setup.vector_store.count() == len(image_paths)
        assert os.path.exists(os.path.join(setup._index_dir, "index.faiss"))

    def test_run_index_is_skipped_when_an_index_exists(self, setup, image_paths):
        setup.run_index()
        setup.run_index()  # must not double up
        assert setup.vector_store.count() == len(image_paths)

    def test_force_reindex_rebuilds_from_scratch(self, setup, image_paths):
        setup.run_index()
        setup.run_index(force_reindex=True)
        assert setup.vector_store.count() == len(image_paths)

    def test_force_reindex_rewires_indexer_and_searcher(self, setup):
        setup.run_index()
        setup.run_index(force_reindex=True)
        # A stale Searcher pointing at the discarded store would silently
        # return results from the old index.
        assert setup.searcher.vector_store is setup.vector_store
        assert setup.indexer.vector_store is setup.vector_store

    def test_add_images_to_index_extends_and_persists(self, setup, tmp_path, image_paths):
        setup.run_index()
        later = make_image(tmp_path / "later.png", (9, 9, 9))
        setup.add_images_to_index([later])
        assert setup.vector_store.count() == len(image_paths) + 1

    def test_an_existing_index_is_reloaded_on_construction(self, tmp_path, image_paths):
        metadata_dir = str(tmp_path / "metadata-files")
        Search_Setup(image_list=image_paths, metadata_dir=metadata_dir).run_index()

        reopened = Search_Setup(image_list=image_paths, metadata_dir=metadata_dir)
        assert reopened.vector_store.count() == len(image_paths)


class TestSearch:
    def test_get_similar_images_returns_index_to_path(self, setup, image_paths):
        setup.run_index()
        similar = setup.get_similar_images(image_paths[0], number_of_images=2)
        assert list(similar) == [0, 1]
        assert similar[0] == image_paths[0]

    def test_plot_similar_images_delegates(self, setup, image_paths, monkeypatch):
        setup.run_index()
        calls = []
        monkeypatch.setattr(setup.searcher, "plot_similar_images", lambda p, n: calls.append((p, n)))
        setup.plot_similar_images(image_paths[0], number_of_images=4)
        assert calls == [(image_paths[0], 4)]

    def test_metadata_file_summarises_the_index(self, setup, image_paths):
        setup.run_index()
        meta = setup.get_image_metadata_file()
        assert meta["total_images"] == len(image_paths)
        assert meta["model"] == "vgg19"
        assert meta["index_dir"] == setup._index_dir


def test_module_still_exports_the_v2_names():
    assert set(v2_module.__all__) == {"Load_Data", "Search_Setup"}
    # `from DeepImageSearch.DeepImageSearch import Load_Data` is the v2 import path.
    for name in v2_module.__all__:
        assert getattr(v2_module, name, None) is not None
