# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
The remaining branches: matplotlib plotting, FAISS IVF/HNSW index types, and
the store-type routing shared by SearchEngine and ImageSearchTool.

Plotting runs on the non-interactive Agg backend with plt.show() stubbed, so
nothing opens a window in CI.
"""

import numpy as np
import pytest

import matplotlib  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from DeepImageSearch import search_engine as se_module  # noqa: E402
from DeepImageSearch.agents import tool_interface as ti_module  # noqa: E402
from DeepImageSearch.agents.tool_interface import ImageSearchTool  # noqa: E402
from DeepImageSearch.core.indexer import Indexer  # noqa: E402
from DeepImageSearch.core.searcher import Searcher  # noqa: E402
from DeepImageSearch.search_engine import SearchEngine  # noqa: E402
from DeepImageSearch.vectorstores.faiss_store import FAISSStore  # noqa: E402
from tests.conftest import DIM, DummyEmbedding  # noqa: E402


@pytest.fixture
def headless(monkeypatch):
    """Count draw calls without opening a window."""
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    yield shown
    plt.close("all")


@pytest.fixture
def searcher(embedding, image_paths):
    store = FAISSStore(dimension=DIM)
    Indexer(embedding=embedding, vector_store=store).index(image_paths)
    return Searcher(embedding=embedding, vector_store=store)


class TestPlotting:
    def test_plots_the_query_and_the_results(self, searcher, image_paths, headless):
        searcher.plot_similar_images(image_paths[0], number_of_images=3)
        assert len(headless) == 2  # query figure, then the results grid

    def test_survives_a_result_whose_file_has_vanished(self, embedding, image_paths, tmp_path, headless):
        from tests.conftest import make_image

        doomed = make_image(tmp_path / "doomed.png", (3, 3, 3))
        store = FAISSStore(dimension=DIM)
        Indexer(embedding=embedding, vector_store=store).index(image_paths + [doomed])
        import os

        os.remove(doomed)

        # A deleted image must not take the whole plot down.
        Searcher(embedding, store).plot_similar_images(image_paths[0], number_of_images=4)
        assert len(headless) == 2

    def test_handles_results_without_a_stored_path(self, embedding, image_paths, headless):
        store = FAISSStore(dimension=DIM)
        Indexer(embedding=embedding, vector_store=store).index(image_paths)
        store._metadata = [{} for _ in store._metadata]  # metadata stripped
        Searcher(embedding, store).plot_similar_images(image_paths[0], number_of_images=2)
        assert len(headless) == 2


class TestFaissIndexTypes:
    def test_ivf_index_trains_before_adding(self):
        store = FAISSStore(dimension=DIM, index_type="ivf")
        assert store.index.is_trained is False

        rng = np.random.default_rng(0)
        vectors = rng.random((256, DIM)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store.add([str(i) for i in range(256)], vectors)

        assert store.index.is_trained is True
        assert store.count() == 256

    def test_ivf_search_sets_nprobe_and_returns_results(self):
        store = FAISSStore(dimension=DIM, index_type="ivf")
        rng = np.random.default_rng(1)
        vectors = rng.random((256, DIM)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store.add([str(i) for i in range(256)], vectors)

        results = store.search(vectors[0], k=5)
        assert results
        assert store.index.nprobe == min(20, store.index.nlist)

    def test_hnsw_index_round_trips(self, tmp_path):
        store = FAISSStore(dimension=DIM, index_type="hnsw")
        vectors = np.eye(DIM, dtype=np.float32)[:3]
        store.add(["a", "b", "c"], vectors, [{"n": i} for i in range(3)])
        assert store.search(vectors[0], k=1)[0]["id"] == "a"

        store.save(str(tmp_path))
        reloaded = FAISSStore(dimension=DIM)
        reloaded.load(str(tmp_path))
        assert reloaded.index_type == "hnsw"
        assert reloaded.count() == 3


class TestStoreRouting:
    """SearchEngine and ImageSearchTool both map a backend name to a store class."""

    @pytest.fixture(autouse=True)
    def no_model_downloads(self, monkeypatch):
        for module in (se_module, ti_module):
            monkeypatch.setattr(
                module.EmbeddingManager, "create",
                staticmethod(lambda *a, **k: DummyEmbedding()),
            )

    def test_engine_builds_a_chroma_store(self, tmp_path):
        pytest.importorskip("chromadb", reason="install the [chroma] extra")
        engine = SearchEngine(vector_store="chroma", index_dir=str(tmp_path / "chroma"))
        assert type(engine.vector_store).__name__ == "ChromaStore"

    def test_engine_builds_a_qdrant_store(self, tmp_path):
        pytest.importorskip("qdrant_client", reason="install the [qdrant] extra")
        engine = SearchEngine(vector_store="qdrant", index_dir=str(tmp_path / "qdrant"))
        assert type(engine.vector_store).__name__ == "QdrantStore"
        engine.vector_store.client.close()

    def test_engine_indexes_and_searches_through_chroma(self, tmp_path, image_paths):
        pytest.importorskip("chromadb", reason="install the [chroma] extra")
        engine = SearchEngine(vector_store="chroma", index_dir=str(tmp_path / "chroma"))
        engine.index(image_paths)
        assert engine.count == len(image_paths)
        assert engine.search_by_image(image_paths[0], k=1)[0]["metadata"]["image_path"] == image_paths[0]

    def test_tool_builds_a_chroma_store(self, tmp_path):
        pytest.importorskip("chromadb", reason="install the [chroma] extra")
        tool = ImageSearchTool(index_path=str(tmp_path / "chroma"), vector_store_type="chroma")
        assert type(tool.vector_store).__name__ == "ChromaStore"

    def test_tool_builds_a_qdrant_store(self, tmp_path):
        pytest.importorskip("qdrant_client", reason="install the [qdrant] extra")
        tool = ImageSearchTool(index_path=str(tmp_path / "qdrant"), vector_store_type="qdrant")
        assert type(tool.vector_store).__name__ == "QdrantStore"
        tool.vector_store.client.close()


def test_chroma_load_reopens_the_collection(tmp_path, image_paths, embedding):
    pytest.importorskip("chromadb", reason="install the [chroma] extra")
    from DeepImageSearch.vectorstores.chroma_store import ChromaStore

    store = ChromaStore(collection_name="reopened", persist_directory=str(tmp_path))
    Indexer(embedding=embedding, vector_store=store).index(image_paths)

    fresh = ChromaStore(collection_name="reopened", persist_directory=str(tmp_path))
    fresh.load(str(tmp_path))
    assert fresh.count() == len(image_paths)


def test_chroma_add_without_metadata(tmp_path):
    pytest.importorskip("chromadb", reason="install the [chroma] extra")
    from DeepImageSearch.vectorstores.chroma_store import ChromaStore

    store = ChromaStore(collection_name="no_meta", persist_directory=str(tmp_path))
    store.add(["a"], np.eye(4, dtype=np.float32)[:1])
    assert store.count() == 1
    assert store.search(np.eye(4, dtype=np.float32)[0], k=1)[0]["metadata"] == {}


def test_qdrant_load_reopens_local_storage(tmp_path, image_paths, embedding):
    pytest.importorskip("qdrant_client", reason="install the [qdrant] extra")
    from DeepImageSearch.vectorstores.qdrant_store import QdrantStore

    store = QdrantStore(collection_name="reopened", path=str(tmp_path), dimension=DIM)
    Indexer(embedding=embedding, vector_store=store).index(image_paths)
    store.client.close()

    fresh = QdrantStore(collection_name="reopened", path=str(tmp_path), dimension=DIM)
    assert fresh.count() == len(image_paths)
    fresh.client.close()
