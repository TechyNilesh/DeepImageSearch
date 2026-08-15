# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Contract tests for the optional vector store backends.

Chroma and Qdrant are behind extras, so each class skips unless its dependency
is installed. Both run against a real in-process instance — no server needed.
The point is that every backend honours the same BaseVectorStore contract as
FAISS, since SearchEngine swaps between them by name.
"""

import numpy as np
import pytest

from DeepImageSearch.vectorstores.base import BaseVectorStore

DIM = 4


def unit(*values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


IDS = ["a", "b", "c"]
VECTORS = np.vstack([unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)])
METADATA = [{"colour": "red"}, {"colour": "green"}, {"colour": "red"}]


class VectorStoreContract:
    """Shared expectations; subclasses provide a `store` fixture."""

    def test_count_reflects_added_vectors(self, store):
        assert store.count() == 3

    def test_search_returns_nearest_first(self, store):
        results = store.search(unit(1, 0, 0, 0), k=3)
        assert results[0]["id"] == "a"

    def test_result_shape_matches_the_contract(self, store):
        result = store.search(unit(1, 0, 0, 0), k=1)[0]
        assert set(result) == {"id", "score", "metadata"}
        assert isinstance(result["score"], float)
        assert result["metadata"]["colour"] == "red"

    def test_search_respects_k(self, store):
        assert len(store.search(unit(1, 0, 0, 0), k=2)) == 2

    def test_equality_filter(self, store):
        results = store.search(unit(1, 0, 0, 0), k=3, filters={"colour": "red"})
        assert {r["id"] for r in results} == {"a", "c"}

    def test_filter_with_no_matches_returns_nothing(self, store):
        assert store.search(unit(1, 0, 0, 0), k=3, filters={"colour": "puce"}) == []

    def test_delete_removes_the_vector(self, store):
        store.delete(["a"])
        assert store.count() == 2
        assert "a" not in {r["id"] for r in store.search(unit(1, 0, 0, 0), k=3)}

    def test_implements_the_full_interface(self, store):
        assert not type(store).__abstractmethods__
        assert isinstance(store, BaseVectorStore)


class TestChromaStore(VectorStoreContract):
    @pytest.fixture
    def store(self):
        pytest.importorskip("chromadb", reason="install the [chroma] extra to run these")
        from DeepImageSearch.vectorstores.chroma_store import ChromaStore

        store = ChromaStore(collection_name="test_contract")
        store.add(IDS, VECTORS, METADATA)
        yield store
        store.client.delete_collection("test_contract")

    def test_non_primitive_metadata_is_stringified(self):
        pytest.importorskip("chromadb")
        from DeepImageSearch.vectorstores.chroma_store import ChromaStore

        store = ChromaStore(collection_name="test_coercion")
        try:
            store.add(["a"], unit(1, 0, 0, 0).reshape(1, -1), [{"tags": ["x", "y"], "n": 3}])
            meta = store.search(unit(1, 0, 0, 0), k=1)[0]["metadata"]
            assert meta["tags"] == "['x', 'y']"  # coerced, not dropped
            assert meta["n"] == 3                # primitives kept as-is
        finally:
            store.client.delete_collection("test_coercion")

    def test_persistent_client_writes_to_disk(self, tmp_path):
        pytest.importorskip("chromadb")
        from DeepImageSearch.vectorstores.chroma_store import ChromaStore

        store = ChromaStore(collection_name="persisted", persist_directory=str(tmp_path))
        store.add(IDS, VECTORS, METADATA)
        store.save(str(tmp_path))  # documented as a no-op; must not raise

        reopened = ChromaStore(collection_name="persisted", persist_directory=str(tmp_path))
        assert reopened.count() == 3


class TestQdrantStore(VectorStoreContract):
    @pytest.fixture
    def store(self):
        pytest.importorskip("qdrant_client", reason="install the [qdrant] extra to run these")
        from DeepImageSearch.vectorstores.qdrant_store import QdrantStore

        store = QdrantStore(collection_name="test_contract", dimension=DIM)
        store.add(IDS, VECTORS, METADATA)
        return store

    def test_original_string_ids_survive_the_uuid_round_trip(self, store):
        # Qdrant requires numeric/UUID point ids, so the real id rides in the payload.
        results = store.search(unit(1, 0, 0, 0), k=3)
        assert {r["id"] for r in results} == set(IDS)
        assert all("_original_id" not in r["metadata"] for r in results)

    def test_reindexing_the_same_id_updates_in_place(self, store):
        store.add(["a"], unit(0, 0, 0, 1).reshape(1, -1), [{"colour": "blue"}])
        assert store.count() == 3
        assert store.search(unit(0, 0, 0, 1), k=1)[0]["id"] == "a"

    def test_local_path_storage_persists(self, tmp_path):
        pytest.importorskip("qdrant_client")
        from DeepImageSearch.vectorstores.qdrant_store import QdrantStore

        store = QdrantStore(collection_name="persisted", path=str(tmp_path), dimension=DIM)
        store.add(IDS, VECTORS, METADATA)
        store.save(str(tmp_path))  # documented as a no-op; must not raise
        store.client.close()

        reopened = QdrantStore(collection_name="persisted", path=str(tmp_path), dimension=DIM)
        assert reopened.count() == 3
        reopened.client.close()


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [("chromadb", "ChromaStore"), ("qdrant_client", "QdrantStore")],
)
def test_optional_store_is_exported_exactly_when_its_dependency_is_installed(module_name, class_name):
    import importlib

    from DeepImageSearch import vectorstores

    installed = importlib.util.find_spec(module_name) is not None
    assert (class_name in vectorstores.__all__) is installed
    assert hasattr(vectorstores, class_name) is installed
