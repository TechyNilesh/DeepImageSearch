# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Tests for the FAISS vector store and the BaseVectorStore contract."""

import numpy as np
import pytest

from DeepImageSearch.vectorstores.base import BaseVectorStore
from DeepImageSearch.vectorstores.faiss_store import FAISSStore

DIM = 4


def unit(*values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


@pytest.fixture
def populated_store():
    """Three orthogonal unit vectors, so nearest-neighbour order is unambiguous."""
    store = FAISSStore(dimension=DIM)
    store.add(
        ids=["a", "b", "c"],
        vectors=np.vstack([unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)]),
        metadata=[{"colour": "red"}, {"colour": "green"}, {"colour": "red"}],
    )
    return store


class TestConstruction:
    @pytest.mark.parametrize("index_type", ["flat", "hnsw"])
    def test_supported_index_types_build(self, index_type):
        store = FAISSStore(dimension=DIM, index_type=index_type)
        assert store.count() == 0
        assert store.index.d == DIM

    def test_unknown_index_type_raises(self):
        with pytest.raises(ValueError, match="Unknown index_type"):
            FAISSStore(dimension=DIM, index_type="quantum")


class TestAdd:
    def test_count_reflects_added_vectors(self, populated_store):
        assert populated_store.count() == 3

    def test_metadata_defaults_to_empty_dicts(self):
        store = FAISSStore(dimension=DIM)
        store.add(ids=["a"], vectors=unit(1, 0, 0, 0).reshape(1, -1))
        assert store.search(unit(1, 0, 0, 0), k=1)[0]["metadata"] == {}

    def test_accepts_non_contiguous_float64_input(self):
        store = FAISSStore(dimension=DIM)
        vectors = np.eye(4, dtype=np.float64)[:2]  # float64, and a view
        store.add(ids=["a", "b"], vectors=vectors)
        assert store.count() == 2


class TestSearch:
    def test_returns_nearest_first(self, populated_store):
        results = populated_store.search(unit(1, 0, 0, 0), k=3)
        assert results[0]["id"] == "a"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_respects_k(self, populated_store):
        assert len(populated_store.search(unit(1, 0, 0, 0), k=2)) == 2

    def test_result_shape(self, populated_store):
        result = populated_store.search(unit(1, 0, 0, 0), k=1)[0]
        assert set(result) == {"id", "score", "metadata"}
        assert isinstance(result["score"], float)

    def test_empty_store_returns_no_results(self):
        assert FAISSStore(dimension=DIM).search(unit(1, 0, 0, 0), k=5) == []

    def test_k_larger_than_index_is_clamped(self, populated_store):
        assert len(populated_store.search(unit(1, 0, 0, 0), k=99)) == 3

    def test_accepts_a_row_vector(self, populated_store):
        results = populated_store.search(unit(1, 0, 0, 0).reshape(1, -1), k=1)
        assert results[0]["id"] == "a"


class TestFilters:
    def test_equality_filter(self, populated_store):
        results = populated_store.search(unit(1, 0, 0, 0), k=3, filters={"colour": "red"})
        assert {r["id"] for r in results} == {"a", "c"}

    def test_list_filter_matches_any_value(self, populated_store):
        results = populated_store.search(unit(1, 0, 0, 0), k=3, filters={"colour": ["green", "red"]})
        assert len(results) == 3

    def test_filter_on_absent_key_excludes_the_record(self, populated_store):
        assert populated_store.search(unit(1, 0, 0, 0), k=3, filters={"missing": "x"}) == []

    def test_filters_are_combined_with_and(self, populated_store):
        results = populated_store.search(unit(1, 0, 0, 0), k=3, filters={"colour": "red", "missing": 1})
        assert results == []

    def test_filtered_search_still_honours_k(self, populated_store):
        assert len(populated_store.search(unit(1, 0, 0, 0), k=1, filters={"colour": "red"})) == 1


class TestDelete:
    def test_removes_vector_and_metadata(self, populated_store):
        populated_store.delete(["a"])
        assert populated_store.count() == 2
        assert "a" not in {r["id"] for r in populated_store.search(unit(1, 0, 0, 0), k=3)}

    def test_surviving_records_keep_their_metadata(self, populated_store):
        populated_store.delete(["a"])
        results = populated_store.search(unit(0, 1, 0, 0), k=1)
        assert results[0]["id"] == "b"
        assert results[0]["metadata"] == {"colour": "green"}

    def test_deleting_unknown_id_is_a_no_op(self, populated_store):
        populated_store.delete(["never-added"])
        assert populated_store.count() == 3

    def test_deleting_everything_empties_the_store(self, populated_store):
        populated_store.delete(["a", "b", "c"])
        assert populated_store.count() == 0
        assert populated_store.search(unit(1, 0, 0, 0), k=1) == []


class TestPersistence:
    def test_save_load_roundtrip_preserves_results(self, populated_store, tmp_path):
        populated_store.save(str(tmp_path))

        reloaded = FAISSStore(dimension=DIM)
        reloaded.load(str(tmp_path))

        assert reloaded.count() == 3
        assert reloaded.dimension == DIM
        assert reloaded.index_type == "flat"
        results = reloaded.search(unit(0, 0, 1, 0), k=1)
        assert results[0]["id"] == "c"
        assert results[0]["metadata"] == {"colour": "red"}

    def test_save_creates_missing_directories(self, populated_store, tmp_path):
        target = tmp_path / "nested" / "deeper"
        populated_store.save(str(target))
        assert (target / "index.faiss").exists()
        assert (target / "metadata.json").exists()

    def test_reloaded_store_accepts_more_vectors(self, populated_store, tmp_path):
        populated_store.save(str(tmp_path))
        reloaded = FAISSStore(dimension=DIM)
        reloaded.load(str(tmp_path))
        reloaded.add(ids=["d"], vectors=unit(0, 0, 0, 1).reshape(1, -1), metadata=[{"colour": "blue"}])
        assert reloaded.count() == 4
        assert reloaded.search(unit(0, 0, 0, 1), k=1)[0]["id"] == "d"


def test_faiss_store_implements_the_full_interface():
    abstract = BaseVectorStore.__abstractmethods__
    assert abstract
    assert not abstract - set(dir(FAISSStore))
    assert not FAISSStore.__abstractmethods__
