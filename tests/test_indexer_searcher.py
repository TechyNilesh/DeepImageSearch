# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Integration tests for the index → search pipeline.

These wire the real Indexer, FAISSStore, JsonMetadataStore and Searcher
together, substituting only the embedding backend so no weights are downloaded.
"""

import numpy as np
import pytest
from PIL import Image

from DeepImageSearch.core.indexer import Indexer, _path_to_id
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.metadatastore.json_store import JsonMetadataStore
from DeepImageSearch.vectorstores.faiss_store import FAISSStore
from tests.conftest import DIM, make_image


@pytest.fixture
def pipeline(embedding):
    """A ready-to-use (indexer, searcher, vector store, metadata store) bundle."""
    store = FAISSStore(dimension=DIM)
    records = JsonMetadataStore()
    indexer = Indexer(embedding=embedding, vector_store=store, metadata_store=records)
    searcher = Searcher(embedding=embedding, vector_store=store)
    return indexer, searcher, store, records


class TestIndexing:
    def test_indexes_every_image(self, pipeline, image_paths):
        indexer, _, store, records = pipeline
        assert indexer.index(image_paths) == len(image_paths)
        assert store.count() == len(image_paths)
        assert records.count() == len(image_paths)

    def test_records_carry_path_name_and_index(self, pipeline, image_paths):
        indexer, _, _, records = pipeline
        indexer.index(image_paths)
        stored = records.list_all()
        assert [r.image_index for r in stored] == list(range(len(image_paths)))
        assert {r.image_path for r in stored} == set(image_paths)
        assert all(r.image_name and r.indexed_at for r in stored)

    def test_image_id_is_a_deterministic_function_of_path(self, pipeline, image_paths):
        indexer, _, _, records = pipeline
        indexer.index(image_paths)
        assert records.get(_path_to_id(image_paths[0])) is not None

    def test_empty_input_indexes_nothing(self, pipeline):
        indexer, _, store, _ = pipeline
        assert indexer.index([]) == 0
        assert store.count() == 0

    def test_unreadable_images_are_skipped_not_fatal(self, pipeline, image_paths, tmp_path):
        indexer, _, store, _ = pipeline
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not a png")
        assert indexer.index(image_paths + [str(broken)]) == len(image_paths)
        assert store.count() == len(image_paths)

    def test_batching_covers_every_image(self, embedding, image_paths):
        store = FAISSStore(dimension=DIM)
        indexer = Indexer(embedding=embedding, vector_store=store, batch_size=2)
        assert indexer.index(image_paths) == len(image_paths)
        assert embedding.embed_images_calls > 1  # genuinely batched
        assert store.count() == len(image_paths)

    def test_works_without_a_metadata_store(self, embedding, image_paths):
        store = FAISSStore(dimension=DIM)
        assert Indexer(embedding=embedding, vector_store=store).index(image_paths) == len(image_paths)

    def test_mismatched_extra_metadata_length_raises(self, pipeline, image_paths):
        indexer, _, _, _ = pipeline
        with pytest.raises(ValueError, match="must match"):
            indexer.index(image_paths, extra_metadata=[{"tag": "x"}])


class TestIncrementalIndexing:
    def test_add_images_continues_the_index_sequence(self, pipeline, image_paths, tmp_path):
        indexer, _, store, records = pipeline
        indexer.index(image_paths)

        later = make_image(tmp_path / "later.png", (10, 20, 30))
        indexer.add_images([later])

        assert store.count() == len(image_paths) + 1
        assert [r.image_index for r in records.list_all()] == list(range(len(image_paths) + 1))
        assert records.get(_path_to_id(later)).image_index == len(image_paths)


class TestSearch:
    def test_image_search_ranks_the_query_image_first(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        results = searcher.search_by_image(image_paths[0], k=3)
        assert results[0]["metadata"]["image_path"] == image_paths[0]

    def test_results_expose_path_score_and_id(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        result = searcher.search_by_image(image_paths[0], k=1)[0]
        assert set(result) == {"id", "score", "metadata"}
        assert result["metadata"]["image_path"] in image_paths

    def test_text_search_returns_k_results(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        assert len(searcher.search_by_text("a red square", k=2)) == 2

    def test_text_search_rejects_image_only_backends(self, image_only_embedding, image_paths):
        store = FAISSStore(dimension=DIM)
        Indexer(embedding=image_only_embedding, vector_store=store).index(image_paths)
        searcher = Searcher(embedding=image_only_embedding, vector_store=store)
        with pytest.raises(ValueError, match="CLIP-family"):
            searcher.search_by_text("a red square")

    def test_precomputed_vector_query_is_used_as_is(self, pipeline, image_paths, embedding):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        vector = embedding.embed_image(Image.open(image_paths[0]))
        assert searcher.search(vector, k=1)[0]["metadata"]["image_path"] == image_paths[0]

    def test_pil_image_query(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        with Image.open(image_paths[0]) as img:
            results = searcher.search(img, k=1)
        assert results[0]["metadata"]["image_path"] == image_paths[0]

    def test_unsupported_query_type_raises(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        with pytest.raises(TypeError, match="Unsupported query type"):
            searcher.search(42)

    def test_auto_mode_treats_a_path_as_an_image(self, pipeline, image_paths, embedding):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        before = embedding.embed_texts_calls
        searcher.search(image_paths[0], k=1)
        assert embedding.embed_texts_calls == before  # went down the image path

    def test_auto_mode_treats_a_sentence_as_text(self, pipeline, image_paths, embedding):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        before = embedding.embed_texts_calls
        searcher.search("a photograph of a red square", k=1)
        assert embedding.embed_texts_calls == before + 1

    def test_text_mode_forces_text_even_for_pathlike_strings(self, pipeline, image_paths, embedding):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        before = embedding.embed_texts_calls
        searcher.search(image_paths[0], k=1, mode="text")
        assert embedding.embed_texts_calls == before + 1


class TestFilteredSearch:
    def test_filters_narrow_results_to_matching_metadata(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        tags = [{"tag": "keep"}] + [{"tag": "drop"}] * (len(image_paths) - 1)
        indexer.index(image_paths, extra_metadata=tags)
        results = searcher.search_by_image(image_paths[0], k=5, filters={"tag": "keep"})
        assert len(results) == 1
        assert results[0]["metadata"]["image_path"] == image_paths[0]

    def test_extra_metadata_is_stored_on_the_record(self, pipeline, image_paths):
        indexer, _, _, records = pipeline
        indexer.index(image_paths, extra_metadata=[{"tag": f"t{i}"} for i in range(len(image_paths))])
        assert records.get(_path_to_id(image_paths[0])).extra == {"tag": "t0"}


class TestHybridSearch:
    def test_combines_text_and_image_queries(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        results = searcher.search("a red square", k=2, mode="hybrid", image_query=image_paths[0])
        assert len(results) == 2

    def test_weighting_shifts_the_query_vector(self, pipeline, image_paths, embedding):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        text_heavy = searcher.search("a red square", k=3, mode="hybrid",
                                     image_query=image_paths[0], text_weight=0.9)
        image_heavy = searcher.search("a red square", k=3, mode="hybrid",
                                      image_query=image_paths[0], text_weight=0.1)
        assert image_heavy[0]["metadata"]["image_path"] == image_paths[0]
        assert [r["score"] for r in text_heavy] != [r["score"] for r in image_heavy]

    def test_accepts_a_pil_image_as_the_image_query(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        with Image.open(image_paths[0]) as img:
            assert searcher.search("a red square", k=1, mode="hybrid", image_query=img)

    def test_missing_image_query_raises(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        with pytest.raises(ValueError, match="image_query must be provided"):
            searcher.search("a red square", k=1, mode="hybrid")

    def test_non_text_primary_query_raises(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        with pytest.raises(ValueError, match="must be a text string"):
            searcher.search(np.zeros(DIM, dtype=np.float32), k=1, mode="hybrid", image_query=image_paths[0])

    def test_bad_image_query_type_raises(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        with pytest.raises(ValueError, match="image_query must be"):
            searcher.search("a red square", k=1, mode="hybrid", image_query=123)

    def test_rejected_on_image_only_backends(self, image_only_embedding, image_paths):
        store = FAISSStore(dimension=DIM)
        Indexer(embedding=image_only_embedding, vector_store=store).index(image_paths)
        searcher = Searcher(embedding=image_only_embedding, vector_store=store)
        with pytest.raises(ValueError, match="Hybrid search requires"):
            searcher.search("a red square", mode="hybrid", image_query=image_paths[0])


class TestBackwardCompatibleApi:
    def test_get_similar_images_returns_index_to_path_mapping(self, pipeline, image_paths):
        indexer, searcher, _, _ = pipeline
        indexer.index(image_paths)
        similar = searcher.get_similar_images(image_paths[0], number_of_images=2)
        assert list(similar) == [0, 1]
        assert similar[0] == image_paths[0]


class TestPersistenceAcrossSessions:
    def test_saved_index_returns_the_same_results_after_reload(self, pipeline, image_paths, tmp_path, embedding):
        indexer, _, store, records = pipeline
        indexer.index(image_paths)
        store.save(str(tmp_path))
        records.save(str(tmp_path))

        reloaded_store = FAISSStore(dimension=DIM)
        reloaded_store.load(str(tmp_path))
        reloaded_records = JsonMetadataStore()
        reloaded_records.load(str(tmp_path))

        results = Searcher(embedding=embedding, vector_store=reloaded_store).search_by_image(image_paths[0], k=1)
        assert results[0]["metadata"]["image_path"] == image_paths[0]
        assert reloaded_records.count() == len(image_paths)
