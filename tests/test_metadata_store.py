# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""Tests for ImageRecord and JsonMetadataStore."""

import json
import os

import pytest

from DeepImageSearch.metadatastore.base import BaseMetadataStore, ImageRecord
from DeepImageSearch.metadatastore.json_store import RECORDS_FILENAME, JsonMetadataStore


def record(index: int, image_id: str = None, **kwargs) -> ImageRecord:
    return ImageRecord(
        image_id=image_id or f"id{index}",
        image_index=index,
        image_name=f"img{index}.png",
        image_path=f"/images/img{index}.png",
        indexed_at="2026-01-01T00:00:00+00:00",
        **kwargs,
    )


class TestImageRecord:
    def test_roundtrip_through_dict(self):
        original = record(1, caption="a cat", extra={"tag": "pet"})
        assert ImageRecord.from_dict(original.to_dict()) == original

    def test_from_dict_tolerates_missing_optional_fields(self):
        rec = ImageRecord.from_dict({
            "image_id": "abc",
            "image_index": 0,
            "image_name": "a.png",
            "image_path": "/a.png",
        })
        assert rec.caption is None
        assert rec.indexed_at == ""
        assert rec.extra == {}

    def test_from_dict_requires_core_fields(self):
        with pytest.raises(KeyError):
            ImageRecord.from_dict({"image_id": "abc"})

    def test_extra_defaults_are_not_shared_between_instances(self):
        first, second = record(1), record(2)
        first.extra["tag"] = "only-first"
        assert second.extra == {}


class TestJsonMetadataStore:
    def test_add_and_get(self):
        store = JsonMetadataStore()
        store.add([record(0), record(1)])
        assert store.count() == 2
        assert store.get("id0").image_name == "img0.png"

    def test_get_unknown_id_returns_none(self):
        assert JsonMetadataStore().get("nope") is None

    def test_get_by_index(self):
        store = JsonMetadataStore()
        store.add([record(0), record(1)])
        assert store.get_by_index(1).image_id == "id1"
        assert store.get_by_index(99) is None

    def test_add_same_id_twice_updates_in_place(self):
        store = JsonMetadataStore()
        store.add([record(0)])
        store.add([record(0, caption="updated")])
        assert store.count() == 1
        assert store.get("id0").caption == "updated"

    def test_list_all_is_sorted_by_index(self):
        store = JsonMetadataStore()
        store.add([record(2), record(0), record(1)])
        assert [r.image_index for r in store.list_all()] == [0, 1, 2]

    def test_delete_removes_only_named_ids(self):
        store = JsonMetadataStore()
        store.add([record(0), record(1)])
        store.delete(["id0"])
        assert store.count() == 1
        assert store.get("id0") is None

    def test_delete_unknown_id_is_a_no_op(self):
        store = JsonMetadataStore()
        store.add([record(0)])
        store.delete(["never-added"])
        assert store.count() == 1

    def test_next_index_on_empty_store(self):
        assert JsonMetadataStore().next_index() == 0

    def test_next_index_follows_highest_existing(self):
        store = JsonMetadataStore()
        store.add([record(0), record(7)])
        assert store.next_index() == 8

    def test_next_index_after_deleting_the_highest(self):
        # Reusing a freed index would collide with vectors already in the store,
        # but next_index() is defined off what remains — pin the actual behaviour.
        store = JsonMetadataStore()
        store.add([record(0), record(7)])
        store.delete(["id7"])
        assert store.next_index() == 1


class TestPersistence:
    def test_save_then_load_roundtrip(self, tmp_path):
        store = JsonMetadataStore()
        store.add([record(0, caption="a cat", extra={"tag": "pet"}), record(1)])
        store.save(str(tmp_path))

        reloaded = JsonMetadataStore()
        reloaded.load(str(tmp_path))
        assert reloaded.count() == 2
        assert reloaded.get("id0").caption == "a cat"
        assert reloaded.get("id0").extra == {"tag": "pet"}

    def test_save_creates_directory_and_file(self, tmp_path):
        target = tmp_path / "nested" / "deeper"
        store = JsonMetadataStore()
        store.add([record(0)])
        store.save(str(target))
        assert (target / RECORDS_FILENAME).exists()

    def test_saved_file_is_a_json_array_of_records(self, tmp_path):
        store = JsonMetadataStore()
        store.add([record(0)])
        store.save(str(tmp_path))
        data = json.loads((tmp_path / RECORDS_FILENAME).read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert data[0]["image_id"] == "id0"

    def test_load_from_missing_file_starts_fresh(self, tmp_path):
        store = JsonMetadataStore()
        store.load(str(tmp_path))
        assert store.count() == 0

    def test_load_replaces_existing_records(self, tmp_path):
        saved = JsonMetadataStore()
        saved.add([record(0)])
        saved.save(str(tmp_path))

        store = JsonMetadataStore()
        store.add([record(5, image_id="stale")])
        store.load(str(tmp_path))
        assert store.get("stale") is None
        assert store.count() == 1

    def test_save_handles_non_ascii_captions(self, tmp_path):
        store = JsonMetadataStore()
        store.add([record(0, caption="chat noir — 猫")])
        store.save(str(tmp_path))
        reloaded = JsonMetadataStore()
        reloaded.load(str(tmp_path))
        assert reloaded.get("id0").caption == "chat noir — 猫"

    def test_save_is_idempotent(self, tmp_path):
        store = JsonMetadataStore()
        store.add([record(0)])
        store.save(str(tmp_path))
        first = (tmp_path / RECORDS_FILENAME).read_text(encoding="utf-8")
        store.save(str(tmp_path))
        assert (tmp_path / RECORDS_FILENAME).read_text(encoding="utf-8") == first


def test_json_store_implements_the_full_interface():
    abstract = BaseMetadataStore.__abstractmethods__
    assert abstract  # guard against the ABC losing its @abstractmethod markers
    assert not abstract - set(dir(JsonMetadataStore))
    assert not JsonMetadataStore.__abstractmethods__


def test_records_filename_is_stable():
    # Downstream users load this file directly; renaming it is a breaking change.
    assert RECORDS_FILENAME == "image_records.json"
    assert not os.path.isabs(RECORDS_FILENAME)
