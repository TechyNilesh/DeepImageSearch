# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for PostgresMetadataStore.

A running PostgreSQL server is not required (and would make CI flaky): a fake
`psycopg2` module records the SQL and parameters the store emits and replays
canned rows. That covers the parts the store owns — statement shape, parameter
order, table-name substitution, and row-to-record mapping.
"""

import json
import sys
import types

import pytest

from DeepImageSearch.metadatastore.base import BaseMetadataStore, ImageRecord

ROW = ("id0", 0, "img0.png", "/images/img0.png", "a caption", "2026-01-01T00:00:00+00:00", {"tag": "pet"})


class FakeCursor:
    def __init__(self, log, rows):
        self.log = log
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows["one"]

    def fetchall(self):
        return self.rows["all"]


class FakeConnection:
    def __init__(self, log, rows):
        self.log = log
        self.rows = rows
        self.autocommit = False
        self.closed = False

    def cursor(self):
        return FakeCursor(self.log, self.rows)

    def close(self):
        self.closed = True


@pytest.fixture
def pg(monkeypatch):
    """Install a fake psycopg2; returns a factory for (store, log, rows)."""
    state = {"log": [], "rows": {"one": None, "all": []}, "connections": []}

    def connect(connection_string):
        state["connection_string"] = connection_string
        conn = FakeConnection(state["log"], state["rows"])
        state["connections"].append(conn)
        return conn

    module = types.ModuleType("psycopg2")
    module.connect = connect
    monkeypatch.setitem(sys.modules, "psycopg2", module)

    def build(**kwargs):
        from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

        store = PostgresMetadataStore(connection_string="postgresql://u:p@localhost/db", **kwargs)
        state["log"].clear()  # drop the CREATE TABLE noise
        return store, state

    return build


def record(index=0, **kwargs):
    return ImageRecord(
        image_id=f"id{index}",
        image_index=index,
        image_name=f"img{index}.png",
        image_path=f"/images/img{index}.png",
        indexed_at="2026-01-01T00:00:00+00:00",
        **kwargs,
    )


class TestConnection:
    def test_connects_with_the_given_string_and_sets_autocommit(self, pg):
        store, state = pg()
        assert state["connection_string"] == "postgresql://u:p@localhost/db"
        assert store.conn.autocommit is True

    def test_creates_the_table_by_default(self, monkeypatch, pg):
        from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

        state = {"log": [], "rows": {"one": None, "all": []}, "connections": []}
        module = types.ModuleType("psycopg2")
        module.connect = lambda cs: FakeConnection(state["log"], state["rows"])
        monkeypatch.setitem(sys.modules, "psycopg2", module)

        PostgresMetadataStore(connection_string="postgresql://x")
        assert any("CREATE TABLE IF NOT EXISTS" in sql for sql, _ in state["log"])

    def test_auto_create_can_be_disabled(self, monkeypatch):
        from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

        state = {"log": [], "rows": {"one": None, "all": []}}
        module = types.ModuleType("psycopg2")
        module.connect = lambda cs: FakeConnection(state["log"], state["rows"])
        monkeypatch.setitem(sys.modules, "psycopg2", module)

        PostgresMetadataStore(connection_string="postgresql://x", auto_create=False)
        assert state["log"] == []

    def test_missing_driver_gives_an_actionable_error(self, monkeypatch):
        from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

        monkeypatch.setitem(sys.modules, "psycopg2", None)
        with pytest.raises(ImportError, match=r"DeepImageSearch\[postgres\]"):
            PostgresMetadataStore(connection_string="postgresql://x")


class TestWrites:
    def test_add_upserts_each_record_with_fields_in_column_order(self, pg):
        store, state = pg()
        store.add([record(0, caption="a cat", extra={"tag": "pet"})])

        sql, params = state["log"][0]
        assert sql.startswith("INSERT INTO image_records")
        assert "ON CONFLICT (image_id) DO UPDATE" in sql
        assert params[:6] == ("id0", 0, "img0.png", "/images/img0.png", "a cat", "2026-01-01T00:00:00+00:00")
        assert json.loads(params[6]) == {"tag": "pet"}

    def test_empty_extra_is_stored_as_null(self, pg):
        store, state = pg()
        store.add([record(0)])
        assert state["log"][0][1][6] is None

    def test_add_issues_one_statement_per_record(self, pg):
        store, state = pg()
        store.add([record(0), record(1), record(2)])
        assert len(state["log"]) == 3

    def test_delete_uses_one_placeholder_per_id(self, pg):
        store, state = pg()
        store.delete(["a", "b", "c"])
        sql, params = state["log"][0]
        assert "WHERE image_id IN (%s, %s, %s)" in sql
        assert params == ("a", "b", "c")

    def test_delete_with_no_ids_touches_the_database(self, pg):
        store, state = pg()
        store.delete([])
        assert state["log"] == []


class TestReads:
    def test_get_maps_a_row_to_a_record(self, pg):
        store, state = pg()
        state["rows"]["one"] = ROW
        result = store.get("id0")
        assert result == ImageRecord(*ROW)
        assert state["log"][0][1] == ("id0",)

    def test_get_returns_none_when_absent(self, pg):
        store, state = pg()
        state["rows"]["one"] = None
        assert store.get("missing") is None

    def test_null_extra_becomes_an_empty_dict(self, pg):
        store, state = pg()
        state["rows"]["one"] = ROW[:6] + (None,)
        assert store.get("id0").extra == {}

    def test_get_by_index_queries_the_index_column(self, pg):
        store, state = pg()
        state["rows"]["one"] = ROW
        assert store.get_by_index(0).image_id == "id0"
        assert "WHERE image_index = %s" in state["log"][0][0]

    def test_list_all_is_ordered_by_index(self, pg):
        store, state = pg()
        state["rows"]["all"] = [ROW, ("id1", 1) + ROW[2:]]
        records = store.list_all()
        assert [r.image_index for r in records] == [0, 1]
        assert "ORDER BY image_index" in state["log"][0][0]

    def test_count_reads_the_scalar(self, pg):
        store, state = pg()
        state["rows"]["one"] = (7,)
        assert store.count() == 7
        assert "SELECT COUNT(*)" in state["log"][0][0]

    def test_count_of_an_empty_table(self, pg):
        store, state = pg()
        state["rows"]["one"] = None
        assert store.count() == 0

    def test_next_index_follows_the_maximum(self, pg):
        store, state = pg()
        state["rows"]["one"] = (7,)
        assert store.next_index() == 8
        assert "SELECT MAX(image_index)" in state["log"][0][0]

    def test_next_index_on_an_empty_table_is_zero(self, pg):
        store, state = pg()
        state["rows"]["one"] = (None,)
        assert store.next_index() == 0


class TestCustomTableName:
    def test_statements_target_the_configured_table(self, pg):
        store, state = pg(table_name="my_images")
        state["rows"]["one"] = (0,)
        store.count()
        state["rows"]["one"] = ROW
        store.get("id0")
        state["rows"]["all"] = [ROW]
        store.list_all()
        assert all("my_images" in sql for sql, _ in state["log"])
        assert not any("FROM image_records" in sql for sql, _ in state["log"])


class TestLifecycle:
    def test_save_and_load_are_no_ops(self, pg):
        store, state = pg()
        store.save("/anywhere")
        store.load("/anywhere")
        assert state["log"] == []

    def test_close_closes_the_connection(self, pg):
        store, _ = pg()
        store.close()
        assert store.conn.closed is True

    def test_close_is_idempotent(self, pg):
        store, _ = pg()
        store.close()
        store.close()
        assert store.conn.closed is True

    def test_del_after_a_failed_init_does_not_raise(self, monkeypatch):
        """__del__ runs even when __init__ bailed before setting self.conn."""
        from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

        monkeypatch.setitem(sys.modules, "psycopg2", None)
        with pytest.raises(ImportError):
            PostgresMetadataStore(connection_string="postgresql://x")

        half_built = PostgresMetadataStore.__new__(PostgresMetadataStore)
        half_built.__del__()  # must not raise AttributeError


def test_implements_the_full_interface():
    from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

    assert not PostgresMetadataStore.__abstractmethods__
    assert not BaseMetadataStore.__abstractmethods__ - set(dir(PostgresMetadataStore))
