"""
PostgreSQL-based metadata store.

Stores image records in a PostgreSQL table using psycopg2.
Schema maps directly from the ImageRecord dataclass.

Requires: uv pip install 'DeepImageSearch[postgres]'
"""

import json
import logging
from typing import List, Optional

from DeepImageSearch.metadatastore.base import BaseMetadataStore, ImageRecord

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS image_records (
    image_id    TEXT PRIMARY KEY,
    image_index INTEGER UNIQUE NOT NULL,
    image_name  TEXT NOT NULL,
    image_path  TEXT NOT NULL,
    caption     TEXT,
    indexed_at  TEXT NOT NULL,
    extra       JSONB
);
CREATE INDEX IF NOT EXISTS idx_image_records_index ON image_records (image_index);
CREATE INDEX IF NOT EXISTS idx_image_records_name ON image_records (image_name);
"""

UPSERT_SQL = """
INSERT INTO image_records (image_id, image_index, image_name, image_path, caption, indexed_at, extra)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (image_id) DO UPDATE SET
    image_index = EXCLUDED.image_index,
    image_name  = EXCLUDED.image_name,
    image_path  = EXCLUDED.image_path,
    caption     = EXCLUDED.caption,
    indexed_at  = EXCLUDED.indexed_at,
    extra       = EXCLUDED.extra;
"""


class PostgresMetadataStore(BaseMetadataStore):
    """
    PostgreSQL backend for image records.

    Parameters
    ----------
    connection_string : str
        PostgreSQL connection string.
        e.g. "postgresql://user:pass@localhost:5432/dbname"
    table_name : str
        Table name for image records. Default: 'image_records'.
    auto_create : bool
        Automatically create the table if it doesn't exist. Default: True.

    Examples
    --------
    store = PostgresMetadataStore(
        connection_string="postgresql://user:pass@localhost:5432/mydb"
    )
    engine = SearchEngine(
        model_name="clip-vit-b-32",
        metadata_store=store,
    )
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "image_records",
        auto_create: bool = True,
    ):
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required. Install with: uv pip install 'DeepImageSearch[postgres]'"
            )

        self.connection_string = connection_string
        self.table_name = table_name
        self.conn = psycopg2.connect(connection_string)
        self.conn.autocommit = True

        if auto_create:
            self._create_table()

        logger.info(f"PostgresMetadataStore connected: table={table_name}")

    def _create_table(self) -> None:
        sql = CREATE_TABLE_SQL.replace("image_records", self.table_name)
        with self.conn.cursor() as cur:
            cur.execute(sql)

    def _execute(self, sql: str, params=None):
        sql = sql.replace("image_records", self.table_name)
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur

    def _fetchone(self, sql: str, params=None):
        sql = sql.replace("image_records", self.table_name)
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def _fetchall(self, sql: str, params=None):
        sql = sql.replace("image_records", self.table_name)
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    @staticmethod
    def _row_to_record(row) -> ImageRecord:
        return ImageRecord(
            image_id=row[0],
            image_index=row[1],
            image_name=row[2],
            image_path=row[3],
            caption=row[4],
            indexed_at=row[5],
            extra=row[6] if row[6] else {},
        )

    def add(self, records: List[ImageRecord]) -> None:
        sql = UPSERT_SQL.replace("image_records", self.table_name)
        with self.conn.cursor() as cur:
            for record in records:
                extra_json = json.dumps(record.extra) if record.extra else None
                cur.execute(sql, (
                    record.image_id,
                    record.image_index,
                    record.image_name,
                    record.image_path,
                    record.caption,
                    record.indexed_at,
                    extra_json,
                ))
        logger.debug(f"Added {len(records)} records to PostgreSQL")

    def get(self, image_id: str) -> Optional[ImageRecord]:
        row = self._fetchone(
            "SELECT * FROM image_records WHERE image_id = %s", (image_id,)
        )
        return self._row_to_record(row) if row else None

    def get_by_index(self, image_index: int) -> Optional[ImageRecord]:
        row = self._fetchone(
            "SELECT * FROM image_records WHERE image_index = %s", (image_index,)
        )
        return self._row_to_record(row) if row else None

    def list_all(self) -> List[ImageRecord]:
        rows = self._fetchall(
            "SELECT * FROM image_records ORDER BY image_index"
        )
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) FROM image_records")
        return row[0] if row else 0

    def delete(self, image_ids: List[str]) -> None:
        if not image_ids:
            return
        placeholders = ", ".join(["%s"] * len(image_ids))
        self._execute(
            f"DELETE FROM image_records WHERE image_id IN ({placeholders})",
            tuple(image_ids),
        )

    def next_index(self) -> int:
        row = self._fetchone("SELECT MAX(image_index) FROM image_records")
        if row and row[0] is not None:
            return row[0] + 1
        return 0

    def save(self, path: str) -> None:
        # PostgreSQL auto-persists, no-op
        pass

    def load(self, path: str) -> None:
        # PostgreSQL auto-persists, no-op
        pass

    def close(self) -> None:
        """Close the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __del__(self):
        self.close()
