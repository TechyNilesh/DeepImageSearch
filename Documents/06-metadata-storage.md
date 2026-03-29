# Metadata Storage

Every indexed image is tracked as a structured `ImageRecord`. Records are stored separately from vectors, enabling database-backed storage for production.

## ImageRecord Schema

Each record has these fields (maps directly to a SQL table):

| Field | Type | Description |
|---|---|---|
| `image_id` | `str` (PK) | Deterministic MD5 hash of the file path |
| `image_index` | `int` (UNIQUE) | Auto-incrementing position (0, 1, 2, ...) |
| `image_name` | `str` | Original filename (e.g. `"sunset_042.jpg"`) |
| `image_path` | `str` | Full absolute path to the image file |
| `caption` | `str` or `None` | LLM-generated caption (if captioning enabled) |
| `indexed_at` | `str` | ISO 8601 timestamp of when the image was indexed |
| `extra` | `dict` or `None` | Additional user-provided metadata |

```python
from DeepImageSearch.metadatastore.base import ImageRecord

record = ImageRecord(
    image_id="a1b2c3d4",
    image_index=0,
    image_name="photo.jpg",
    image_path="/data/photos/photo.jpg",
    caption="A sunset over mountains",
    indexed_at="2026-03-28T10:30:00+00:00",
    extra={"source": "camera", "album": "vacation"},
)

# Convert to/from dict
d = record.to_dict()
record = ImageRecord.from_dict(d)
```

## JSON Store (Default)

Records are saved as a JSON array in `image_records.json` inside the index directory. This is the default, zero-config option.

```python
from DeepImageSearch import SearchEngine

engine = SearchEngine(model_name="clip-vit-b-32")
engine.index("./photos")

# Records saved automatically to: deep_image_search_index/image_records.json
```

### Using JsonMetadataStore Directly

```python
from DeepImageSearch.metadatastore.json_store import JsonMetadataStore

store = JsonMetadataStore()
store.load("./my_index")              # load existing records
print(store.count())                   # total records
records = store.list_all()             # all records sorted by index
record = store.get("a1b2c3d4")        # by image_id
record = store.get_by_index(42)       # by position
print(store.next_index())             # next available index
store.save("./my_index")              # persist to disk
```

### JSON File Format

```json
[
    {
        "image_id": "a1b2c3d4",
        "image_index": 0,
        "image_name": "sunset.jpg",
        "image_path": "/data/photos/sunset.jpg",
        "caption": "A sunset over mountains",
        "indexed_at": "2026-03-28T10:30:00+00:00",
        "extra": {"source": "camera"}
    }
]
```

## PostgreSQL Store

For production deployments, store records in PostgreSQL.

```shell
uv pip install "DeepImageSearch[postgres]"
```

```python
from DeepImageSearch import SearchEngine
from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

store = PostgresMetadataStore(
    connection_string="postgresql://user:pass@localhost:5432/mydb",
    table_name="image_records",    # default
    auto_create=True,              # create table if not exists
)

engine = SearchEngine(
    model_name="clip-vit-b-32",
    metadata_store=store,
)
engine.index("./photos")
# Records go to PostgreSQL, vectors go to FAISS
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection_string` | `str` | required | PostgreSQL connection string. |
| `table_name` | `str` | `"image_records"` | Table name for records. |
| `auto_create` | `bool` | `True` | Create table + indexes automatically. |

### SQL Schema

The table is created automatically with:

```sql
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
```

### PostgreSQL Methods

All `BaseMetadataStore` methods work the same, plus:

```python
store.close()  # close the database connection
```

The `save()` and `load()` methods are no-ops for PostgreSQL since it auto-persists.

## Querying Records from SearchEngine

```python
# All records as list of dicts
records = engine.get_records()

# Single record by image_id
record = engine.get_record("a1b2c3d4")

# Image count
print(engine.count)
```

## Custom Metadata Store

Implement `BaseMetadataStore` for your own backend (SQLite, MongoDB, DynamoDB, etc.):

```python
from DeepImageSearch.metadatastore.base import BaseMetadataStore, ImageRecord

class MyStore(BaseMetadataStore):
    def add(self, records: list[ImageRecord]) -> None: ...
    def get(self, image_id: str) -> ImageRecord | None: ...
    def get_by_index(self, image_index: int) -> ImageRecord | None: ...
    def list_all(self) -> list[ImageRecord]: ...
    def count(self) -> int: ...
    def delete(self, image_ids: list[str]) -> None: ...
    def next_index(self) -> int: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

engine = SearchEngine(model_name="clip-vit-b-32", metadata_store=MyStore())
```
