"""
Demo 7: Metadata Storage (JSON and PostgreSQL)

Track every indexed image as a structured record.
Default: local JSON file. Production: PostgreSQL.

Requires (for PostgreSQL): uv pip install "DeepImageSearch[postgres]"

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine

# === Local JSON Store (default) ===
print("=== JSON Metadata Store ===")
engine = SearchEngine(model_name="clip-vit-b-32", index_dir="./demo_index")
engine.index("./images")

# View all records
records = engine.get_records()
print(f"Total records: {len(records)}")

# Each record has: image_id, image_index, image_name, image_path, caption, indexed_at, extra
for r in records[:3]:
    print(f"  [{r['image_index']}] {r['image_name']} (id={r['image_id'][:8]}...)")

# Look up by ID
first_id = records[0]["image_id"]
record = engine.get_record(first_id)
print(f"\nLookup: {record}")

# Records saved to: ./demo_index/image_records.json


# === PostgreSQL Store ===
# Uncomment below to use PostgreSQL:
#
# from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore
#
# store = PostgresMetadataStore(
#     connection_string="postgresql://user:pass@localhost:5432/mydb",
#     table_name="image_records",
# )
#
# engine = SearchEngine(
#     model_name="clip-vit-b-32",
#     metadata_store=store,
#     index_dir="./pg_demo_index",
# )
# engine.index("./images")
#
# # Records are now in PostgreSQL
# records = engine.get_records()
# print(f"PostgreSQL records: {len(records)}")


# === Custom Metadata Store ===
# Implement BaseMetadataStore for any backend:
#
# from DeepImageSearch.metadatastore.base import BaseMetadataStore, ImageRecord
#
# class SQLiteStore(BaseMetadataStore):
#     def add(self, records): ...
#     def get(self, image_id): ...
#     def get_by_index(self, image_index): ...
#     def list_all(self): ...
#     def count(self): ...
#     def delete(self, image_ids): ...
#     def next_index(self): ...
#     def save(self, path): ...
#     def load(self, path): ...
