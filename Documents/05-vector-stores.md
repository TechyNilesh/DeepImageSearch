# Vector Stores

DeepImageSearch supports three vector store backends for storing and searching image embeddings.

## FAISS (Default)

[FAISS](https://github.com/facebookresearch/faiss) is Facebook's library for fast similarity search. It's the default and requires no extra dependencies.

```python
engine = SearchEngine(model_name="clip-vit-b-32", vector_store="faiss")
```

### Index Types

| Type | Method | Speed | Accuracy | Best For |
|---|---|---|---|---|
| `flat` | Exact search | Slowest | 100% | Small datasets (<100K) |
| `ivf` | Inverted file index | Fast | ~95% | Medium datasets |
| `hnsw` | Graph-based | Fastest | ~97% | Large datasets |

```python
engine = SearchEngine(model_name="clip-vit-b-32", index_type="hnsw")
```

### Files Saved

- `index.faiss` -- binary FAISS index
- `metadata.json` -- IDs and per-image metadata

### Using FAISSStore Directly

```python
from DeepImageSearch.vectorstores.faiss_store import FAISSStore

store = FAISSStore(dimension=512, index_type="flat")
store.add(ids=["id1", "id2"], vectors=vectors_array, metadata=[{"key": "val"}, {}])
results = store.search(query_vector, k=10, filters={"key": "val"})
store.save("./my_index")
store.load("./my_index")
```

## ChromaDB

[ChromaDB](https://www.trychroma.com/) is a local-first vector database with built-in metadata filtering.

```shell
uv pip install "DeepImageSearch[chroma]"
```

```python
engine = SearchEngine(
    model_name="clip-vit-b-32",
    vector_store="chroma",
    index_dir="./chroma_index",
    chroma_collection="my_images",
)
```

### Using ChromaStore Directly

```python
from DeepImageSearch.vectorstores.chroma_store import ChromaStore

store = ChromaStore(
    collection_name="my_images",
    persist_directory="./chroma_data",  # None for in-memory
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `collection_name` | `str` | `"deep_image_search"` | Chroma collection name. |
| `persist_directory` | `str` or `None` | `None` | Path for persistent storage. None for in-memory. |

ChromaDB auto-persists when using `PersistentClient`.

## Qdrant

[Qdrant](https://qdrant.tech/) is a production-grade vector database with advanced filtering.

```shell
uv pip install "DeepImageSearch[qdrant]"
```

```python
engine = SearchEngine(
    model_name="clip-vit-b-32",
    vector_store="qdrant",
    index_dir="./qdrant_index",
)
```

### Using QdrantStore Directly

```python
from DeepImageSearch.vectorstores.qdrant_store import QdrantStore

# In-memory
store = QdrantStore(dimension=512)

# Local persistent
store = QdrantStore(dimension=512, path="./qdrant_data")

# Remote server
store = QdrantStore(dimension=512, location="http://localhost:6333")
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `collection_name` | `str` | `"deep_image_search"` | Collection name. |
| `location` | `str` or `None` | `None` | Server URL. None for in-memory. |
| `path` | `str` or `None` | `None` | Local persistent path. |
| `dimension` | `int` | `512` | Vector dimension. |

## Common Interface

All vector stores implement `BaseVectorStore`:

```python
store.add(ids, vectors, metadata)          # add vectors
store.search(query_vector, k=10, filters)  # search
store.delete(ids)                          # delete by ID
store.count()                              # total vectors
store.save(path)                           # persist
store.load(path)                           # load
```

### Search Result Format

```python
[
    {"id": "abc123", "score": 0.92, "metadata": {"image_path": "...", ...}},
    {"id": "def456", "score": 0.87, "metadata": {"image_path": "...", ...}},
]
```

### Metadata Filtering

Pass `filters` dict to `search()`. Behavior varies by backend:

- **FAISS**: Post-filters results (searches extra candidates, then filters)
- **ChromaDB**: Native `where` clause filtering
- **Qdrant**: Native `FieldCondition` filtering

```python
results = engine.search("sunset", filters={"source": "instagram"})
```

## Custom Vector Store

Implement `BaseVectorStore` for your own backend:

```python
from DeepImageSearch.vectorstores.base import BaseVectorStore

class MyStore(BaseVectorStore):
    def add(self, ids, vectors, metadata=None): ...
    def search(self, query_vector, k=10, filters=None): ...
    def delete(self, ids): ...
    def count(self): ...
    def save(self, path): ...
    def load(self, path): ...

engine = SearchEngine(model_name="clip-vit-b-32", vector_store=MyStore())
```
