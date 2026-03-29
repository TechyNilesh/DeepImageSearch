# Search Engine

`SearchEngine` is the main entry point for DeepImageSearch. It wraps embeddings, indexing, vector store, metadata store, and search into a single class.

## Creating a Search Engine

```python
from DeepImageSearch import SearchEngine

engine = SearchEngine(
    model_name="clip-vit-b-32",         # embedding model preset
    vector_store="faiss",                # "faiss", "chroma", or "qdrant"
    metadata_store=None,                 # None = default JsonMetadataStore
    index_dir="deep_image_search_index", # where to save index files
    device=None,                         # "cuda", "mps", "cpu", or None (auto)
    index_type="flat",                   # FAISS only: "flat", "ivf", "hnsw"
    batch_size=64,                       # images per batch during indexing
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"clip-vit-b-32"` | Embedding model. See [Embeddings](./03-embeddings.md) for presets. |
| `vector_store` | `str` or `BaseVectorStore` | `"faiss"` | Vector store backend. |
| `metadata_store` | `BaseMetadataStore` or `None` | `None` | Image record store. None creates `JsonMetadataStore`. |
| `index_dir` | `str` | `"deep_image_search_index"` | Directory for persisted files. |
| `device` | `str` or `None` | `None` | Compute device. None auto-detects. |
| `index_type` | `str` | `"flat"` | FAISS index type. `"flat"` (exact), `"ivf"` (fast approximate), `"hnsw"` (fast approximate). |
| `batch_size` | `int` | `64` | Batch size for embedding extraction. |
| `captioner_model` | `str` or `None` | `None` | Vision LLM model name for captioning. |
| `captioner_api_key` | `str` or `None` | `None` | API key for the captioning provider. |
| `captioner_base_url` | `str` or `None` | `None` | API base URL for the captioning provider. |
| `chroma_collection` | `str` | `"deep_image_search"` | ChromaDB collection name. |
| `qdrant_location` | `str` or `None` | `None` | Qdrant server URL. |

## Indexing Images

```python
# From a folder path
engine.index("./photos")

# From a list of paths
engine.index(["img1.jpg", "img2.jpg", "img3.jpg"])

# Single image
engine.index("single_image.jpg")

# With extra metadata per image
engine.index(
    ["img1.jpg", "img2.jpg"],
    metadata=[{"source": "camera"}, {"source": "web"}]
)

# With LLM captioning
engine.index("./photos", generate_captions=True)

# Custom caption prompt
engine.index("./photos", generate_captions=True, caption_prompt="Describe briefly.")

# Don't auto-save (useful for batch operations)
engine.index("./photos", save=False)
engine.save()  # save manually later
```

### `index()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_paths` | `str` or `List[str]` | required | Folder path, single image path, or list of image paths. |
| `metadata` | `List[Dict]` or `None` | `None` | Extra metadata per image. Length must match image count. |
| `generate_captions` | `bool` | `False` | Auto-generate captions (requires captioner config). |
| `caption_prompt` | `str` or `None` | `None` | Custom captioning prompt. |
| `save` | `bool` | `True` | Persist index to disk after indexing. |

**Returns:** `int` -- number of images indexed.

### Adding More Images

```python
engine.add_images("./more_photos")
engine.add_images(["new1.jpg", "new2.jpg"])
```

Same parameters as `index()`.

## Searching

### Text Search

```python
results = engine.search("a red car parked near a lake")
results = engine.search("sunset", k=20)
```

### Image Search

```python
results = engine.search("query.jpg")
```

### Hybrid Search

```python
results = engine.search(
    "outdoor scene",
    image_query="photo.jpg",
    mode="hybrid",
    text_weight=0.6,  # 60% text, 40% image
)
```

### Filtered Search

```python
results = engine.search("sunset", filters={"source": "instagram"})
```

### `search()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str`, `PIL.Image`, or `np.ndarray` | required | Text, image, or vector query. |
| `k` | `int` | `10` | Number of results. |
| `filters` | `Dict` or `None` | `None` | Metadata filters. |
| `mode` | `str` | `"auto"` | `"auto"`, `"text"`, `"image"`, or `"hybrid"`. |
| `text_weight` | `float` | `0.5` | Weight for text in hybrid mode (0-1). |
| `image_query` | `str`, `PIL.Image`, or `None` | `None` | Secondary image for hybrid mode. |

**Returns:** `List[Dict]` -- each dict has `id`, `score`, `metadata`.

### Convenience Methods

```python
# Text-only search
results = engine.search_by_text("sunset photo", k=5)

# Image-only search
results = engine.search_by_image("query.jpg", k=5)
```

## Visualization

```python
engine.plot_similar_images("query.jpg", number_of_images=9)
```

Displays the query image and a grid of similar results using matplotlib.

## Image Records

```python
# All records
records = engine.get_records()

# Single record by ID
record = engine.get_record("a1b2c3d4...")

# Count
print(engine.count)

# Engine info
print(engine.info())
```

## Persistence

```python
# Save index + records to disk
engine.save()

# Load existing index + records
engine.load()
```

Files saved to `index_dir`:
- `index.faiss` -- FAISS vector index
- `metadata.json` -- vector store metadata (IDs + per-image metadata)
- `image_records.json` -- structured image records

## v2 Backward Compatibility

The old `Search_Setup` API still works:

```python
from DeepImageSearch import Load_Data, Search_Setup

image_list = Load_Data().from_folder(["folder_path"])
st = Search_Setup(image_list=image_list, model_name="vgg19", pretrained=True)
st.run_index()
st.get_similar_images(image_path="query.jpg", number_of_images=10)
st.plot_similar_images(image_path="query.jpg", number_of_images=9)
```
