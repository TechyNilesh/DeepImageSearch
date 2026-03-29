# API Reference

Complete reference for all public classes and methods.

## SearchEngine

`DeepImageSearch.search_engine.SearchEngine`

```python
SearchEngine(
    model_name: str = "clip-vit-b-32",
    vector_store: Union[str, BaseVectorStore] = "faiss",
    metadata_store: Optional[BaseMetadataStore] = None,
    index_dir: str = "deep_image_search_index",
    device: Optional[str] = None,
    index_type: str = "flat",
    batch_size: int = 64,
    captioner_model: Optional[str] = None,
    captioner_api_key: Optional[str] = None,
    captioner_base_url: Optional[str] = None,
    chroma_collection: str = "deep_image_search",
    qdrant_location: Optional[str] = None,
)
```

| Method | Returns | Description |
|---|---|---|
| `index(image_paths, metadata, generate_captions, caption_prompt, save)` | `int` | Index images |
| `add_images(image_paths, metadata, generate_captions, save)` | `int` | Add images to existing index |
| `search(query, k, filters, mode, text_weight, image_query)` | `List[Dict]` | Search for images |
| `search_by_text(text, k, filters)` | `List[Dict]` | Text-to-image search |
| `search_by_image(image_path, k, filters)` | `List[Dict]` | Image-to-image search |
| `get_similar_images(image_path, number_of_images)` | `Dict[int, str]` | v2-compatible search |
| `plot_similar_images(image_path, number_of_images)` | `None` | Plot results |
| `get_records()` | `List[Dict]` | All image records |
| `get_record(image_id)` | `Dict` or `None` | Single record |
| `save()` | `None` | Save to disk |
| `load()` | `None` | Load from disk |
| `info()` | `Dict` | Engine summary |
| `count` (property) | `int` | Number of indexed images |
| `supports_text_search` (property) | `bool` | Whether text search is available |

---

## EmbeddingManager

`DeepImageSearch.core.embeddings.EmbeddingManager`

| Method | Returns | Description |
|---|---|---|
| `create(model_name, device, batch_size, **kwargs)` | `BaseEmbedding` | Create embedding backend |
| `list_presets()` | `dict` | Available CLIP presets |

## CLIPEmbedding

`DeepImageSearch.core.embeddings.CLIPEmbedding`

```python
CLIPEmbedding(model_name="ViT-B-32", pretrained="openai", device=None, batch_size=64)
```

| Method | Returns | Description |
|---|---|---|
| `embed_images(images: List[Image])` | `np.ndarray` | (N, D) image embeddings |
| `embed_texts(texts: List[str])` | `np.ndarray` | (N, D) text embeddings |
| `embed_image(image: Image)` | `np.ndarray` | Single image vector |
| `embed_text(text: str)` | `np.ndarray` | Single text vector |

Properties: `dimension: int`, `supports_text: bool = True`

## TimmEmbedding

`DeepImageSearch.core.embeddings.TimmEmbedding`

```python
TimmEmbedding(model_name="vgg19", pretrained=True, image_size=224, device=None, batch_size=64)
```

| Method | Returns | Description |
|---|---|---|
| `embed_images(images: List[Image])` | `np.ndarray` | (N, D) image embeddings |
| `embed_image(image: Image)` | `np.ndarray` | Single image vector |

Properties: `dimension: int`, `supports_text: bool = False`

---

## Indexer

`DeepImageSearch.core.indexer.Indexer`

```python
Indexer(embedding, vector_store, metadata_store=None, captioner=None, batch_size=64)
```

| Method | Returns | Description |
|---|---|---|
| `index(image_paths, extra_metadata, generate_captions, caption_prompt)` | `int` | Index images |
| `add_images(image_paths, extra_metadata, generate_captions)` | `int` | Add to index |

## Searcher

`DeepImageSearch.core.searcher.Searcher`

```python
Searcher(embedding, vector_store)
```

| Method | Returns | Description |
|---|---|---|
| `search(query, k, filters, mode, text_weight, image_query)` | `List[Dict]` | Multi-modal search |
| `search_by_text(text, k, filters)` | `List[Dict]` | Text search |
| `search_by_image(image_path, k, filters)` | `List[Dict]` | Image search |
| `get_similar_images(image_path, number_of_images)` | `Dict[int, str]` | v2-compatible |
| `plot_similar_images(image_path, number_of_images)` | `None` | Visualization |

---

## Captioner

`DeepImageSearch.core.captioner.Captioner`

```python
Captioner(model: str, api_key: str, base_url: str, max_image_size=1024, max_tokens=500)
```

| Method | Returns | Description |
|---|---|---|
| `caption(image_path, prompt)` | `str` | Caption single image |
| `caption_batch(image_paths, prompt, on_error)` | `Dict[str, str]` | Caption multiple images |
| `extract_metadata(image_path, prompt)` | `Dict` | Extract structured JSON metadata |

---

## Vector Stores

### BaseVectorStore

`DeepImageSearch.vectorstores.base.BaseVectorStore` (abstract)

| Method | Returns | Description |
|---|---|---|
| `add(ids, vectors, metadata)` | `None` | Add vectors |
| `search(query_vector, k, filters)` | `List[Dict]` | Search |
| `delete(ids)` | `None` | Delete by ID |
| `count()` | `int` | Total vectors |
| `save(path)` | `None` | Persist |
| `load(path)` | `None` | Load |

### FAISSStore

```python
FAISSStore(dimension: int, index_type: str = "flat")
```

### ChromaStore

```python
ChromaStore(collection_name="deep_image_search", persist_directory=None)
```

### QdrantStore

```python
QdrantStore(collection_name="deep_image_search", location=None, path=None, dimension=512)
```

---

## Metadata Stores

### ImageRecord

`DeepImageSearch.metadatastore.base.ImageRecord` (dataclass)

| Field | Type | Default | Description |
|---|---|---|---|
| `image_id` | `str` | required | MD5 hash of path |
| `image_index` | `int` | required | Position in index |
| `image_name` | `str` | required | Filename |
| `image_path` | `str` | required | Full path |
| `caption` | `str` or `None` | `None` | LLM caption |
| `indexed_at` | `str` | `""` | ISO timestamp |
| `extra` | `Dict` or `None` | `{}` | Extra metadata |

Methods: `to_dict()`, `from_dict(data)`

### BaseMetadataStore

`DeepImageSearch.metadatastore.base.BaseMetadataStore` (abstract)

| Method | Returns | Description |
|---|---|---|
| `add(records)` | `None` | Add records |
| `get(image_id)` | `ImageRecord` or `None` | By ID |
| `get_by_index(image_index)` | `ImageRecord` or `None` | By index |
| `list_all()` | `List[ImageRecord]` | All records sorted |
| `count()` | `int` | Total records |
| `delete(image_ids)` | `None` | Delete by ID |
| `next_index()` | `int` | Next available index |
| `save(path)` | `None` | Persist |
| `load(path)` | `None` | Load |

### JsonMetadataStore

```python
JsonMetadataStore()
```

Saves to `image_records.json`. Implements all `BaseMetadataStore` methods.

### PostgresMetadataStore

```python
PostgresMetadataStore(connection_string: str, table_name="image_records", auto_create=True)
```

Additional method: `close()` -- close DB connection.

---

## Load_Data

`DeepImageSearch.data.loader.Load_Data`

| Method | Returns | Description |
|---|---|---|
| `from_folder(folder_list, recursive=True, validate=True)` | `List[str]` | Scan folders |
| `from_csv(csv_file_path, images_column_name)` | `List[str]` | Read CSV |
| `from_list(image_paths, validate=True)` | `List[str]` | Validate paths |

---

## ImageSearchTool

`DeepImageSearch.agents.tool_interface.ImageSearchTool`

```python
ImageSearchTool(index_path: str, model_name="clip-vit-b-32", vector_store_type="faiss", device=None)
```

| Method | Returns | Description |
|---|---|---|
| `__call__(query, k=5, mode="auto", filters=None)` | `List[Dict]` | Search |
| `tool_definition` (property) | `Dict` | Function calling schema |

---

## Search_Setup (v2 Backward Compat)

`DeepImageSearch.DeepImageSearch.Search_Setup`

```python
Search_Setup(image_list, model_name="vgg19", pretrained=True, image_count=None, image_size=224, metadata_dir="metadata-files", use_gpu=False, index_type="flat")
```

| Method | Returns | Description |
|---|---|---|
| `run_index(force_reindex=False)` | `None` | Index images |
| `add_images_to_index(new_image_paths)` | `None` | Add images |
| `get_similar_images(image_path, number_of_images=10)` | `Dict[int, str]` | Search |
| `plot_similar_images(image_path, number_of_images=6)` | `None` | Plot |
| `get_image_metadata_file()` | `Dict` | Index info |
