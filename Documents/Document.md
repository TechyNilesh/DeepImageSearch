# DeepImageSearch v3 Documentation

Complete documentation for DeepImageSearch, organized by feature.

## Table of Contents

1. [Getting Started](./01-getting-started.md) -- Installation, quick start, basic concepts
2. [Search Engine](./02-search-engine.md) -- `SearchEngine` class, indexing, searching, hybrid search
3. [Embeddings](./03-embeddings.md) -- CLIP, SigLIP, timm models, presets, custom embeddings
4. [LLM Captioning](./04-captioning.md) -- Auto-captioning with any OpenAI SDK-compatible provider
5. [Vector Stores](./05-vector-stores.md) -- FAISS, ChromaDB, Qdrant backends
6. [Metadata Storage](./06-metadata-storage.md) -- Image records, JSON store, PostgreSQL store
7. [Agentic Integration](./07-agentic-integration.md) -- MCP server, LangChain tool, generic tool
8. [Data Loading](./08-data-loading.md) -- Loading images from folders, CSV, lists
9. [API Reference](./09-api-reference.md) -- Full API reference for all classes and methods

## Examples

Ready-to-run demo scripts in the [`Demo/`](../Demo) folder:

| # | Demo | Description |
|---|---|---|
| 1 | [Basic Image Search](../Demo/01_basic_image_search.py) | Index a folder, find similar images, plot results |
| 2 | [Text-to-Image Search](../Demo/02_text_to_image_search.py) | Search images with natural language queries |
| 3 | [Hybrid Search](../Demo/03_hybrid_search.py) | Combine text + image queries with weight tuning |
| 4 | [Filtered Search](../Demo/04_filtered_search.py) | Attach metadata and filter results |
| 5 | [LLM Captioning](../Demo/05_llm_captioning.py) | Auto-generate captions with any vision LLM |
| 6 | [Vector Stores](../Demo/06_vector_stores.py) | FAISS vs ChromaDB vs Qdrant |
| 7 | [Metadata Storage](../Demo/07_metadata_storage.py) | JSON records, PostgreSQL, custom stores |
| 8 | [Agentic Tools](../Demo/08_agentic_tools.py) | MCP server, LangChain tool, generic tool |
| 9 | [Embedding Models](../Demo/09_embedding_models.py) | Compare CLIP presets and timm models |
| 10 | [Incremental Indexing](../Demo/10_incremental_indexing.py) | Add images over time, save/reload |
