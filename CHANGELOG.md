# Changelog

All notable changes to DeepImageSearch will be documented in this file.

---

## [3.0.0] - 2026-03-29

### Complete rewrite for the agentic RAG / LLM era.

### New Features

#### Text-to-Image Search
- Search images using natural language queries (e.g. "a red car parked near a lake")
- Powered by CLIP/SigLIP/EVA-CLIP multimodal embeddings via open_clip
- 8 CLIP presets: `clip-vit-b-32`, `clip-vit-b-16`, `clip-vit-l-14`, `clip-vit-l-14-336`, `clip-vit-bigg-14`, `eva-clip-vit-b-16`, `siglip-vit-b-16`, `siglip-vit-l-16`

#### Hybrid Search
- Combine text and image queries with configurable weight fusion
- `engine.search("outdoor scene", image_query="photo.jpg", mode="hybrid", text_weight=0.6)`

#### LLM-Powered Image Captioning
- Auto-generate captions during indexing using any OpenAI SDK-compatible vision LLM
- Works with OpenAI, Gemini, Claude, Ollama, Together AI, Groq, vLLM, or any compatible endpoint
- Just provide `model`, `api_key`, `base_url` -- no provider-specific code
- Structured metadata extraction (JSON) with objects, scene, colors, tags

#### Image Records & Metadata Storage
- Every indexed image tracked as a structured `ImageRecord` (like a database row)
- Fields: `image_id`, `image_index`, `image_name`, `image_path`, `caption`, `indexed_at`, `extra`
- Schema maps directly to SQL tables
- **JsonMetadataStore** (default) -- saves `image_records.json` locally
- **PostgresMetadataStore** -- production-grade PostgreSQL backend with indexes and upserts
- Custom backends via `BaseMetadataStore` abstract class
- `engine.get_records()` and `engine.get_record(id)` for querying

#### Multiple Vector Store Backends
- **FAISS** (default) -- flat, ivf, hnsw index types with metadata sidecar
- **ChromaDB** -- persistent, with native metadata filtering
- **Qdrant** -- production-grade with in-memory, local, or remote server modes
- Pluggable via `BaseVectorStore` abstract interface

#### Agentic Integration
- **MCP Server** -- expose image search as tools for Claude Code / Claude Desktop
  - CLI: `deep-image-search-mcp --index-path ./my_index`
  - Tools: `search_images`, `get_index_info`
- **LangChain Tool** -- `create_langchain_tool()` returns a `StructuredTool`
- **Generic Tool** -- `ImageSearchTool` callable with function calling schema

#### Folder Path Indexing
- `engine.index("./photos")` -- pass a folder path directly
- `engine.index(["img1.jpg", "img2.jpg"])` -- or a list of paths
- `engine.index("single.jpg")` -- or a single file
- Auto-loads images using `Load_Data.from_folder()` internally

#### SearchEngine Unified API
- Single `SearchEngine` class wraps everything: embeddings, indexing, search, captioning, persistence
- `engine.search()` auto-detects query type (text vs image path vs PIL Image vs numpy vector)
- `engine.info()`, `engine.count`, `engine.supports_text_search` properties
- `engine.save()` / `engine.load()` for full persistence

### Architecture Changes

#### New modular package structure
```
DeepImageSearch/
├── core/embeddings.py      -- CLIP/SigLIP/EVA-CLIP + timm
├── core/indexer.py          -- batch indexing pipeline
├── core/searcher.py         -- text/image/hybrid search
├── core/captioner.py        -- OpenAI SDK LLM captioning
├── vectorstores/faiss_store.py
├── vectorstores/chroma_store.py
├── vectorstores/qdrant_store.py
├── metadatastore/json_store.py
├── metadatastore/postgres_store.py
├── agents/mcp_server.py
├── agents/langchain_tool.py
├── agents/tool_interface.py
├── data/loader.py
├── search_engine.py         -- unified high-level API
└── DeepImageSearch.py       -- v2 backward-compatible shim
```

#### Removed
- `setup.py` -- replaced by `pyproject.toml` with hatchling
- `config.py` -- no longer needed
- `pandas` dependency -- replaced with `csv` stdlib for data loading
- Old 638-line monolithic `DeepImageSearch.py` -- replaced with 160-line thin shim delegating to v3 modules
- `torch.autograd.Variable` usage -- modern PyTorch doesn't need it

### Packaging & Dependencies

#### Modern packaging with uv support
- `pyproject.toml` with hatchling build backend
- Python 3.10+ required
- Optional dependency groups: `[llm]`, `[chroma]`, `[qdrant]`, `[postgres]`, `[mcp]`, `[langchain]`, `[all]`
- `uv pip install DeepImageSearch` or `pip install DeepImageSearch`

#### Updated core dependencies
- `torch >= 2.2.0` (was 2.0.0)
- `torchvision >= 0.17.0` (was 0.15.1)
- `open-clip-torch >= 2.26.0` (new)
- `timm >= 1.0.0` (was 0.6.13)
- `faiss-cpu >= 1.8.0` (was 1.7.3)
- `numpy >= 1.26.0` (was 1.24.2)
- `Pillow >= 10.0.0` (was 9.5.0)
- `tqdm >= 4.66.0` (was 4.65.0)
- `matplotlib >= 3.8.0` (was 3.5.2)

#### Removed core dependencies
- `pandas` -- no longer a core dependency

### Backward Compatibility

- v2 API (`Load_Data`, `Search_Setup`) still works unchanged
- `Search_Setup` is now a thin wrapper delegating to v3 modules
- Old `run_index()`, `get_similar_images()`, `plot_similar_images()` methods preserved

### Documentation

- 9 feature-specific documentation files in `Documents/`
- 10 ready-to-run demo scripts in `Demo/`
- Full API reference with all class/method signatures
- Updated README with screenshots and examples table

---

## [2.5] - Previous Release

### Features
- 500+ pre-trained models via timm
- FAISS integration for similarity search
- Basic image-to-image search
- CSV and folder data loading
- Matplotlib visualization

### Bug Fixes (pre-3.0)
- Fixed directory check using `os.path.exists()` instead of `os.listdir()`
- Removed blocking `input()` call, replaced with `force_reindex` parameter
- Fixed bare `except:` clauses with specific exceptions
- Added batch processing for memory efficiency
- Replaced print statements with logging module
- Added comprehensive type hints and input validation
- Added GPU support, configurable image size, multiple FAISS index types
