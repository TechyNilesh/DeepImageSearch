# Deep Image Search - AI-Based Image Search Engine
<p align="center"><img src="https://raw.githubusercontent.com/TechyNilesh/DeepImageSearch/786e96c48561d67be47dccbab2bc8debced414a3/images/deep%20image%20search%20logo%20New.png" alt="Deep+Image+Search+logo" height="218" width="350"></p>

**DeepImageSearch** is a Python library for building AI-powered image search systems. It supports text-to-image search, image-to-image search, hybrid search, and LLM-powered captioning using CLIP/SigLIP/EVA-CLIP multimodal embeddings with FAISS/ChromaDB/Qdrant vector indexing. Built for the agentic RAG era with MCP server, LangChain tool, and PostgreSQL metadata storage out of the box.

[![PyPI version](https://img.shields.io/pypi/v/DeepImageSearch.svg)](https://pypi.org/project/DeepImageSearch/) ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![CLIP](https://img.shields.io/badge/CLIP-Multimodal-purple.svg) ![MCP](https://img.shields.io/badge/MCP-Agentic-orange.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Downloads](https://static.pepy.tech/personalized-badge/deepimagesearch?period=total&units=none&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/deepimagesearch) [![GitHub stars](https://img.shields.io/github/stars/TechyNilesh/DeepImageSearch)](https://github.com/TechyNilesh/DeepImageSearch/stargazers) [![Last Commit](https://img.shields.io/github/last-commit/TechyNilesh/DeepImageSearch)](https://github.com/TechyNilesh/DeepImageSearch/commits/main)

## Features

- **Text-to-Image Search** -- find images using natural language queries like *"a red car parked near a lake"*
- **Image-to-Image Search** -- find visually similar images from a query image
- **Hybrid Search** -- combine text and image queries with weighted fusion
- **Multimodal Embeddings** -- CLIP, SigLIP, EVA-CLIP via [open_clip](https://github.com/mlfoundations/open_clip), plus 500+ legacy [timm](https://huggingface.co/timm) models
- **LLM Captioning** -- auto-generate image captions using any OpenAI SDK-compatible provider
- **Image Records** -- every image tracked with ID, index, name, path, caption, timestamp (like a database)
- **Multiple Vector Stores** -- FAISS (default), ChromaDB, Qdrant with metadata filtering
- **Metadata Storage** -- local JSON (default) or PostgreSQL for production
- **Agentic Integration** -- MCP server for Claude, LangChain tool for agent pipelines
- **GPU & CPU Support** -- auto-detects CUDA, MPS (Apple Silicon), or CPU
- **Modern Packaging** -- `uv`/`pip` compatible via `pyproject.toml`, Python 3.10+

## Installation

### From PyPI (stable release)

```shell
pip install DeepImageSearch --upgrade
```

### From GitHub (latest v3)

```shell
pip install git+https://github.com/TechyNilesh/DeepImageSearch.git
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```shell
uv pip install git+https://github.com/TechyNilesh/DeepImageSearch.git
```

With optional extras from GitHub:

```shell
pip install "DeepImageSearch[all] @ git+https://github.com/TechyNilesh/DeepImageSearch.git"
```

### Optional Extras

```shell
pip install "DeepImageSearch[llm]"          # LLM captioning (OpenAI SDK)
pip install "DeepImageSearch[chroma]"       # ChromaDB vector store
pip install "DeepImageSearch[qdrant]"       # Qdrant vector store
pip install "DeepImageSearch[postgres]"     # PostgreSQL metadata store
pip install "DeepImageSearch[mcp]"          # MCP server for Claude
pip install "DeepImageSearch[langchain]"    # LangChain agent tool
pip install "DeepImageSearch[all]"          # Everything
```

> If using a GPU, uninstall `faiss-cpu` and install `faiss-gpu` instead.

## Quick Start

```python
from DeepImageSearch import SearchEngine

engine = SearchEngine(model_name="clip-vit-b-32")

# Index from a folder or list of paths
engine.index("./photos")
engine.index(["img1.jpg", "img2.jpg", "img3.jpg"])

# Text search
results = engine.search("a sunset over mountains")

# Image search
results = engine.search("query.jpg")

# Hybrid search
results = engine.search("outdoor scene", image_query="photo.jpg", mode="hybrid")

# Filtered search
results = engine.search("red car", filters={"source": "instagram"})

# Plot results
engine.plot_similar_images("query.jpg", number_of_images=9)
```

### Image-to-Image Search

<p align="center"><img src="images/screenshots/01_image_to_image_search.png" alt="Image to Image Search" width="700"></p>

### Text-to-Image Search

<p align="center"><img src="images/screenshots/02_text_to_image_search.png" alt="Text to Image Search" width="800"></p>

### Search Mode Comparison (Image vs Text vs Hybrid)

<p align="center"><img src="images/screenshots/03_search_mode_comparison.png" alt="Search Mode Comparison" width="800"></p>

### Search Results

Each result contains full image identity -- you always know which image matched:

```python
{
    "id": "a1b2c3...",
    "score": 0.87,
    "metadata": {
        "image_id": "a1b2c3...",
        "image_index": 42,
        "image_name": "sunset_042.jpg",
        "image_path": "/data/photos/sunset_042.jpg",
        "caption": "A sunset over mountains with orange sky",
        "indexed_at": "2026-03-28T10:30:00+00:00"
    }
}
```

### Image Records

Every indexed image is tracked as a structured record (maps directly to SQL):

```python
records = engine.get_records()              # all records
record = engine.get_record("a1b2c3...")     # by ID
print(engine.count)                         # total indexed
print(engine.info())                        # engine summary
```

## LLM Captioning

Auto-generate image captions using any OpenAI SDK-compatible provider. Just pass `model`, `api_key`, and `base_url`:

```python
from DeepImageSearch import SearchEngine

engine = SearchEngine(
    model_name="clip-vit-l-14",
    captioner_model="your-model-name",
    captioner_api_key="your-api-key",
    captioner_base_url="https://your-provider.com/v1",
)

engine.index("./photos", generate_captions=True)
results = engine.search("person holding umbrella")
```

Works with OpenAI, Google Gemini, Anthropic Claude, Ollama, Together AI, Groq, vLLM, or any OpenAI SDK-compatible endpoint.

## Vector Stores

```python
# FAISS (default)
engine = SearchEngine(model_name="clip-vit-b-32")

# ChromaDB
engine = SearchEngine(model_name="clip-vit-b-32", vector_store="chroma")

# Qdrant
engine = SearchEngine(model_name="clip-vit-b-32", vector_store="qdrant")
```

## Metadata Storage

Image records are stored locally in `image_records.json` by default. For production, use PostgreSQL:

```python
from DeepImageSearch import SearchEngine
from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore

store = PostgresMetadataStore(
    connection_string="postgresql://user:pass@localhost:5432/mydb"
)
engine = SearchEngine(model_name="clip-vit-b-32", metadata_store=store)
engine.index("./photos")   # records go to PostgreSQL, vectors go to FAISS
```

You can implement your own backend by subclassing `BaseMetadataStore`.

## Embedding Presets

| Preset | Model | Text Search | Best For |
|---|---|---|---|
| `clip-vit-b-32` | CLIP ViT-B/32 | Yes | Fast, general purpose |
| `clip-vit-b-16` | CLIP ViT-B/16 | Yes | Better accuracy |
| `clip-vit-l-14` | CLIP ViT-L/14 | Yes | High accuracy |
| `clip-vit-l-14-336` | CLIP ViT-L/14@336 | Yes | Highest accuracy |
| `siglip-vit-b-16` | SigLIP ViT-B/16 | Yes | Improved zero-shot |
| `clip-vit-bigg-14` | CLIP ViT-bigG/14 | Yes | Maximum quality |
| `vgg19` | VGG-19 (timm) | No | Legacy, image-only |
| `resnet50` | ResNet-50 (timm) | No | Legacy, image-only |

Any [timm model name](https://huggingface.co/timm) also works for image-only search.

## Agentic Integration

### MCP Server

Expose your image index as a tool for Claude:

```shell
deep-image-search-mcp --index-path ./my_index --model clip-vit-l-14
```

Claude Desktop config:
```json
{
  "mcpServers": {
    "image-search": {
      "command": "deep-image-search-mcp",
      "args": ["--index-path", "./my_index"]
    }
  }
}
```

### LangChain Tool

```python
from DeepImageSearch.agents.langchain_tool import create_langchain_tool

tool = create_langchain_tool(index_path="./my_index")
```

### Generic Tool

```python
from DeepImageSearch import ImageSearchTool

tool = ImageSearchTool(index_path="./my_index")
results = tool("a photo of a dog", k=5)
```

## Advanced Usage

For full control, use core modules directly:

```python
from DeepImageSearch.core.embeddings import EmbeddingManager
from DeepImageSearch.core.indexer import Indexer
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.core.captioner import Captioner
from DeepImageSearch.vectorstores.faiss_store import FAISSStore
from DeepImageSearch.metadatastore.json_store import JsonMetadataStore

embedding = EmbeddingManager.create("clip-vit-l-14", device="cuda")
store = FAISSStore(dimension=embedding.dimension, index_type="hnsw")
metadata = JsonMetadataStore()

captioner = Captioner(
    model="your-model",
    api_key="your-key",
    base_url="https://your-provider.com/v1",
)

indexer = Indexer(embedding=embedding, vector_store=store, metadata_store=metadata, captioner=captioner)
searcher = Searcher(embedding=embedding, vector_store=store)

indexer.index(image_paths, generate_captions=True)
results = searcher.search_by_text("sunset photo")
```

### Backward Compatibility (v2 API)

Existing v2 code continues to work:

```python
from DeepImageSearch import Load_Data, Search_Setup

image_list = Load_Data().from_folder(["folder_path"])
st = Search_Setup(image_list=image_list, model_name="vgg19", pretrained=True)
st.run_index()
st.get_similar_images(image_path="query.jpg", number_of_images=10)
st.plot_similar_images(image_path="query.jpg", number_of_images=9)
```

## Architecture

```
DeepImageSearch/
├── core/
│   ├── embeddings.py      # CLIP/SigLIP/EVA-CLIP + timm backends
│   ├── indexer.py          # Batch indexing pipeline
│   ├── searcher.py         # Text/image/hybrid search + plotting
│   └── captioner.py        # OpenAI SDK-compatible LLM captioning
├── vectorstores/
│   ├── base.py             # Abstract vector store interface
│   ├── faiss_store.py      # FAISS with metadata sidecar
│   ├── chroma_store.py     # ChromaDB integration
│   └── qdrant_store.py     # Qdrant integration
├── metadatastore/
│   ├── base.py             # ImageRecord dataclass + abstract interface
│   ├── json_store.py       # Local JSON file backend (default)
│   └── postgres_store.py   # PostgreSQL backend
├── agents/
│   ├── tool_interface.py   # Generic agent tool
│   ├── mcp_server.py       # MCP server for Claude
│   └── langchain_tool.py   # LangChain tool wrapper
├── data/
│   └── loader.py           # Image loading from folders/CSV/lists
├── search_engine.py        # High-level unified API
└── DeepImageSearch.py       # v2 backward-compatible shim
```

## Examples

Ready-to-run demo scripts in the [`Demo/`](https://github.com/TechyNilesh/DeepImageSearch/tree/main/Demo) folder:

| # | Demo | Description |
|---|---|---|
| 1 | [Basic Image Search](Demo/01_basic_image_search.py) | Index a folder, find similar images, plot results |
| 2 | [Text-to-Image Search](Demo/02_text_to_image_search.py) | Search images with natural language queries |
| 3 | [Hybrid Search](Demo/03_hybrid_search.py) | Combine text + image queries with weight tuning |
| 4 | [Filtered Search](Demo/04_filtered_search.py) | Attach metadata and filter results |
| 5 | [LLM Captioning](Demo/05_llm_captioning.py) | Auto-generate captions with any vision LLM |
| 6 | [Vector Stores](Demo/06_vector_stores.py) | FAISS vs ChromaDB vs Qdrant |
| 7 | [Metadata Storage](Demo/07_metadata_storage.py) | JSON records, PostgreSQL, custom stores |
| 8 | [Agentic Tools](Demo/08_agentic_tools.py) | MCP server, LangChain tool, generic tool |
| 9 | [Embedding Models](Demo/09_embedding_models.py) | Compare CLIP presets and timm models |
| 10 | [Incremental Indexing](Demo/10_incremental_indexing.py) | Add images over time, save/reload |

## Documentation

For detailed documentation: [Read Full Documents](https://github.com/TechyNilesh/DeepImageSearch/blob/main/Documents/Document.md)

## Core Contributors

<a href="https://github.com/TechyNilesh">
  <img src="https://github.com/TechyNilesh.png" width="80" height="80" style="border-radius:50%" alt="Nilesh Verma"/>
  <br/>
  <sub><b>Nilesh Verma</b></sub>
</a>

## Citation

If you use DeepImageSearch in your Research/Product, please cite:

```latex
@misc{TechyNilesh/DeepImageSearch,
  author = {VERMA, NILESH},
  title = {Deep Image Search - AI-Based Image Search Engine},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TechyNilesh/DeepImageSearch}},
}
```

### Please do STAR the repository, if it helped you in any way.

**Feel free to give suggestions, report bugs and contribute.**
