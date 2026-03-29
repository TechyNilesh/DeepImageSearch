# Getting Started

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

### GPU Support

The library auto-detects CUDA, MPS (Apple Silicon), or falls back to CPU. If using a GPU for FAISS, uninstall `faiss-cpu` and install `faiss-gpu`:

```shell
pip uninstall faiss-cpu && pip install faiss-gpu
```

## Quick Start

```python
from DeepImageSearch import SearchEngine

# Create engine with CLIP model (enables text + image search)
engine = SearchEngine(model_name="clip-vit-b-32")

# Index images from a folder
engine.index("./photos")

# Text search
results = engine.search("a sunset over mountains")

# Image search
results = engine.search("query.jpg")

# Hybrid search
results = engine.search("outdoor scene", image_query="photo.jpg", mode="hybrid")

# Plot results
engine.plot_similar_images("query.jpg", number_of_images=9)
```

## Core Concepts

### Embeddings

DeepImageSearch converts images (and text) into high-dimensional vectors using neural network models. CLIP-family models can embed both text and images into the same vector space, enabling text-to-image search.

### Vector Store

Vectors are stored in a vector index (FAISS, ChromaDB, or Qdrant) for fast nearest-neighbor search. When you call `engine.search()`, it converts your query to a vector and finds the closest matches.

### Metadata Store

Every indexed image is tracked as an `ImageRecord` with fields like `image_id`, `image_index`, `image_name`, `image_path`, `caption`, and `indexed_at`. Records are stored in a local JSON file by default, or PostgreSQL for production.

### Search Modes

| Mode | Query Type | Requires |
|---|---|---|
| `text` | Natural language string | CLIP model |
| `image` | Image file path or PIL Image | Any model |
| `hybrid` | Text + image combined | CLIP model |
| `auto` | Auto-detects text vs image | Depends on query |

## Understanding Search Results

Every search result is a dictionary:

```python
{
    "id": "a1b2c3d4...",           # unique image ID (MD5 hash)
    "score": 0.87,                  # similarity score (0-1, higher is better)
    "metadata": {
        "image_id": "a1b2c3d4...",
        "image_index": 42,          # position in the index
        "image_name": "sunset.jpg", # original filename
        "image_path": "/data/photos/sunset.jpg",
        "caption": "A sunset over mountains",
        "indexed_at": "2026-03-28T10:30:00+00:00"
    }
}
```

## Next Steps

- [Search Engine](./02-search-engine.md) -- full indexing and search API
- [Embeddings](./03-embeddings.md) -- choosing the right model
- [LLM Captioning](./04-captioning.md) -- auto-generate image descriptions
