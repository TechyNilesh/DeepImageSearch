# Agentic Integration

DeepImageSearch can be used as a tool by LLM agents via MCP, LangChain, or a generic callable interface.

## MCP Server

The MCP (Model Context Protocol) server exposes image search as tools that Claude Code, Claude Desktop, or any MCP client can use.

### Installation

```shell
uv pip install "DeepImageSearch[mcp]"
```

### Running the Server

```shell
# Basic usage
deep-image-search-mcp --index-path ./my_index

# With options
deep-image-search-mcp --index-path ./my_index --model clip-vit-l-14 --store-type faiss --device cuda
```

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--index-path` | required | Path to the saved vector store index. |
| `--model` | `"clip-vit-b-32"` | Embedding model preset. |
| `--store-type` | `"faiss"` | `"faiss"`, `"chroma"`, or `"qdrant"`. |
| `--device` | `None` | Device for inference. |

### Claude Desktop Configuration

Add to your Claude Desktop `claude_desktop_config.json`:

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

### Available MCP Tools

The server exposes two tools:

#### `search_images`

Search the indexed image collection.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Text query or image file path. |
| `k` | `int` | `5` | Number of results. |
| `mode` | `str` | `"auto"` | `"text"`, `"image"`, or `"auto"`. |

Returns JSON with image paths, scores, captions, and metadata.

#### `get_index_info`

Returns information about the index: total images, model, dimension, text search support.

### Programmatic Usage

```python
from DeepImageSearch.agents.mcp_server import create_mcp_server

mcp = create_mcp_server(
    index_path="./my_index",
    model_name="clip-vit-l-14",
    vector_store_type="faiss",
    device=None,
)
mcp.run()
```

## LangChain Tool

Create a LangChain-compatible `StructuredTool` for use in agent pipelines.

### Installation

```shell
uv pip install "DeepImageSearch[langchain]"
```

### Usage

```python
from DeepImageSearch.agents.langchain_tool import create_langchain_tool

tool = create_langchain_tool(
    index_path="./my_index",
    model_name="clip-vit-b-32",
    vector_store_type="faiss",
    device=None,
    k=5,
)

# Use in a LangChain agent
from langchain.agents import create_react_agent
agent = create_react_agent(llm, [tool])
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `index_path` | `str` | required | Path to saved vector store. |
| `model_name` | `str` | `"clip-vit-b-32"` | Embedding model preset. |
| `vector_store_type` | `str` | `"faiss"` | Vector store type. |
| `device` | `str` or `None` | `None` | Device for inference. |
| `k` | `int` | `5` | Default number of results. |

**Returns:** `langchain_core.tools.StructuredTool`

The tool accepts: `query` (str), `k` (int), `mode` (str). Returns JSON string.

## Generic Tool Interface

`ImageSearchTool` is a framework-agnostic callable that works with any agent system.

```python
from DeepImageSearch import ImageSearchTool

tool = ImageSearchTool(
    index_path="./my_index",
    model_name="clip-vit-b-32",
    vector_store_type="faiss",
    device=None,
)

# Call directly
results = tool("a photo of a dog", k=5)
results = tool("query.jpg", k=10, mode="image")
```

#### `__call__()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Text query or image file path. |
| `k` | `int` | `5` | Number of results. |
| `mode` | `str` | `"auto"` | `"auto"`, `"text"`, or `"image"`. |
| `filters` | `Dict` or `None` | `None` | Metadata filters. |

**Returns:** `List[Dict]` with `id`, `score`, `metadata`.

### Tool Definition

Get a function calling definition compatible with Claude or OpenAI:

```python
definition = tool.tool_definition
# {
#     "name": "search_images",
#     "description": "Search an indexed image collection...",
#     "input_schema": { ... }
# }
```

Use this to register the tool in custom agent loops.
