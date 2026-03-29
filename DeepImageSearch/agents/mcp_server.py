"""
MCP (Model Context Protocol) server for DeepImageSearch.

Exposes image search as tools that any MCP-compatible client
(Claude Code, Claude Desktop, etc.) can use.

Run with:
    deep-image-search-mcp --index-path ./my_index
    # or
    uv run deep-image-search-mcp --index-path ./my_index
"""

import argparse
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_mcp_server(
    index_path: str,
    model_name: str = "clip-vit-b-32",
    vector_store_type: str = "faiss",
    device: str | None = None,
):
    """
    Create and return an MCP server with image search tools.

    Parameters
    ----------
    index_path : str
        Path to the saved vector store index.
    model_name : str
        Embedding model preset.
    vector_store_type : str
        Vector store backend type.
    device : str or None
        Device for inference.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "mcp is required for the MCP server. Install with: uv pip install 'DeepImageSearch[mcp]'"
        )

    from DeepImageSearch.agents.tool_interface import ImageSearchTool

    # Initialize search tool
    search_tool = ImageSearchTool(
        index_path=index_path,
        model_name=model_name,
        vector_store_type=vector_store_type,
        device=device,
    )

    mcp = FastMCP("DeepImageSearch")

    @mcp.tool()
    def search_images(query: str, k: int = 5, mode: str = "auto") -> str:
        """
        Search an indexed image collection.

        Args:
            query: Natural language description (e.g. 'a red car') or image file path.
            k: Number of results to return.
            mode: 'text' for semantic search, 'image' for visual similarity, 'auto' to detect.

        Returns:
            JSON string with search results including image paths and similarity scores.
        """
        results = search_tool(query=query, k=k, mode=mode)
        # Format for LLM consumption
        formatted = []
        for r in results:
            formatted.append({
                "image_path": r["metadata"].get("image_path", ""),
                "score": round(r["score"], 4),
                "caption": r["metadata"].get("caption", ""),
                "metadata": {k: v for k, v in r["metadata"].items() if k not in ("image_path", "caption")},
            })
        return json.dumps(formatted, indent=2)

    @mcp.tool()
    def get_index_info() -> str:
        """Get information about the current image search index."""
        count = search_tool.vector_store.count()
        supports_text = search_tool.embedding.supports_text
        return json.dumps({
            "total_images": count,
            "model": model_name,
            "supports_text_search": supports_text,
            "vector_dimension": search_tool.embedding.dimension,
            "vector_store": vector_store_type,
        }, indent=2)

    return mcp


def main():
    parser = argparse.ArgumentParser(description="DeepImageSearch MCP Server")
    parser.add_argument("--index-path", required=True, help="Path to the vector store index")
    parser.add_argument("--model", default="clip-vit-b-32", help="Embedding model preset")
    parser.add_argument("--store-type", default="faiss", choices=["faiss", "chroma", "qdrant"])
    parser.add_argument("--device", default=None, help="Device (cuda, mps, cpu)")
    args = parser.parse_args()

    mcp = create_mcp_server(
        index_path=args.index_path,
        model_name=args.model,
        vector_store_type=args.store_type,
        device=args.device,
    )
    mcp.run()


if __name__ == "__main__":
    main()
