"""
LangChain tool wrapper for DeepImageSearch.

Usage:
    from DeepImageSearch.agents.langchain_tool import create_langchain_tool

    tool = create_langchain_tool(index_path="./my_index")
    # Use in a LangChain agent
    agent = create_react_agent(llm, [tool])
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def create_langchain_tool(
    index_path: str,
    model_name: str = "clip-vit-b-32",
    vector_store_type: str = "faiss",
    device: Optional[str] = None,
    k: int = 5,
):
    """
    Create a LangChain-compatible tool for image search.

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
    k : int
        Default number of results.

    Returns
    -------
    langchain_core.tools.StructuredTool
    """
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "langchain-core is required. Install with: uv pip install 'DeepImageSearch[langchain]'"
        )

    from DeepImageSearch.agents.tool_interface import ImageSearchTool

    search_tool = ImageSearchTool(
        index_path=index_path,
        model_name=model_name,
        vector_store_type=vector_store_type,
        device=device,
    )

    class SearchImagesInput(BaseModel):
        query: str = Field(description="Natural language query or image file path")
        k: int = Field(default=k, description="Number of results to return")
        mode: str = Field(default="auto", description="Search mode: 'text', 'image', or 'auto'")

    def _search(query: str, k: int = 5, mode: str = "auto") -> str:
        results = search_tool(query=query, k=k, mode=mode)
        formatted = []
        for r in results:
            formatted.append({
                "image_path": r["metadata"].get("image_path", ""),
                "score": round(r["score"], 4),
                "caption": r["metadata"].get("caption", ""),
            })
        return json.dumps(formatted, indent=2)

    return StructuredTool.from_function(
        func=_search,
        name="search_images",
        description=(
            "Search an indexed image collection using natural language text queries "
            "or image file paths. Returns similar images with similarity scores."
        ),
        args_schema=SearchImagesInput,
    )
