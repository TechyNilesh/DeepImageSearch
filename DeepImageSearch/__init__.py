"""
DeepImageSearch — AI-powered image search with multimodal embeddings,
text-to-image search, LLM captioning, and agentic RAG integration.

Usage:
    from DeepImageSearch import Load_Data, SearchEngine

    images = Load_Data().from_folder(["./photos"])
    engine = SearchEngine(model_name="clip-vit-b-32")
    engine.index(images)

    # Text search
    results = engine.search("a sunset over mountains")

    # Image search
    results = engine.search("query.jpg")

    # Plot results
    engine.plot_similar_images("query.jpg")
"""

__version__ = "3.0.0"

# Main API
from DeepImageSearch.search_engine import SearchEngine
from DeepImageSearch.data.loader import Load_Data

# Backward-compatible v2 import
from DeepImageSearch.DeepImageSearch import Search_Setup

# Core components (for advanced usage)
from DeepImageSearch.core.embeddings import EmbeddingManager, CLIPEmbedding, TimmEmbedding
from DeepImageSearch.core.indexer import Indexer
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.core.captioner import Captioner

# Vector stores
from DeepImageSearch.vectorstores.faiss_store import FAISSStore

# Metadata stores
from DeepImageSearch.metadatastore.base import ImageRecord, BaseMetadataStore
from DeepImageSearch.metadatastore.json_store import JsonMetadataStore

try:
    from DeepImageSearch.metadatastore.postgres_store import PostgresMetadataStore
except ImportError:
    pass

# Agent tools
from DeepImageSearch.agents.tool_interface import ImageSearchTool

__all__ = [
    # High-level
    "SearchEngine",
    "Load_Data",
    "Search_Setup",
    # Core
    "EmbeddingManager",
    "CLIPEmbedding",
    "TimmEmbedding",
    "Indexer",
    "Searcher",
    "Captioner",
    # Vector stores
    "FAISSStore",
    # Metadata stores
    "ImageRecord",
    "BaseMetadataStore",
    "JsonMetadataStore",
    # Agent tools
    "ImageSearchTool",
]
