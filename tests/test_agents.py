# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Nilesh Verma
"""
Tests for the agent-facing wrappers: the generic ImageSearchTool, the LangChain
tool, and the MCP server.

All of them build an embedding backend, so EmbeddingManager.create is patched to
the dummy. The LangChain and MCP tests skip unless their extra is installed.
"""

import json
import sys

import pytest

from DeepImageSearch.agents import tool_interface as ti_module
from DeepImageSearch.agents.tool_interface import ImageSearchTool
from DeepImageSearch.core.indexer import Indexer
from DeepImageSearch.vectorstores.faiss_store import FAISSStore
from tests.conftest import DIM, DummyEmbedding


@pytest.fixture(autouse=True)
def no_model_downloads(monkeypatch):
    monkeypatch.setattr(
        ti_module.EmbeddingManager, "create",
        staticmethod(lambda *args, **kwargs: DummyEmbedding()),
    )


@pytest.fixture
def saved_index(tmp_path, image_paths, embedding):
    """A FAISS index on disk, ready for a tool to load."""
    store = FAISSStore(dimension=DIM)
    Indexer(embedding=embedding, vector_store=store).index(
        image_paths, extra_metadata=[{"album": "trip"} for _ in image_paths]
    )
    index_dir = tmp_path / "index"
    store.save(str(index_dir))
    return str(index_dir)


class TestImageSearchTool:
    def test_loads_a_saved_index(self, saved_index, image_paths):
        tool = ImageSearchTool(index_path=saved_index)
        assert tool.vector_store.count() == len(image_paths)

    def test_is_callable_with_a_text_query(self, saved_index):
        results = ImageSearchTool(index_path=saved_index)(query="a red square", k=2)
        assert len(results) == 2
        assert set(results[0]) == {"id", "score", "metadata"}

    def test_is_callable_with_an_image_path(self, saved_index, image_paths):
        results = ImageSearchTool(index_path=saved_index)(query=image_paths[0], k=1)
        assert results[0]["metadata"]["image_path"] == image_paths[0]

    def test_forwards_filters(self, saved_index):
        tool = ImageSearchTool(index_path=saved_index)
        assert tool(query="a red square", k=5, filters={"album": "trip"})
        assert tool(query="a red square", k=5, filters={"album": "nope"}) == []

    def test_forwards_mode(self, saved_index, image_paths):
        tool = ImageSearchTool(index_path=saved_index)
        # mode='text' must not treat a path-looking string as an image
        assert tool(query=image_paths[0], k=1, mode="text")

    def test_unknown_store_type_raises(self, saved_index):
        with pytest.raises(ValueError, match="Unknown store type"):
            ImageSearchTool(index_path=saved_index, vector_store_type="pinecone")

    def test_tool_definition_is_valid_function_calling_schema(self, saved_index):
        definition = ImageSearchTool(index_path=saved_index).tool_definition
        assert definition["name"] == "search_images"
        assert definition["description"]

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert schema["required"] == ["query"]
        assert set(schema["properties"]) == {"query", "k", "mode"}
        assert schema["properties"]["mode"]["enum"] == ["auto", "text", "image"]
        assert schema["properties"]["k"]["type"] == "integer"

    def test_tool_definition_is_json_serialisable(self, saved_index):
        definition = ImageSearchTool(index_path=saved_index).tool_definition
        assert json.loads(json.dumps(definition)) == definition


class TestLangChainTool:
    @pytest.fixture(autouse=True)
    def requires_langchain(self):
        pytest.importorskip("langchain_core", reason="install the [langchain] extra to run these")

    def test_creates_a_structured_tool(self, saved_index):
        from DeepImageSearch.agents.langchain_tool import create_langchain_tool

        tool = create_langchain_tool(index_path=saved_index)
        assert tool.name == "search_images"
        assert tool.description

    def test_invoking_returns_json_with_path_score_and_caption(self, saved_index, image_paths):
        from DeepImageSearch.agents.langchain_tool import create_langchain_tool

        tool = create_langchain_tool(index_path=saved_index)
        payload = json.loads(tool.invoke({"query": image_paths[0], "k": 2}))
        assert len(payload) == 2
        assert set(payload[0]) == {"image_path", "score", "caption"}
        assert payload[0]["image_path"] == image_paths[0]
        assert isinstance(payload[0]["score"], float)

    def test_args_schema_exposes_query_k_and_mode(self, saved_index):
        from DeepImageSearch.agents.langchain_tool import create_langchain_tool

        fields = create_langchain_tool(index_path=saved_index).args_schema.model_fields
        assert set(fields) == {"query", "k", "mode"}

    def test_default_k_is_configurable(self, saved_index):
        from DeepImageSearch.agents.langchain_tool import create_langchain_tool

        tool = create_langchain_tool(index_path=saved_index, k=3)
        assert tool.args_schema.model_fields["k"].default == 3

    def test_missing_dependency_gives_an_actionable_error(self, saved_index, monkeypatch):
        from DeepImageSearch.agents.langchain_tool import create_langchain_tool

        monkeypatch.setitem(sys.modules, "langchain_core.tools", None)
        with pytest.raises(ImportError, match=r"DeepImageSearch\[langchain\]"):
            create_langchain_tool(index_path=saved_index)


class TestMcpServer:
    @pytest.fixture(autouse=True)
    def requires_mcp(self):
        pytest.importorskip("mcp", reason="install the [mcp] extra to run these")

    def test_creates_a_server_exposing_both_tools(self, saved_index):
        import anyio

        from DeepImageSearch.agents.mcp_server import create_mcp_server

        mcp = create_mcp_server(index_path=saved_index)
        names = {t.name for t in anyio.run(mcp.list_tools)}
        assert names == {"search_images", "get_index_info"}

    def test_search_tool_returns_json_results(self, saved_index, image_paths):
        import anyio

        from DeepImageSearch.agents.mcp_server import create_mcp_server

        mcp = create_mcp_server(index_path=saved_index)
        result = anyio.run(lambda: mcp.call_tool("search_images", {"query": image_paths[0], "k": 2}))
        payload = json.loads(_first_text(result))
        assert len(payload) == 2
        assert payload[0]["image_path"] == image_paths[0]
        # image_path and caption are lifted out of the nested metadata blob
        assert "image_path" not in payload[0]["metadata"]

    def test_info_tool_reports_the_index(self, saved_index, image_paths):
        import anyio

        from DeepImageSearch.agents.mcp_server import create_mcp_server

        mcp = create_mcp_server(index_path=saved_index)
        info = json.loads(_first_text(anyio.run(lambda: mcp.call_tool("get_index_info", {}))))
        assert info["total_images"] == len(image_paths)
        assert info["vector_dimension"] == DIM
        assert info["supports_text_search"] is True
        assert info["vector_store"] == "faiss"

    def test_missing_dependency_gives_an_actionable_error(self, saved_index, monkeypatch):
        from DeepImageSearch.agents.mcp_server import create_mcp_server

        # Block both the mcp >= 2.0 path and the 1.x fallback.
        monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
        monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
        with pytest.raises(ImportError, match=r"DeepImageSearch\[mcp\]"):
            create_mcp_server(index_path=saved_index)

    def test_cli_wires_arguments_through_and_runs(self, saved_index, monkeypatch):
        from DeepImageSearch.agents import mcp_server

        captured = {}

        class FakeServer:
            def run(self):
                captured["ran"] = True

        monkeypatch.setattr(mcp_server, "create_mcp_server", lambda **kwargs: captured.update(kwargs) or FakeServer())
        monkeypatch.setattr(sys, "argv", ["deep-image-search-mcp", "--index-path", saved_index,
                                          "--store-type", "faiss", "--device", "cpu"])
        mcp_server.main()

        assert captured["index_path"] == saved_index
        assert captured["vector_store_type"] == "faiss"
        assert captured["device"] == "cpu"
        assert captured["ran"] is True

    def test_cli_requires_an_index_path(self, monkeypatch):
        from DeepImageSearch.agents import mcp_server

        monkeypatch.setattr(sys, "argv", ["deep-image-search-mcp"])
        with pytest.raises(SystemExit):
            mcp_server.main()


def _first_text(call_tool_result):
    """FastMCP returns (content, ...) or a result object depending on version."""
    content = call_tool_result[0] if isinstance(call_tool_result, tuple) else call_tool_result
    if hasattr(content, "content"):
        content = content.content
    return content[0].text
