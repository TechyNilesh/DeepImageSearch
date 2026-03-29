"""
Demo 8: Agentic Integration (MCP, LangChain, Generic Tool)

Use DeepImageSearch as a tool in LLM agent pipelines.

Requires:
  MCP:       uv pip install "DeepImageSearch[mcp]"
  LangChain: uv pip install "DeepImageSearch[langchain]"

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

# --- First, create an index to use with tools ---
from DeepImageSearch import SearchEngine

engine = SearchEngine(model_name="clip-vit-b-32", index_dir="./demo_index")
engine.index("./images")
engine.save()


# === Generic Tool (works with any framework) ===
from DeepImageSearch import ImageSearchTool

tool = ImageSearchTool(index_path="./demo_index")

# Call directly
results = tool("a photo of a dog", k=5)
print("=== Generic Tool Results ===")
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']}")

# Get tool definition for function calling (Claude/OpenAI compatible)
print(f"\nTool definition: {tool.tool_definition['name']}")


# === LangChain Tool ===
# from DeepImageSearch.agents.langchain_tool import create_langchain_tool
#
# lc_tool = create_langchain_tool(index_path="./demo_index", k=5)
#
# # Use in a LangChain agent
# from langchain.agents import create_react_agent
# agent = create_react_agent(llm, [lc_tool])
# response = agent.invoke({"input": "Find images of cats"})


# === MCP Server ===
# Run from command line:
#   deep-image-search-mcp --index-path ./demo_index
#
# Or programmatically:
# from DeepImageSearch.agents.mcp_server import create_mcp_server
# mcp = create_mcp_server(index_path="./demo_index")
# mcp.run()
