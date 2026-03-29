"""
Demo 6: Different Vector Store Backends

Use FAISS (default), ChromaDB, or Qdrant as the vector store.

Requires:
  ChromaDB: uv pip install "DeepImageSearch[chroma]"
  Qdrant:   uv pip install "DeepImageSearch[qdrant]"

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine

# === FAISS (default, local) ===
print("=== FAISS ===")
engine = SearchEngine(
    model_name="clip-vit-b-32",
    vector_store="faiss",
    index_type="flat",            # "flat" (exact), "ivf" (fast), "hnsw" (fastest)
    index_dir="./faiss_index",
)
engine.index("./images")
results = engine.search("sunset", k=3)
print(f"FAISS results: {len(results)} | store: {engine.info()['vector_store']}")


# === ChromaDB (persistent, metadata filtering) ===
print("\n=== ChromaDB ===")
engine = SearchEngine(
    model_name="clip-vit-b-32",
    vector_store="chroma",
    index_dir="./chroma_index",
    chroma_collection="demo_images",
)
engine.index("./images")
results = engine.search("sunset", k=3)
print(f"Chroma results: {len(results)} | store: {engine.info()['vector_store']}")


# === Qdrant (production-grade) ===
print("\n=== Qdrant ===")
engine = SearchEngine(
    model_name="clip-vit-b-32",
    vector_store="qdrant",
    index_dir="./qdrant_index",
)
engine.index("./images")
results = engine.search("sunset", k=3)
print(f"Qdrant results: {len(results)} | store: {engine.info()['vector_store']}")
