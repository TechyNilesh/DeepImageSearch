"""
Demo 2: Text-to-Image Search

Search images using natural language queries.
Requires a CLIP-family model (supports both text and image embeddings).

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine

# 1. Create engine with CLIP (text search requires CLIP model)
engine = SearchEngine(
    model_name="clip-vit-b-32",
    index_dir="./demo_index",
)

# 2. Index images
engine.index("./images")

# 3. Text-to-image search
print("=== Search: 'a cat sitting on a couch' ===")
results = engine.search("a cat sitting on a couch", k=5)
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']}")

print("\n=== Search: 'sunset over the ocean' ===")
results = engine.search("sunset over the ocean", k=5)
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']}")

print("\n=== Search: 'a person riding a bicycle' ===")
results = engine.search("a person riding a bicycle", k=5)
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']}")

# 4. Verify text search is supported
print(f"\nText search supported: {engine.supports_text_search}")
