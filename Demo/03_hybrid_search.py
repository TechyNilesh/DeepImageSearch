"""
Demo 3: Hybrid Search (Text + Image)

Combine a text query with an image query using weighted fusion.
Useful when you want "images like this one, but specifically showing X".

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine

# 1. Create and index
engine = SearchEngine(model_name="clip-vit-b-32", index_dir="./demo_index")
engine.index("./images")

# 2. Hybrid search: text + image combined
#    "Find images similar to this photo, but specifically outdoor scenes"
results = engine.search(
    "outdoor nature scene",          # text query
    image_query="./images/ref.jpg",  # image query
    mode="hybrid",
    text_weight=0.6,                 # 60% text, 40% image
    k=5,
)

print("=== Hybrid Search Results (60% text, 40% image) ===")
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']}")

# 3. Try different weights
print("\n=== More image-weighted (20% text, 80% image) ===")
results = engine.search(
    "outdoor nature scene",
    image_query="./images/ref.jpg",
    mode="hybrid",
    text_weight=0.2,
    k=5,
)
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']}")
