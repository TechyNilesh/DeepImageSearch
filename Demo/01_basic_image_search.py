"""
Demo 1: Basic Image-to-Image Search

Index a folder of images and find visually similar images
using CLIP embeddings and FAISS vector search.

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine

# 1. Create search engine with CLIP model
engine = SearchEngine(
    model_name="clip-vit-b-32",   # fast, general purpose
    index_dir="./demo_index",      # where to save index files
)

# 2. Index images from a folder
count = engine.index("./images")
print(f"Indexed {count} images")

# 3. Search by image
results = engine.search_by_image("./images/query.jpg", k=5)

for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']} | {r['metadata']['image_path']}")

# 4. Plot results visually
engine.plot_similar_images("./images/query.jpg", number_of_images=6)

# 5. Check engine info
print(engine.info())
