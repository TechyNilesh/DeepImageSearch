"""
Demo 4: Search with Metadata Filtering

Index images with custom metadata and filter search results
by metadata fields (source, category, date, etc.).

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine, Load_Data

# 1. Create engine
engine = SearchEngine(model_name="clip-vit-b-32", index_dir="./demo_index")

# 2. Load images and prepare metadata
images = Load_Data().from_folder(["./images"])

# Attach metadata to each image
metadata = []
for i, path in enumerate(images):
    metadata.append({
        "source": "camera" if i % 2 == 0 else "web",
        "category": "animal" if i % 3 == 0 else "landscape",
    })

# 3. Index with metadata
engine.index(images, metadata=metadata)

# 4. Search with filters
print("=== All results for 'dog' ===")
results = engine.search("dog", k=5)
for r in results:
    print(f"  {r['metadata']['image_name']} | source={r['metadata'].get('source')}")

print("\n=== Only 'camera' source ===")
results = engine.search("dog", k=5, filters={"source": "camera"})
for r in results:
    print(f"  {r['metadata']['image_name']} | source={r['metadata'].get('source')}")

print("\n=== Only 'animal' category ===")
results = engine.search("dog", k=5, filters={"category": "animal"})
for r in results:
    print(f"  {r['metadata']['image_name']} | category={r['metadata'].get('category')}")
