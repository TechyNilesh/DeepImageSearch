"""
Demo 10: Incremental Indexing and Persistence

Build an index over time: index a batch, save, add more later,
reload from disk. Shows the full lifecycle.

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine, Load_Data

# === Step 1: Initial indexing ===
print("=== Step 1: Initial Index ===")
engine = SearchEngine(model_name="clip-vit-b-32", index_dir="./incremental_index")

images = Load_Data().from_folder(["./images/batch1"])
count = engine.index(images)
print(f"Indexed: {count} images | Total: {engine.count}")

# === Step 2: Add more images later ===
print("\n=== Step 2: Add More Images ===")
new_images = Load_Data().from_folder(["./images/batch2"])
count = engine.add_images(new_images)
print(f"Added: {count} images | Total: {engine.count}")

# Can also add from a folder path directly
engine.add_images("./images/batch3")
print(f"After batch3 | Total: {engine.count}")

# === Step 3: Check records ===
print("\n=== Step 3: Image Records ===")
records = engine.get_records()
for r in records[:5]:
    print(f"  [{r['image_index']}] {r['image_name']}")
print(f"  ... total {len(records)} records")

# === Step 4: Save and reload ===
print("\n=== Step 4: Save & Reload ===")
engine.save()
print(f"Saved to ./incremental_index/")

# Simulate restart: create new engine, it auto-loads
engine2 = SearchEngine(model_name="clip-vit-b-32", index_dir="./incremental_index")
print(f"Reloaded: {engine2.count} images")

# Search works on reloaded engine
results = engine2.search("a cat", k=3)
for r in results:
    print(f"  {r['score']:.4f} | {r['metadata']['image_name']}")

# === Step 5: Engine info ===
print("\n=== Engine Info ===")
info = engine2.info()
for k, v in info.items():
    print(f"  {k}: {v}")
