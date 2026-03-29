"""
Demo 9: Different Embedding Models

Compare CLIP presets and timm legacy models.

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine
from DeepImageSearch.core.embeddings import EmbeddingManager

# === List available CLIP presets ===
print("=== Available CLIP Presets ===")
for name, (model, weights) in EmbeddingManager.list_presets().items():
    print(f"  {name:25s} -> {model} ({weights})")

# === CLIP ViT-B/32 (fast, general purpose) ===
print("\n=== clip-vit-b-32 ===")
engine = SearchEngine(model_name="clip-vit-b-32", index_dir="./idx_b32")
engine.index("./images")
results = engine.search("a dog playing", k=3)
for r in results:
    print(f"  {r['score']:.4f} | {r['metadata']['image_name']}")
print(f"  Dimension: {engine.embedding.dimension}, Text: {engine.supports_text_search}")

# === CLIP ViT-L/14 (high accuracy) ===
print("\n=== clip-vit-l-14 ===")
engine = SearchEngine(model_name="clip-vit-l-14", index_dir="./idx_l14")
engine.index("./images")
results = engine.search("a dog playing", k=3)
for r in results:
    print(f"  {r['score']:.4f} | {r['metadata']['image_name']}")
print(f"  Dimension: {engine.embedding.dimension}, Text: {engine.supports_text_search}")

# === timm VGG19 (legacy, image-only) ===
print("\n=== vgg19 (timm, no text search) ===")
engine = SearchEngine(model_name="vgg19", index_dir="./idx_vgg19")
engine.index("./images")
results = engine.search_by_image("./images/query.jpg", k=3)
for r in results:
    print(f"  {r['score']:.4f} | {r['metadata']['image_name']}")
print(f"  Dimension: {engine.embedding.dimension}, Text: {engine.supports_text_search}")

# === Using embeddings directly ===
print("\n=== Direct Embedding Usage ===")
from PIL import Image

emb = EmbeddingManager.create("clip-vit-b-32")

img = Image.open("./images/query.jpg")
img_vector = emb.embed_image(img)
txt_vector = emb.embed_text("a photo of a dog")

print(f"  Image vector shape: {img_vector.shape}")
print(f"  Text vector shape:  {txt_vector.shape}")

# Cosine similarity
import numpy as np
similarity = np.dot(img_vector, txt_vector)
print(f"  Cosine similarity:  {similarity:.4f}")
