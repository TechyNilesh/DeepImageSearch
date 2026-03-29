"""
Demo 5: LLM-Powered Image Captioning

Auto-generate rich text captions during indexing using any
OpenAI SDK-compatible vision LLM (OpenAI, Gemini, Claude, Ollama, etc.).

Requires: uv pip install "DeepImageSearch[llm]"

Sample Dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
"""

from DeepImageSearch import SearchEngine

# 1. Create engine with captioning enabled
#    Replace with your provider's model, api_key, and base_url
engine = SearchEngine(
    model_name="clip-vit-b-32",
    index_dir="./demo_captioned_index",
    captioner_model="your-model-name",
    captioner_api_key="your-api-key",
    captioner_base_url="https://your-provider.com/v1",
)

# 2. Index with auto-captioning
count = engine.index("./images", generate_captions=True)
print(f"Indexed {count} images with captions")

# 3. View generated captions
records = engine.get_records()
for r in records[:5]:
    print(f"  {r['image_name']}: {r.get('caption', 'N/A')}")

# 4. Search -- captions are now part of metadata
results = engine.search("person holding umbrella", k=5)
for r in results:
    print(f"  Score: {r['score']:.4f} | {r['metadata']['image_name']}")
    print(f"    Caption: {r['metadata'].get('caption', 'N/A')}")

# --- Using Captioner directly ---
from DeepImageSearch.core.captioner import Captioner

captioner = Captioner(
    model="your-model-name",
    api_key="your-api-key",
    base_url="https://your-provider.com/v1",
)

# Single caption
caption = captioner.caption("./images/photo.jpg")
print(f"\nCaption: {caption}")

# Structured metadata extraction (returns JSON)
metadata = captioner.extract_metadata("./images/photo.jpg")
print(f"Objects: {metadata.get('objects', [])}")
print(f"Tags: {metadata.get('tags', [])}")
