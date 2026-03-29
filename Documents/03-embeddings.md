# Embeddings

DeepImageSearch supports two families of embedding models: CLIP-family (text + image) and timm (image-only).

## CLIP Presets

CLIP-family models embed both text and images into the same vector space, enabling text-to-image search.

| Preset Name | Model | Dimension | Text Search | Best For |
|---|---|---|---|---|
| `clip-vit-b-32` | CLIP ViT-B/32 | 512 | Yes | Fast, general purpose |
| `clip-vit-b-16` | CLIP ViT-B/16 | 512 | Yes | Better accuracy |
| `clip-vit-l-14` | CLIP ViT-L/14 | 768 | Yes | High accuracy |
| `clip-vit-l-14-336` | CLIP ViT-L/14@336 | 768 | Yes | Highest accuracy |
| `clip-vit-bigg-14` | CLIP ViT-bigG/14 | 1280 | Yes | Maximum quality |
| `eva-clip-vit-b-16` | EVA02-B-16 | 512 | Yes | Efficient |
| `siglip-vit-b-16` | SigLIP ViT-B/16 | 512 | Yes | Improved zero-shot |
| `siglip-vit-l-16` | SigLIP ViT-L/16 | 768 | Yes | High quality zero-shot |

```python
from DeepImageSearch import SearchEngine

# Use a preset
engine = SearchEngine(model_name="clip-vit-l-14")
```

### List All Presets

```python
from DeepImageSearch.core.embeddings import EmbeddingManager

presets = EmbeddingManager.list_presets()
for name, (model, weights) in presets.items():
    print(f"{name}: {model} ({weights})")
```

## timm Models (Legacy)

Any model from the [timm library](https://huggingface.co/timm) works for image-only search. Over 500 models available.

```python
engine = SearchEngine(model_name="vgg19")
engine = SearchEngine(model_name="resnet50")
engine = SearchEngine(model_name="efficientnet_b0")
engine = SearchEngine(model_name="vit_base_patch16_224")
```

timm models do NOT support text search. Use CLIP presets for text-to-image.

## Using Embeddings Directly

### EmbeddingManager (Factory)

```python
from DeepImageSearch.core.embeddings import EmbeddingManager

# Create from preset
emb = EmbeddingManager.create("clip-vit-b-32")
emb = EmbeddingManager.create("clip-vit-l-14", device="cuda")

# Create from timm model name
emb = EmbeddingManager.create("vgg19", image_size=224)
```

#### `EmbeddingManager.create()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"clip-vit-b-32"` | Preset name or timm model name. |
| `device` | `str` or `None` | `None` | `"cuda"`, `"mps"`, `"cpu"`, or None (auto). |
| `batch_size` | `int` | `64` | Batch size for extraction. |
| `**kwargs` | | | Passed to underlying model (e.g. `pretrained`, `image_size`). |

### CLIPEmbedding

```python
from DeepImageSearch.core.embeddings import CLIPEmbedding

emb = CLIPEmbedding(
    model_name="ViT-B-32",    # open_clip model name
    pretrained="openai",       # pretrained weights tag
    device="cuda",
    batch_size=64,
)

# Embed images
from PIL import Image
img = Image.open("photo.jpg")
vector = emb.embed_image(img)         # single image -> 1D array
vectors = emb.embed_images([img])     # batch -> (N, D) array

# Embed text
vector = emb.embed_text("a red car")
vectors = emb.embed_texts(["a red car", "a blue boat"])
```

#### Properties

- `emb.dimension` -- embedding dimension (e.g. 512)
- `emb.supports_text` -- `True` for CLIP, `False` for timm

### TimmEmbedding

```python
from DeepImageSearch.core.embeddings import TimmEmbedding

emb = TimmEmbedding(
    model_name="vgg19",
    pretrained=True,
    image_size=224,
    device="cpu",
    batch_size=64,
)

vector = emb.embed_image(img)  # image only, no text support
```

## Device Selection

The library auto-detects the best available device:

1. **CUDA** -- if `torch.cuda.is_available()`
2. **MPS** -- if on Apple Silicon with `torch.backends.mps.is_available()`
3. **CPU** -- fallback

Override with `device="cuda"`, `device="mps"`, or `device="cpu"`.
