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

### CustomEmbedding

Use any custom model (ONNX, TorchScript, TensorFlow, sklearn, etc.) as a feature extractor. Just pass a callable that takes a PIL Image and returns a feature vector.

```python
from DeepImageSearch import CustomEmbedding

# Example: ONNX model
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("my_model.onnx")

def onnx_extractor(img):
    arr = np.array(img.resize((224, 224))).astype(np.float32)
    arr = arr.transpose(2, 0, 1)[None] / 255.0
    return session.run(None, {"input": arr})[0].flatten()

emb = CustomEmbedding(extract_fn=onnx_extractor, dimension=512, name="my-onnx-model")
```

```python
# Example: TorchScript model
import torch

scripted_model = torch.jit.load("model.pt")
scripted_model.eval()

def torchscript_extractor(img):
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        return scripted_model(t(img).unsqueeze(0)).squeeze().numpy()

emb = CustomEmbedding(extract_fn=torchscript_extractor, dimension=512)
```

```python
# Example: TensorFlow / Keras model
import tensorflow as tf
import numpy as np

tf_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")

def tf_extractor(img):
    arr = np.array(img.resize((224, 224))).astype(np.float32)[None]
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return tf_model.predict(arr, verbose=0).flatten()

emb = CustomEmbedding(extract_fn=tf_extractor, dimension=2048, name="resnet50-tf")
```

#### Using CustomEmbedding with SearchEngine

```python
from DeepImageSearch import SearchEngine, CustomEmbedding
from DeepImageSearch.core.indexer import Indexer
from DeepImageSearch.core.searcher import Searcher
from DeepImageSearch.vectorstores.faiss_store import FAISSStore

emb = CustomEmbedding(extract_fn=my_extractor, dimension=512)
store = FAISSStore(dimension=512)

indexer = Indexer(embedding=emb, vector_store=store)
searcher = Searcher(embedding=emb, vector_store=store)

indexer.index(image_paths)
results = searcher.search_by_image("query.jpg", k=5)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `extract_fn` | `callable` | required | Function: PIL Image -> feature vector (numpy array, list, or tensor) |
| `dimension` | `int` | required | Output dimension of the feature vector |
| `name` | `str` | `"custom"` | Name for logging and metadata |

Properties: `dimension: int`, `supports_text: bool = False`

## Device Selection

The library auto-detects the best available device:

1. **CUDA** -- if `torch.cuda.is_available()`
2. **MPS** -- if on Apple Silicon with `torch.backends.mps.is_available()`
3. **CPU** -- fallback

Override with `device="cuda"`, `device="mps"`, or `device="cpu"`.
