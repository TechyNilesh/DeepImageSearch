# LLM Captioning

DeepImageSearch can auto-generate rich text captions for images during indexing using any vision LLM that supports the OpenAI SDK chat completions format.

## Setup

Provide three things: `model`, `api_key`, `base_url`.

```python
from DeepImageSearch import SearchEngine

engine = SearchEngine(
    model_name="clip-vit-b-32",
    captioner_model="your-model-name",
    captioner_api_key="your-api-key",
    captioner_base_url="https://your-provider.com/v1",
)

# Index with auto-captioning
engine.index("./photos", generate_captions=True)
```

Captions are stored in both the vector store metadata and the image records. They improve text search quality by providing richer descriptions for each image.

## Supported Providers

Any provider with an OpenAI SDK-compatible chat completions endpoint works. Set the `base_url` accordingly.

## Using the Captioner Directly

```python
from DeepImageSearch.core.captioner import Captioner

captioner = Captioner(
    model="your-model-name",
    api_key="your-api-key",
    base_url="https://your-provider.com/v1",
    max_image_size=1024,  # resize before sending (saves tokens)
    max_tokens=500,       # max response length
)
```

### `Captioner.__init__()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Model name as per your provider's API. |
| `api_key` | `str` | required | API key for the provider. |
| `base_url` | `str` | required | API base URL for the provider. |
| `max_image_size` | `int` | `1024` | Max pixel dimension before sending. |
| `max_tokens` | `int` | `500` | Max tokens for LLM response. |

### Caption a Single Image

```python
caption = captioner.caption("photo.jpg")
print(caption)
# "A golden retriever playing fetch in a park with green trees"
```

### Custom Prompt

```python
caption = captioner.caption("photo.jpg", prompt="Describe in one sentence.")
```

### Batch Captioning

```python
captions = captioner.caption_batch(
    ["img1.jpg", "img2.jpg", "img3.jpg"],
    on_error="skip",  # or "raise"
)
# {"img1.jpg": "A sunset...", "img2.jpg": "A cat...", "img3.jpg": ""}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image_paths` | `List[str]` | required | Paths to image files. |
| `prompt` | `str` or `None` | `None` | Custom prompt for all images. |
| `on_error` | `str` | `"skip"` | `"skip"` continues on failure, `"raise"` stops. |

**Returns:** `Dict[str, str]` -- mapping of image_path to caption.

### Extract Structured Metadata

Returns a parsed JSON dict with objects, scene, colors, tags, etc.

```python
metadata = captioner.extract_metadata("photo.jpg")
# {
#     "caption": "A golden retriever in a park",
#     "objects": ["dog", "ball", "tree"],
#     "scene": "outdoor",
#     "colors": ["green", "brown", "gold"],
#     "tags": ["dog", "park", "playing", "fetch", "outdoor", "pet"],
#     "text_content": ""
# }
```

You can provide a custom JSON extraction prompt:

```python
metadata = captioner.extract_metadata("photo.jpg", prompt="Return JSON with: caption, mood, style")
```

## How Captioning Works with Search

When `generate_captions=True` is passed to `engine.index()`:

1. Each image is sent to the vision LLM with a descriptive prompt
2. The returned caption is stored in the image's metadata (`caption` field)
3. The caption is included in the vector store metadata for filtering
4. The caption is stored in the `ImageRecord` for retrieval

Captions enrich search by providing semantic text descriptions that the CLIP model can match against text queries.
