"""
LLM-powered image captioning using OpenAI SDK standard.

Works with ANY provider that supports the OpenAI chat completions format.
User provides base_url + api_key + model. That's it.
"""

import base64
import io
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail for search indexing. Include:\n"
    "1. Main subjects and objects visible\n"
    "2. Actions or activities happening\n"
    "3. Colors, textures, and visual style\n"
    "4. Setting/environment/background\n"
    "5. Any visible text\n"
    "6. Mood or atmosphere\n\n"
    "Keep the description factual and concise (2-4 sentences)."
)

DEFAULT_METADATA_PROMPT = (
    "Analyze this image and return a JSON object with these fields:\n"
    '- "caption": A one-sentence description\n'
    '- "objects": List of main objects/subjects visible\n'
    '- "scene": The type of scene (indoor, outdoor, studio, etc.)\n'
    '- "colors": Dominant colors\n'
    '- "tags": 5-10 relevant search tags\n'
    '- "text_content": Any visible text (empty string if none)\n\n'
    "Return ONLY valid JSON, no markdown."
)


def _image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """Load an image, resize if needed, and return base64-encoded JPEG."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


class Captioner:
    """
    Generate captions and structured metadata for images using any
    OpenAI SDK-compatible vision LLM.

    Just provide model, api_key, and base_url — works with any provider
    (OpenAI, Gemini, Claude, Ollama, Together, Groq, vLLM, etc.).

    Parameters
    ----------
    model : str
        Model name as per your provider's API.
    api_key : str
        API key for the provider.
    base_url : str
        API base URL for the provider.
    max_image_size : int
        Max dimension to resize images before sending (saves tokens/cost).
    max_tokens : int
        Max tokens for the LLM response.

    Examples
    --------
    captioner = Captioner(
        model="your-model-name",
        api_key="your-api-key",
        base_url="https://your-provider.com/v1",
    )
    caption = captioner.caption("photo.jpg")
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_image_size: int = 1024,
        max_tokens: int = 500,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: uv pip install 'DeepImageSearch[llm]'")

        self.model = model
        self.max_image_size = max_image_size
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        logger.info(f"Captioner ready: model={model}, base_url={base_url}")

    def caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a text caption for a single image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str or None
            Custom prompt. Defaults to DEFAULT_CAPTION_PROMPT.

        Returns
        -------
        str
            Generated caption.
        """
        prompt = prompt or DEFAULT_CAPTION_PROMPT
        b64 = _image_to_base64(image_path, self.max_image_size)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.choices[0].message.content

    def caption_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        on_error: str = "skip",
    ) -> Dict[str, str]:
        """
        Generate captions for multiple images.

        Parameters
        ----------
        image_paths : list[str]
            Paths to image files.
        prompt : str or None
            Custom prompt.
        on_error : str
            'skip' to continue on failure, 'raise' to stop on first failure.

        Returns
        -------
        dict[str, str]
            Mapping of image_path -> caption.
        """
        from tqdm import tqdm

        results = {}
        for path in tqdm(image_paths, desc="Captioning images"):
            try:
                results[path] = self.caption(path, prompt)
            except Exception as e:
                logger.error(f"Failed to caption {path}: {e}")
                if on_error == "raise":
                    raise
                results[path] = ""
        return results

    def extract_metadata(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract structured metadata (JSON) from an image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompt : str or None
            Custom JSON extraction prompt. Defaults to DEFAULT_METADATA_PROMPT.

        Returns
        -------
        dict
            Parsed metadata dictionary.
        """
        import json

        prompt = prompt or DEFAULT_METADATA_PROMPT
        raw = self.caption(image_path, prompt)

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON metadata for {image_path}, returning raw text")
            return {"caption": raw, "raw_response": True}
