"""Vision service for image captioning using GPT-4V."""

import base64
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI


class VisionService:
    """Service for generating image descriptions using GPT-4V."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        """Initialize the vision service.

        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            model: Vision model to use. Default is gpt-4o-mini (cheaper, good quality).
        """
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def caption_image(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        detail: str = "auto",
    ) -> str:
        """Generate a detailed caption for an image.

        Args:
            image_path: Path to the image file.
            image_bytes: Raw image bytes.
            image_base64: Base64-encoded image string.
            detail: Image detail level ("low", "high", "auto").

        Returns:
            Detailed description of the image.

        Raises:
            ValueError: If no image source is provided.
        """
        if image_path:
            image_data = self._encode_image_file(image_path)
            media_type = self._get_media_type(image_path)
        elif image_bytes:
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            media_type = "image/jpeg"  # Default, could be detected
        elif image_base64:
            image_data = image_base64
            media_type = "image/jpeg"
        else:
            raise ValueError("Must provide image_path, image_bytes, or image_base64")

        image_url = f"data:{media_type};base64,{image_data}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail for semantic search. "
                                "Include: main subjects, actions, objects, colors, "
                                "setting/location, text visible, mood/style, and any "
                                "notable details. Be specific and descriptive."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": detail,
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        return response.choices[0].message.content or ""

    def _encode_image_file(self, image_path: str) -> str:
        """Encode an image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_media_type(self, image_path: str) -> str:
        """Get MIME type from file extension."""
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(ext, "image/jpeg")
