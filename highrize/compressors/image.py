"""
Image compressor — reduces image token cost for vision models.

Vision APIs charge based on image resolution (tile-based for GPT-4V,
flat token cost for Claude, etc.). This module:
  1. Resizes images to the minimum useful resolution
  2. Re-encodes as JPEG at configurable quality
  3. Converts RGBA → RGB (PNG alpha → white background)
  4. Returns base64 string or bytes ready for any API

Token cost reference:
  GPT-4V   : tiles of 512×512, ~170 tokens/tile (low detail = 85 tokens flat)
  Claude   : ~1500-2000 tokens for full image, less for smaller
  Gemini   : proportional to pixel count
"""

import base64
import io
from typing import Union, Tuple, Optional
from ..models import CompressionResult, Modality


def _pil_available():
    try:
        import PIL
        return True
    except ImportError:
        return False


def _image_tokens_estimate(width: int, height: int, provider: str = "openai") -> int:
    """Rough token estimate by provider."""
    if provider == "openai":
        # GPT-4V tile model: ceil(w/512) * ceil(h/512) * 170
        import math
        return math.ceil(width / 512) * math.ceil(height / 512) * 170
    elif provider == "anthropic":
        # Claude: approx linear with total pixels
        return int((width * height) / 750)
    else:
        # Generic: pixels / 750
        return int((width * height) / 750)


class ImageCompressor:
    """
    Compresses images to reduce vision API token cost.

    Args:
        max_size: (width, height) max dimensions. Image is resized proportionally.
        quality: JPEG quality 1-95. Lower = fewer bytes, more artifact.
        output_format: "JPEG" or "PNG" (JPEG almost always smaller for photos)
        provider: "openai" | "anthropic" | "gemini" — affects token estimation
        low_detail: If True, adds {"detail": "low"} metadata (OpenAI-specific,
                    forces flat 85-token cost regardless of resolution)
    """

    def __init__(
        self,
        token_counter=None,
        max_size: Tuple[int, int] = (1024, 1024),
        quality: int = 75,
        output_format: str = "JPEG",
        provider: str = "openai",
        low_detail: bool = False,
    ):
        from ..tokens import TokenCounter
        if not _pil_available():
            raise ImportError(
                "Pillow is required for image compression. "
                "Install it with: pip install Pillow"
            )
        self.max_size = max_size
        self.quality = quality
        self.output_format = output_format.upper()
        self.provider = provider
        self.low_detail = low_detail
        self.token_counter = token_counter or TokenCounter(provider=provider)

    def compress(
        self,
        image: Union[str, bytes, "PIL.Image.Image"],  # path, b64 string, bytes, or PIL
    ) -> CompressionResult:
        from PIL import Image

        # --- Load ---
        if isinstance(image, str):
            if image.startswith("data:") or image.startswith("base64,") or len(image) > 512:
                # base64 string
                raw_data = image
                if "," in image:
                    _, _, raw_data = image.partition(",")
                
                try:
                    raw = base64.b64decode(raw_data)
                    img = Image.open(io.BytesIO(raw))
                    original_bytes = raw
                except Exception:
                    # Fallback to file path if b64 decode fails
                    img = Image.open(image)
                    with open(image, "rb") as f:
                        original_bytes = f.read()
            else:
                # file path
                img = Image.open(image)
                with open(image, "rb") as f:
                    original_bytes = f.read()
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
            original_bytes = image
        else:
            img = image
            buf = io.BytesIO()
            img.save(buf, format=self.output_format)
            original_bytes = buf.getvalue()

        original_w, original_h = img.size
        original_tokens = _image_tokens_estimate(original_w, original_h, self.provider)

        # --- Convert RGBA → RGB ---
        if img.mode in ("RGBA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # --- Resize ---
        img.thumbnail(self.max_size, Image.LANCZOS)
        new_w, new_h = img.size

        # --- Encode ---
        buf = io.BytesIO()
        fmt = self.output_format
        save_kwargs = {"format": fmt}
        if fmt == "JPEG":
            save_kwargs["quality"] = self.quality
            save_kwargs["optimize"] = True
        img.save(buf, **save_kwargs)
        compressed_bytes = buf.getvalue()
        b64 = base64.b64encode(compressed_bytes).decode("utf-8")
        compressed_b64 = f"data:image/{fmt.lower()};base64,{b64}"

        compressed_tokens = (
            85 if self.low_detail and self.provider == "openai"
            else _image_tokens_estimate(new_w, new_h, self.provider)
        )

        return CompressionResult(
            original=image,
            compressed=compressed_b64,
            modality=Modality.IMAGE,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_size_bytes=len(original_bytes),
            compressed_size_bytes=len(compressed_bytes),
            metadata={
                "original_size": (original_w, original_h),
                "compressed_size": (new_w, new_h),
                "format": fmt,
                "quality": self.quality,
                "low_detail": self.low_detail,
            },
        )
