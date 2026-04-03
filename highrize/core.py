"""
HighRize core — orchestrates all compressors.
Auto-detects modality from input type and routes accordingly.
"""

import os
import base64
import re
from typing import Any, Dict, List, Optional, Union

from .models import CompressionResult, Modality, SavingsReport
from .tokens import TokenCounter
from .compressors.text import TextCompressor
from .compressors.image import ImageCompressor
from .compressors.video import VideoCompressor
from .compressors.audio import AudioCompressor
from .compressors.document import DocumentCompressor


# Cost per 1K input tokens by provider/model (approximate, update as needed)
COST_TABLE = {
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.00015,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.0005,
    "claude-3-5-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "claude-3-opus": 0.015,
    "gemini-1.5-pro": 0.00125,
    "gemini-1.5-flash": 0.000075,
    "default": 0.002,
}


def _detect_modality(content: Any) -> Modality:
    """Auto-detect what kind of content this is."""
    if isinstance(content, str):
        # File path?
        if os.path.exists(content):
            ext = content.lower().rsplit(".", 1)[-1]
            image_exts = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"}
            video_exts = {"mp4", "avi", "mov", "mkv", "webm", "flv"}
            audio_exts = {"mp3", "wav", "m4a", "ogg", "flac", "aac"}
            doc_exts = {"pdf", "docx", "doc", "txt", "html", "htm", "md"}
            if ext in image_exts:
                return Modality.IMAGE
            if ext in video_exts:
                return Modality.VIDEO
            if ext in audio_exts:
                return Modality.AUDIO
            if ext in doc_exts:
                return Modality.DOCUMENT
        # Base64 image?
        if content.startswith("data:image/"):
            return Modality.IMAGE
        # HTML?
        if re.match(r"\s*<!?[Dd][Oo][Cc]|<html|<HTML", content):
            return Modality.DOCUMENT
        # Default: plain text
        return Modality.TEXT

    elif isinstance(content, bytes):
        # Check magic bytes
        if content[:4] in (b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1"):
            return Modality.IMAGE  # JPEG
        if content[:8] == b"\x89PNG\r\n\x1a\n":
            return Modality.IMAGE  # PNG
        if content[:4] == b"%PDF":
            return Modality.DOCUMENT
        return Modality.TEXT

    elif hasattr(content, "mode"):  # PIL Image
        return Modality.IMAGE

    elif isinstance(content, list):
        # Likely a messages list
        return Modality.TEXT

    return Modality.TEXT


class HighRize:
    """
    Universal AI token compressor.

    Usage:
        tp = HighRize(model="gpt-4o", provider="openai")

        # Single item
        result = tp.compress("Your very long prompt here...")
        print(result)  # tokens saved, compressed text

        # Messages list (OpenAI-style)
        messages = [{"role": "user", "content": "..."}]
        compressed_messages, report = tp.compress_messages(messages)

        # Print savings
        print(tp.report.summary())
    """

    def __init__(
        self,
        model: str = "default",
        provider: str = "openai",
        # Text options
        remove_fillers: bool = True,
        deduplicate: bool = True,
        max_examples: Optional[int] = None,
        # Image options
        max_image_size: tuple = (1024, 1024),
        image_quality: int = 75,
        low_detail_images: bool = False,
        # Video options
        max_frames: int = 10,
        # Audio options
        audio_backend: str = "whisper_local",
        audio_model: str = "base",
        # Document options
        doc_token_budget: int = 2000,
        doc_query: Optional[str] = None,
    ):
        self.model = model
        self.provider = provider
        self.token_counter = TokenCounter(model=model, provider=provider)
        self.cost_per_1k = COST_TABLE.get(model, COST_TABLE["default"])
        self.report = SavingsReport()

        # Initialize compressors
        self._text = TextCompressor(
            token_counter=self.token_counter,
            remove_fillers=remove_fillers,
            deduplicate=deduplicate,
            max_examples=max_examples,
        )
        self._image_kwargs = dict(
            token_counter=self.token_counter,
            max_size=max_image_size,
            quality=image_quality,
            provider=provider,
            low_detail=low_detail_images,
        )
        self._video_kwargs = dict(
            token_counter=self.token_counter,
            max_frames=max_frames,
            frame_size=max_image_size,
            quality=image_quality,
            provider=provider,
        )
        self._audio_kwargs = dict(
            token_counter=self.token_counter,
            backend=audio_backend,
            model_size=audio_model,
        )
        self._doc_kwargs = dict(
            token_counter=self.token_counter,
            token_budget=doc_token_budget,
            query=doc_query,
        )

    def compress(self, content: Any, modality: Optional[Modality] = None) -> CompressionResult:
        """
        Compress any single content item.
        Modality is auto-detected if not specified.
        """
        if modality is None:
            modality = _detect_modality(content)

        result = self._route(content, modality)
        self.report.add(result, self.cost_per_1k)
        return result

    def compress_messages(
        self, messages: List[Dict], query: Optional[str] = None
    ) -> tuple:
        """
        Compress an OpenAI-style messages list in-place.

        Returns:
            (compressed_messages, SavingsReport)
        """
        compressed = []
        for msg in messages:
            new_msg = dict(msg)
            content = msg.get("content", "")

            if isinstance(content, str):
                result = self.compress(content, Modality.TEXT)
                new_msg["content"] = result.compressed

            elif isinstance(content, list):
                # Multi-modal message parts
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type", "text")
                        if ptype == "text":
                            result = self.compress(part.get("text", ""), Modality.TEXT)
                            new_parts.append({**part, "text": result.compressed})
                        elif ptype in ("image_url", "image"):
                            url = (
                                part.get("image_url", {}).get("url", "")
                                or part.get("source", {}).get("data", "")
                            )
                            if url:
                                result = self.compress(url, Modality.IMAGE)
                                if ptype == "image_url":
                                    new_parts.append({
                                        **part,
                                        "image_url": {
                                            **part.get("image_url", {}),
                                            "url": result.compressed,
                                        }
                                    })
                                elif ptype == "image":
                                    # Anthropic format: source: {type: 'base64', data: '...', media_type: '...'}
                                    source = part.get("source", {})
                                    data = source.get("data", "")
                                    if data:
                                        result = self.compress(data, Modality.IMAGE)
                                        # Result will be 'data:image/xxx;base64,yyy'
                                        # Need to strip the header for Anthropic
                                        _, _, raw_b64 = result.compressed.partition(",")
                                        new_parts.append({
                                            **part,
                                            "source": {
                                                **source,
                                                "data": raw_b64 or result.compressed
                                            }
                                        })
                                    else:
                                        new_parts.append(part)
                            else:
                                new_parts.append(part)
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                new_msg["content"] = new_parts

            compressed.append(new_msg)

        return compressed, self.report

    def _route(self, content: Any, modality: Modality) -> CompressionResult:
        if modality == Modality.TEXT:
            return self._text.compress(str(content))

        elif modality == Modality.IMAGE:
            img = ImageCompressor(**self._image_kwargs)
            return img.compress(content)

        elif modality == Modality.VIDEO:
            vid = VideoCompressor(**self._video_kwargs)
            return vid.compress(content)

        elif modality == Modality.AUDIO:
            aud = AudioCompressor(**self._audio_kwargs)
            return aud.compress(content)

        elif modality == Modality.DOCUMENT:
            doc = DocumentCompressor(**self._doc_kwargs)
            return doc.compress(content)

        else:
            # Fallback to text
            return self._text.compress(str(content))

    def reset_report(self):
        """Reset the savings tracker."""
        self.report = SavingsReport()
