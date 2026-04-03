"""
highrize — Universal AI token & cost compressor.
Works with any API (OpenAI, Anthropic, Gemini, Ollama, custom)
and any modality (text, image, video, audio, documents).
"""

from .core import HighRize
from .client import CompressedClient
from .models import CompressionResult, SavingsReport, Modality
from .tokens import TokenCounter, count as count_tokens

__version__ = "0.2.0"
__all__ = [
    "HighRize",
    "CompressedClient",
    "CompressionResult",
    "SavingsReport",
    "Modality",
    "TokenCounter",
    "count_tokens",
]
