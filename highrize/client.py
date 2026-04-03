"""
CompressedClient — drop-in wrapper around any AI client.

Intercepts every request, compresses content automatically,
then forwards to the original client. Zero changes to your code.

Supports:
  - OpenAI client  (openai.OpenAI)
  - Anthropic client (anthropic.Anthropic)
  - Any client with a .chat.completions.create() or .messages.create() method
  - httpx / requests based custom clients (via raw_request mode)
"""

from typing import Any, Optional, Dict, List
from .core import HighRize
from .models import Modality


class CompressedClient:
    """
    Wraps any AI client with automatic compression.

    Example — OpenAI:
        from openai import OpenAI
        from highrize import CompressedClient

        client = CompressedClient(OpenAI(), model="gpt-4o")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "very long prompt..."}]
        )
        print(client.tp.report.summary())

    Example — Anthropic:
        from anthropic import Anthropic
        from highrize import CompressedClient

        client = CompressedClient(Anthropic(), provider="anthropic", model="claude-3-5-sonnet")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "..."}]
        )

    Example — Ollama (OpenAI-compatible):
        from openai import OpenAI
        from highrize import CompressedClient

        ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client = CompressedClient(ollama, model="llama3.2")
    """

    def __init__(
        self,
        client: Any,
        model: str = "default",
        provider: str = "openai",
        **highrize_kwargs,
    ):
        self._client = client
        self.tp = HighRize(model=model, provider=provider, **highrize_kwargs)
        self._chat = _ChatProxy(self._client, self.tp)
        self._messages = _MessagesProxy(self._client, self.tp)

    @property
    def chat(self):
        return self._chat

    @property
    def messages(self):
        return self._messages

    def __getattr__(self, name: str):
        # Fall through to the original client for anything else
        return getattr(self._client, name)


class _ChatProxy:
    """Proxies client.chat.completions.create() with compression."""

    def __init__(self, client, tp: HighRize):
        self._client = client
        self.tp = tp
        self.completions = _CompletionsProxy(client, tp)


class _CompletionsProxy:
    def __init__(self, client, tp: HighRize):
        self._client = client
        self.tp = tp

    def create(self, messages: List[Dict], **kwargs) -> Any:
        compressed_messages, _ = self.tp.compress_messages(messages)
        return self._client.chat.completions.create(messages=compressed_messages, **kwargs)


class _MessagesProxy:
    """Proxies client.messages.create() (Anthropic-style) with compression."""

    def __init__(self, client, tp: HighRize):
        self._client = client
        self.tp = tp

    def create(self, messages: List[Dict], system: Optional[str] = None, **kwargs) -> Any:
        compressed_messages, _ = self.tp.compress_messages(messages)

        # Also compress system prompt
        if system:
            result = self.tp.compress(system, Modality.TEXT)
            system = result.compressed

        call_kwargs = dict(messages=compressed_messages, **kwargs)
        if system is not None:
            call_kwargs["system"] = system

        return self._client.messages.create(**call_kwargs)
