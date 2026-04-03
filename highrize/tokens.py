"""
tokpress.tokens — Accurate token counting per provider/model.

Uses tiktoken for OpenAI models, anthropic's tokenizer for Claude,
and falls back to the char/4 heuristic when neither is available.
Centralizes all token estimation so every compressor uses the same logic.
"""

from typing import Union, List, Optional
import re


# ---------------------------------------------------------------------------
# Provider → encoding name map
# ---------------------------------------------------------------------------
_OPENAI_ENCODINGS = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}

_ANTHROPIC_MODELS = {
    "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus",
    "claude-3-haiku", "claude-3-sonnet", "claude-2",
}

_GEMINI_CHARS_PER_TOKEN = 4  # Google doesn't expose a public tokenizer


def _tiktoken_available() -> bool:
    try:
        import tiktoken  # noqa
        return True
    except ImportError:
        return False


def _anthropic_available() -> bool:
    try:
        import anthropic  # noqa
        return True
    except ImportError:
        return False


def _heuristic(text: str) -> int:
    """char/4 fallback — accurate to ±15% for English."""
    return max(1, len(text) // 4)


class TokenCounter:
    """
    Accurate token counter for any provider/model.

    Usage:
        tc = TokenCounter(model="gpt-4o")
        n = tc.count("Hello, world!")

        # Count a full messages list (includes per-message overhead)
        n = tc.count_messages([{"role": "user", "content": "..."}])
    """

    def __init__(self, model: str = "gpt-4o", provider: str = "openai"):
        self.model = model
        self.provider = provider
        self._enc = None
        self._anthropic_client = None
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        if self.provider == "anthropic" or any(
            self.model.startswith(m) for m in _ANTHROPIC_MODELS
        ):
            if _anthropic_available():
                return "anthropic"
        if self.provider == "openai" or self.model in _OPENAI_ENCODINGS:
            if _tiktoken_available():
                return "tiktoken"
        return "heuristic"

    def _get_tiktoken_enc(self):
        if self._enc is None:
            import tiktoken
            enc_name = _OPENAI_ENCODINGS.get(self.model, "cl100k_base")
            try:
                self._enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._enc = tiktoken.get_encoding(enc_name)
        return self._enc

    def count(self, text: str) -> int:
        """Count tokens in a single text string."""
        if not text:
            return 0

        if self._backend == "tiktoken":
            enc = self._get_tiktoken_enc()
            return len(enc.encode(text))

        elif self._backend == "anthropic":
            try:
                import anthropic
                if self._anthropic_client is None:
                    self._anthropic_client = anthropic.Anthropic()
                result = self._anthropic_client.messages.count_tokens(
                    model=self.model if "claude" in self.model else "claude-3-haiku-20240307",
                    messages=[{"role": "user", "content": text}],
                )
                return result.input_tokens
            except Exception:
                return _heuristic(text)

        return _heuristic(text)

    def count_messages(self, messages: list) -> int:
        """Count tokens in an OpenAI-style messages list (includes overhead)."""
        total = 0
        for msg in messages:
            total += 4  # per-message overhead (role + framing tokens)
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += self.count(part.get("text", ""))
        total += 2  # reply primer
        return total

    def count_savings(
        self, original: str, compressed: str
    ) -> dict:
        """Return a dict with full stats."""
        orig = self.count(original)
        comp = self.count(compressed)
        saved = orig - comp
        pct = round(saved / orig * 100, 2) if orig > 0 else 0.0
        return {
            "original_tokens": orig,
            "compressed_tokens": comp,
            "tokens_saved": saved,
            "savings_pct": pct,
            "backend": self._backend,
        }

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        return f"TokenCounter(model={self.model!r}, backend={self._backend!r})"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_counter = TokenCounter()


def count(text: str, model: str = "gpt-4o") -> int:
    """Quick token count with no setup."""
    return TokenCounter(model=model).count(text)


def compare(original: str, compressed: str, model: str = "gpt-4o") -> dict:
    """Quick savings comparison."""
    return TokenCounter(model=model).count_savings(original, compressed)
