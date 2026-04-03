"""
Text compressor — reduces prompt token count while preserving meaning.
Strategies (applied in order):
  1. Whitespace normalization
  2. Redundancy removal (exact duplicate sentences)
  3. Stopword softening (only on non-instruction contexts)
  4. Semantic summarization (for long system prompts / documents)
"""

import re
from typing import Optional
from ..models import CompressionResult, Modality


# Rough token estimator: ~4 chars per token (GPT-style)
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# Common filler phrases that add tokens but not meaning
_FILLER_PATTERNS = [
    r"\bplease\s+note\s+that\b",
    r"\bit\s+is\s+important\s+to\s+note\s+that\b",
    r"\bkindly\s+be\s+advised\s+that\b",
    r"\bas\s+previously\s+mentioned\b",
    r"\bI\s+would\s+like\s+to\s+point\s+out\b",
    r"\bfor\s+your\s+information\b",
    r"\bwith\s+that\s+said\b",
    r"\bin\s+conclusion\b",
    r"\bto\s+summarize\b",
    r"\bwithout\s+further\s+ado\b",
    r"\bI\s+hope\s+this\s+(helps|finds\s+you\s+well|message\s+finds\s+you)\b",
    r"\bfeel\s+free\s+to\b",
    r"\bdon'?t\s+hesitate\s+to\b",
    r"\bAs\s+an\s+AI\s+(language\s+model|assistant)\b",
]

_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_filler_phrases(text: str) -> str:
    return _FILLER_RE.sub("", text)


def _deduplicate_sentences(text: str) -> str:
    """Remove exact duplicate sentences (common in long prompts)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    out = []
    for s in sentences:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return " ".join(out)


def _remove_redundant_examples(text: str, max_examples: int = 3) -> str:
    """
    If a prompt has many repeated example blocks (Few-shot),
    keep only up to max_examples.
    Detects blocks starting with 'Example N:' or 'Q:' / 'A:'.
    """
    example_blocks = re.split(r"(?i)(?=example\s*\d+\s*:|^Q\s*:|^User\s*:)", text, flags=re.MULTILINE)
    if len(example_blocks) <= max_examples + 1:
        return text
    # Keep first block (instructions) + first max_examples example blocks
    trimmed = example_blocks[0] + "".join(example_blocks[1:max_examples + 1])
    removed = len(example_blocks) - 1 - max_examples
    trimmed += f"\n[{removed} additional examples omitted for brevity]"
    return trimmed


class TextCompressor:
    """
    Compresses text prompts.

    Args:
        remove_fillers: Strip filler phrases (default True)
        deduplicate: Remove duplicate sentences (default True)
        max_examples: Max few-shot examples to keep (None = keep all)
        summarize_fn: Optional callable(text: str) -> str for heavy summarization
    """

    def __init__(
        self,
        token_counter=None,
        remove_fillers: bool = True,
        deduplicate: bool = True,
        max_examples: Optional[int] = None,
        summarize_fn=None,
    ):
        from ..tokens import TokenCounter
        self.token_counter = token_counter or TokenCounter()
        self.remove_fillers = remove_fillers
        self.deduplicate = deduplicate
        self.max_examples = max_examples
        self.summarize_fn = summarize_fn

    def compress(self, text: str) -> CompressionResult:
        original_text = text
        original_tokens = self.token_counter.count(text)

        # Pipeline
        text = _normalize_whitespace(text)

        if self.remove_fillers:
            text = _remove_filler_phrases(text)

        if self.deduplicate:
            text = _deduplicate_sentences(text)

        if self.max_examples is not None:
            text = _remove_redundant_examples(text, self.max_examples)

        if self.summarize_fn and len(text) > 2000:
            text = self.summarize_fn(text)

        text = _normalize_whitespace(text)

        return CompressionResult(
            original=original_text,
            compressed=text,
            modality=Modality.TEXT,
            original_tokens=original_tokens,
            compressed_tokens=self.token_counter.count(text),
            original_size_bytes=len(original_text.encode()),
            compressed_size_bytes=len(text.encode()),
        )
