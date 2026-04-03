from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    EMBEDDING = "embedding"


@dataclass
class CompressionResult:
    """Result of compressing a single input."""
    original: Any
    compressed: Any
    modality: Modality
    original_tokens: int
    compressed_tokens: int
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round(self.tokens_saved / self.original_tokens * 100, 2)

    def __repr__(self):
        return (
            f"CompressionResult({self.modality.value}: "
            f"{self.original_tokens} → {self.compressed_tokens} tokens, "
            f"{self.savings_pct}% saved)"
        )


@dataclass
class SavingsReport:
    """Cumulative savings across a session."""
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_cost_original_usd: float = 0.0
    total_cost_compressed_usd: float = 0.0
    requests: int = 0
    results: list = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.total_original_tokens - self.total_compressed_tokens

    @property
    def savings_pct(self) -> float:
        if self.total_original_tokens == 0:
            return 0.0
        return round(self.tokens_saved / self.total_original_tokens * 100, 2)

    @property
    def cost_saved_usd(self) -> float:
        return round(self.total_cost_original_usd - self.total_cost_compressed_usd, 6)

    def add(self, result: CompressionResult, cost_per_1k: float = 0.0):
        self.results.append(result)
        self.total_original_tokens += result.original_tokens
        self.total_compressed_tokens += result.compressed_tokens
        self.total_cost_original_usd += result.original_tokens / 1000 * cost_per_1k
        self.total_cost_compressed_usd += result.compressed_tokens / 1000 * cost_per_1k
        self.requests += 1

    def summary(self) -> str:
        return (
            f"tokpress session summary\n"
            f"  Requests     : {self.requests}\n"
            f"  Tokens       : {self.total_original_tokens:,} → {self.total_compressed_tokens:,}\n"
            f"  Saved        : {self.tokens_saved:,} tokens ({self.savings_pct}%)\n"
            f"  Cost saved   : ${self.cost_saved_usd:.4f} USD"
        )
