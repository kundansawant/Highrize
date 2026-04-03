"""
Document compressor — reduces long documents/PDFs to relevant chunks.

Strategies:
  1. Extraction: Pull plain text from PDF, HTML, DOCX
  2. Chunking: Split into semantic chunks
  3. Ranking: Score chunks by query relevance (BM25 or embedding similarity)
  4. Token budgeting: Keep top-K chunks within token limit
"""

import re
from typing import List, Optional, Union
from ..models import CompressionResult, Modality
from .text import TextCompressor, _estimate_tokens


def _extract_text_from_pdf(path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except ImportError:
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            return "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            raise ImportError("Install pdfplumber or pypdf: pip install pdfplumber")


def _extract_text_from_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        # Naive fallback
        return re.sub(r"<[^>]+>", " ", html)


def _extract_text_from_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _bm25_score(query_terms: List[str], doc: str) -> float:
    """Simple term-frequency based relevance score."""
    doc_lower = doc.lower()
    score = 0.0
    for term in query_terms:
        count = doc_lower.count(term.lower())
        score += count / (count + 1.5) * 2.0  # BM25-like saturation
    return score


class DocumentCompressor:
    """
    Compresses documents (PDF, HTML, DOCX, plain text) to a token budget.

    Args:
        token_budget: Max tokens to return (keeps highest-relevance chunks)
        chunk_size: Words per chunk
        query: Optional query to rank chunks by relevance
        file_type: "pdf" | "html" | "docx" | "text" | None (auto-detect)
        compress_chunks: Apply TextCompressor to each chunk
    """

    def __init__(
        self,
        token_counter=None,
        token_budget: int = 2000,
        chunk_size: int = 300,
        query: Optional[str] = None,
        file_type: Optional[str] = None,
        compress_chunks: bool = True,
    ):
        from ..tokens import TokenCounter
        self.token_budget = token_budget
        self.chunk_size = chunk_size
        self.query = query
        self.file_type = file_type
        self.compress_chunks = compress_chunks
        self.token_counter = token_counter or TokenCounter()
        self._text_compressor = TextCompressor(token_counter=self.token_counter) if compress_chunks else None

    def compress(self, source: Union[str, bytes]) -> CompressionResult:
        """
        Args:
            source: File path, raw HTML string, or raw text string.
        """
        # --- Extract text ---
        text = self._extract(source)
        original_tokens = self.token_counter.count(text)

        # --- Chunk ---
        chunks = _chunk_text(text, self.chunk_size)

        # --- Rank by query relevance ---
        if self.query:
            query_terms = self.query.lower().split()
            scored = [(c, _bm25_score(query_terms, c)) for c in chunks]
            scored.sort(key=lambda x: x[1], reverse=True)
            chunks = [c for c, _ in scored]

        # --- Budget selection ---
        selected = []
        used_tokens = 0
        for chunk in chunks:
            if self._text_compressor:
                r = self._text_compressor.compress(chunk)
                chunk = r.compressed
            t = self.token_counter.count(chunk)
            if used_tokens + t > self.token_budget:
                break
            selected.append(chunk)
            used_tokens += t

        # Re-order by original position for coherence
        selected = [s.strip() for s in selected if s.strip()]
        compressed_text = "\n\n---\n\n".join(selected)
        compressed_tokens = self.token_counter.count(compressed_text)

        return CompressionResult(
            original=source,
            compressed=compressed_text,
            modality=Modality.DOCUMENT,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_size_bytes=len(text.encode()),
            compressed_size_bytes=len(compressed_text.encode()),
            metadata={
                "total_chunks": len(chunks),
                "selected_chunks": len(selected),
                "token_budget": self.token_budget,
                "query": self.query,
            },
        )

    def _extract(self, source: Union[str, bytes]) -> str:
        if isinstance(source, bytes):
            return source.decode("utf-8", errors="replace")

        ft = self.file_type
        if ft is None:
            # Auto-detect
            if source.strip().startswith("<"):
                ft = "html"
            elif source.endswith(".pdf"):
                ft = "pdf"
            elif source.endswith(".docx"):
                ft = "docx"
            else:
                ft = "text"

        if ft == "pdf":
            return _extract_text_from_pdf(source)
        elif ft == "html":
            return _extract_text_from_html(source)
        elif ft == "docx":
            return _extract_text_from_docx(source)
        else:
            if len(source) < 5000 and "\n" in source:
                return source  # looks like raw text
            try:
                with open(source, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                return source
