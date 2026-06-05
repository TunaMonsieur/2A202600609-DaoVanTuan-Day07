from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(' '.join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text]

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        if sep == "":
            # Character-level: split into individual chars, then re-merge into sized chunks
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        parts = current_text.split(sep)
        if len(parts) <= 1:
            return self._split(current_text, rest)

        result = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > self.chunk_size:
                    result.extend(self._split(part, rest))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)

        return [r for r in result if r]


class ParentChildChunker:
    """
    Two-level chunking for structured documents.

    Parent chunks (large) are split by section boundaries (e.g. '## ').
    Child chunks (small) are split within each parent for embedding/search.

    Retrieval flow:
        1. Embed and index child chunks.
        2. Search returns child chunks whose metadata carries parent_content.
        3. Agent uses parent_content as context — richer than the child alone.

    chunk() returns child strings (drop-in compatible with other chunkers).
    chunk_with_parents() returns [(parent, child), ...] for full RAG indexing.
    """

    def __init__(
        self,
        parent_separators: list[str] | None = None,
        child_separators: list[str] | None = None,
        parent_chunk_size: int = 800,
        child_chunk_size: int = 200,
    ) -> None:
        self._parent_chunker = RecursiveChunker(
            separators=parent_separators or ["\n## ", "\n\n", "\n"],
            chunk_size=parent_chunk_size,
        )
        self._child_chunker = RecursiveChunker(
            separators=child_separators or ["\n\n", "\n- ", "\n+ ", "\n", ". ", " ", ""],
            chunk_size=child_chunk_size,
        )

    def chunk(self, text: str) -> list[str]:
        """Return child chunks only (compatible with chunker interface)."""
        if not text:
            return []
        parents = self._parent_chunker.chunk(text)
        children: list[str] = []
        for parent in parents:
            children.extend(self._child_chunker.chunk(parent))
        return children

    def chunk_with_parents(self, text: str) -> list[tuple[str, str]]:
        """Return (parent_text, child_text) pairs for RAG indexing."""
        if not text:
            return []
        parents = self._parent_chunker.chunk(text)
        pairs: list[tuple[str, str]] = []
        for parent in parents:
            for child in self._child_chunker.chunk(parent):
                pairs.append((parent, child))
        return pairs


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            'fixed_size': FixedSizeChunker(chunk_size=chunk_size),
            'by_sentences': SentenceChunker(),
            'recursive': RecursiveChunker(chunk_size=chunk_size),
        }
        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            avg_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
            result[name] = {
                'count': len(chunks),
                'avg_length': avg_length,
                'chunks': chunks,
            }
        return result
