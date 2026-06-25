"""Text chunking for embedding + storage.

Two strategies, both implemented as pure functions so they're trivially testable:

- `split_paragraphs`: splits on blank lines first, falls back to char-based
  chunking for over-long paragraphs. Good default for prose.
- `split_by_chars`: fixed-size chunks with overlap. Fallback when paragraph
  boundaries are unreliable (e.g. minified HTML, raw source code dumps).

Both honor `chunk_size_chars` and `chunk_overlap_chars` from PipelineConfig.
"""
from __future__ import annotations

import re


def split_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text on paragraph boundaries, packing paragraphs into chunks.

    Args:
        text: the source text to chunk.
        chunk_size: target maximum chunk length in characters.
        overlap: number of trailing characters from the previous chunk to
            prepend to the next chunk (for retrieval continuity).

    Returns:
        Ordered list of chunks. Empty input yields an empty list.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    cleaned = text.strip()
    if not cleaned:
        return []

    # Split on one or more blank lines, preserving non-empty paragraphs.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    for paragraph in paragraphs:
        # If a single paragraph exceeds chunk_size, flush the buffer and chunk it directly.
        if len(paragraph) >= chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(split_by_chars(paragraph, chunk_size, overlap))
            continue

        # Try to append; if it overflows, flush buffer and start new chunk.
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # Carry overlap from the tail of the flushed buffer.
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                buffer = f"{tail}\n\n{paragraph}" if tail.strip() else paragraph
            else:
                buffer = paragraph

    if buffer:
        chunks.append(buffer)
    return chunks


def split_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size character chunks with overlap. Order preserved.

    Args:
        text: source text.
        chunk_size: max characters per chunk.
        overlap: characters of overlap between consecutive chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(cleaned), step):
        piece = cleaned[start : start + chunk_size]
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(cleaned):
            break
    return chunks
