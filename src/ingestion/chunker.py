"""Utilities for splitting documents into smaller text chunks."""

from collections.abc import Sequence


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into chunks with a configurable overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words: Sequence[str] = text.split()

    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks