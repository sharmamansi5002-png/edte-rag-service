"""Tests for the document text chunker."""

import pytest

from src.ingestion.chunker import chunk_text


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []


def test_short_text_returns_one_chunk() -> None:
    assert chunk_text("Hello world", chunk_size=5, overlap=1) == [
        "Hello world"
    ]


def test_long_text_is_split() -> None:
    result = chunk_text(
        "one two three four five six",
        chunk_size=3,
        overlap=1,
    )

    assert result == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_invalid_chunk_size_raises_error() -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=3, overlap=3)