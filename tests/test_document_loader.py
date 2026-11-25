"""Tests for Markdown and PDF document loading."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.ingestion.document_loader import load_document


def test_load_markdown_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text(
        "# EDTE\nThis is a sample document.",
        encoding="utf-8",
    )

    result = load_document(file_path)

    assert "# EDTE" in result
    assert "sample document" in result


def test_missing_file_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        load_document(file_path)


def test_unsupported_extension_raises_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Sample text", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)


def test_load_pdf_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    with file_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    result = load_document(file_path)

    assert result == ""
