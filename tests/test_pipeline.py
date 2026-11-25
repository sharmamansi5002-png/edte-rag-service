"""Tests for the document ingestion pipeline."""

from pathlib import Path

from src.ingestion.pipeline import ingest_document


def test_ingest_document_returns_chunks_and_entities(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text(
        "The Finance System uses the Customer table. "
        "The Admin role can access the Finance System.",
        encoding="utf-8",
    )

    result = ingest_document(
        file_path,
        chunk_size=8,
        overlap=2,
    )

    assert result["chunks"]
    assert "Finance System" in result["entities"]["systems"]
    assert "Customer" in result["entities"]["resources"]
    assert "Admin" in result["entities"]["roles"]


def test_ingest_empty_document_returns_empty_results(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "empty.md"
    file_path.write_text("", encoding="utf-8")

    result = ingest_document(file_path)

    {
    "chunks": [],
    "entities": {
        "systems": [],
        "resources": [],
        "roles": [],
        "actors": [],
    },
}