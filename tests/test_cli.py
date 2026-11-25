"""Unit tests for the command-line interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import cli


def test_index_documents_directory_not_found(tmp_path: Path) -> None:
    parser = MagicMock()
    parser.error.side_effect = SystemExit(2)

    with pytest.raises(SystemExit):
        cli.index_documents(tmp_path / "missing", parser)

    parser.error.assert_called_once()


@patch("src.cli.ingest_document")
def test_index_documents_success(
    mock_ingest: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "first.md").write_text("# First")
    (tmp_path / "second.pdf").write_bytes(b"PDF")
    (tmp_path / "ignored.txt").write_text("Ignored")

    mock_ingest.return_value = {"chunks_indexed": 3}

    cli.index_documents(tmp_path, MagicMock())

    assert mock_ingest.call_count == 2
    output = capsys.readouterr().out
    assert "Indexed first.md: 3 chunks" in output
    assert "Indexed second.pdf: 3 chunks" in output


def test_index_documents_no_supported_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "notes.txt").write_text("Notes")

    cli.index_documents(tmp_path, MagicMock())

    assert "No Markdown or PDF documents found" in capsys.readouterr().out


@patch("src.cli.ingest_document")
def test_index_documents_handles_failure(
    mock_ingest: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "broken.md").write_text("# Broken")
    mock_ingest.side_effect = RuntimeError("Test failure")

    cli.index_documents(tmp_path, MagicMock())

    assert "Failed to index broken.md" in capsys.readouterr().out


@patch("src.cli.HybridRetriever")
@patch("src.cli.Neo4jClient")
def test_ask_question_success(
    mock_neo4j: MagicMock,
    mock_retriever: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_retriever.return_value.retrieve.return_value = {
        "answer": "Test answer",
        "sources": [
            {
                "chunk_urn": "urn:chunk:test:0",
                "score": 0.95,
                "text": "Test source text",
            }
        ],
        "graph_context": [
            {
                "entity_urn": "urn:system:test",
                "entity_name": "Test System",
                "related_actors": ["Admin"],
            }
        ],
    }

    cli.ask_question("Test question")

    output = capsys.readouterr().out
    assert "Test answer" in output
    assert "urn:chunk:test:0" in output
    assert "Test System" in output
    assert "Admin" in output
    mock_neo4j.return_value.verify_connection.assert_called_once()
    mock_neo4j.return_value.close.assert_called_once()


@patch("src.cli.Neo4jClient")
def test_ask_question_closes_client_on_failure(
    mock_neo4j: MagicMock,
) -> None:
    mock_neo4j.return_value.verify_connection.side_effect = (
        RuntimeError("Database unavailable")
    )

    with pytest.raises(RuntimeError, match="Database unavailable"):
        cli.ask_question("Test question")

    mock_neo4j.return_value.close.assert_called_once()


@patch("src.cli.ask_question")
def test_main_ask_command(mock_ask: MagicMock) -> None:
    with patch(
        "sys.argv",
        ["cli", "ask", "--query", "Test question", "--top-k", "3",
         "--expand-graph-hops", "2"],
    ):
        cli.main()

    mock_ask.assert_called_once_with(
        question="Test question",
        top_k=3,
        expand_graph_hops=2,
    )


@patch("src.cli.index_documents")
def test_main_index_command(mock_index: MagicMock) -> None:
    with patch("sys.argv", ["cli", "index-docs", "--dir", "docs"]):
        cli.main()

    mock_index.assert_called_once()
    assert mock_index.call_args.args[0] == Path("docs")


def test_main_requires_command() -> None:
    with patch("sys.argv", ["cli"]):
        with pytest.raises(SystemExit):
            cli.main()