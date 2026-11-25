"""Unit tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, QueryRequest

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Semantic GraphRAG Engine API is running."
    )


def test_query_request_validation() -> None:
    request = QueryRequest(question="Test question")

    assert request.question == "Test question"
    assert request.top_k == 5
    assert request.expand_graph_hops == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "Test", "top_k": 0},
        {"question": "Test", "expand_graph_hops": 3},
    ],
)
def test_query_invalid_request(payload: dict) -> None:
    response = client.post("/api/v1/rag/query", json=payload)

    assert response.status_code == 422


@patch("src.api.main.Neo4jClient")
def test_health_success(mock_neo4j: MagicMock) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "database": "connected",
    }

    mock_neo4j.return_value.verify_connection.assert_called_once()
    mock_neo4j.return_value.close.assert_called_once()


@patch("src.api.main.Neo4jClient")
def test_health_failure(mock_neo4j: MagicMock) -> None:
    mock_neo4j.return_value.verify_connection.side_effect = (
        RuntimeError("Database unavailable")
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert "Health check failed" in response.json()["detail"]
    mock_neo4j.return_value.close.assert_called_once()


def test_ingest_rejects_unsupported_file() -> None:
    response = client.post(
        "/api/v1/rag/ingest",
        files={"file": ("test.txt", b"Hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only Markdown and PDF" in response.json()["detail"]


def test_ingest_rejects_empty_file() -> None:
    response = client.post(
        "/api/v1/rag/ingest",
        files={"file": ("empty.md", b"", "text/markdown")},
    )

    assert response.status_code == 400
    assert "uploaded file is empty" in response.json()["detail"]


@patch("src.api.main.ingest_document")
def test_ingest_success(mock_ingest: MagicMock) -> None:
    mock_ingest.return_value = {
        "document_urn": "urn:doc:test",
        "chunks_indexed": 3,
        "entities": {"systems": ["Finance System"]},
    }

    response = client.post(
        "/api/v1/rag/ingest",
        files={
            "file": (
                "test.md",
                b"# Finance System",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["document_urn"] == "urn:doc:test"
    assert response.json()["chunks_indexed"] == 3

    mock_ingest.assert_called_once()

    temporary_path = mock_ingest.call_args.kwargs["file_path"]
    assert not temporary_path.exists()


@patch("src.api.main.ingest_document")
def test_ingest_failure(mock_ingest: MagicMock) -> None:
    mock_ingest.side_effect = RuntimeError("Ingestion failed")

    response = client.post(
        "/api/v1/rag/ingest",
        files={
            "file": (
                "test.md",
                b"# Test",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 500
    assert "Document ingestion failed" in response.json()["detail"]


@patch("src.api.main.HybridRetriever")
@patch("src.api.main.Neo4jClient")
def test_query_success(
    mock_neo4j: MagicMock,
    mock_retriever: MagicMock,
) -> None:
    mock_retriever.return_value.retrieve.return_value = {
        "answer": "The Finance System stores customer data.",
        "sources": [
            {
                "chunk_urn": "urn:chunk:test:0",
                "score": 0.95,
                "text": "Finance System stores customer information.",
            }
        ],
        "graph_context": [
            {
                "urn": "urn:system:finance_system",
                "name": "Finance System",
                "related_actors": ["Admin"],
            }
        ],
    }

    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "What does the Finance System do?",
            "top_k": 5,
            "expand_graph_hops": 1,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "The Finance System stores customer data."
    )
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_urn"] == "urn:chunk:test:0"
    assert data["graph_context"][0]["name"] == "Finance System"

    mock_retriever.return_value.retrieve.assert_called_once()
    mock_neo4j.return_value.close.assert_called_once()


@patch("src.api.main.HybridRetriever")
@patch("src.api.main.Neo4jClient")
def test_query_rejects_blank_question(
    mock_neo4j: MagicMock,
    mock_retriever: MagicMock,
) -> None:
    response = client.post(
        "/api/v1/rag/query",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert "question cannot be empty" in response.json()["detail"]

    mock_neo4j.assert_not_called()
    mock_retriever.assert_not_called()


@patch("src.api.main.Neo4jClient")
def test_query_database_failure(mock_neo4j: MagicMock) -> None:
    mock_neo4j.return_value.verify_connection.side_effect = (
        RuntimeError("Database unavailable")
    )

    response = client.post(
        "/api/v1/rag/query",
        json={"question": "Test question"},
    )

    assert response.status_code == 500
    assert "Query failed" in response.json()["detail"]
    mock_neo4j.return_value.close.assert_called_once()


@patch("src.api.main.HybridRetriever")
@patch("src.api.main.Neo4jClient")
def test_query_value_error(
    mock_neo4j: MagicMock,
    mock_retriever: MagicMock,
) -> None:
    mock_retriever.return_value.retrieve.side_effect = (
        ValueError("Invalid retrieval parameters")
    )

    response = client.post(
        "/api/v1/rag/query",
        json={"question": "Test question"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid retrieval parameters"
    )
    mock_neo4j.return_value.close.assert_called_once()
