"""Unit tests for Neo4j vector search."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.vector_search import VectorSearch


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mocked Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()

    driver.session.return_value.__enter__.return_value = session
    session.run.return_value = MagicMock()

    return driver


def test_rejects_invalid_top_k(mock_driver: MagicMock) -> None:
    search = VectorSearch(mock_driver)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        search.search([0.1, 0.2], top_k=0)

    mock_driver.session.assert_not_called()


def test_search_returns_records(mock_driver: MagicMock) -> None:
    expected = [
        {
            "chunk_urn": "urn:chunk:sample:0",
            "text": "Finance System manages customer records.",
            "score": 0.95,
        },
        {
            "chunk_urn": "urn:chunk:sample:1",
            "text": "Customer information is stored in a table.",
            "score": 0.85,
        },
    ]

    records = []
    for item in expected:
        record = MagicMock()
        record.data.return_value = item
        records.append(record)

    session = mock_driver.session.return_value.__enter__.return_value
    session.run.return_value = records

    search = VectorSearch(mock_driver)
    result = search.search([0.1, 0.2, 0.3], top_k=2)

    assert result == expected


def test_search_passes_correct_parameters(
    mock_driver: MagicMock,
) -> None:
    search = VectorSearch(mock_driver)
    embedding = [0.1, 0.2, 0.3]

    search.search(embedding, top_k=7)

    session = mock_driver.session.return_value.__enter__.return_value
    session.run.assert_called_once()

    _, kwargs = session.run.call_args

    assert kwargs["index_name"] == "chunk_embedding"
    assert kwargs["top_k"] == 7
    assert kwargs["query_embedding"] == embedding


def test_create_index_executes_query(
    mock_driver: MagicMock,
) -> None:
    search = VectorSearch(mock_driver)

    search.create_index()

    session = mock_driver.session.return_value.__enter__.return_value
    session.run.assert_called_once()

    query = session.run.call_args[0][0]

    assert "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS" in query
    assert "vector.dimensions" in query
    assert "vector.similarity_function" in query

    session.run.return_value.consume.assert_called_once()


def test_search_uses_default_top_k(
    mock_driver: MagicMock,
) -> None:
    search = VectorSearch(mock_driver)

    search.search([0.1, 0.2])

    session = mock_driver.session.return_value.__enter__.return_value
    _, kwargs = session.run.call_args

    assert kwargs["top_k"] == 5