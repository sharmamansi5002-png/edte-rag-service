"""Unit tests for the hybrid retrieval pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_services() -> dict[str, MagicMock]:
    """Create mocked retrieval services."""
    return {
        "embedding": MagicMock(),
        "vector": MagicMock(),
        "graph": MagicMock(),
        "gemini": MagicMock(),
    }


@pytest.fixture
def retriever(mock_services: dict[str, MagicMock]) -> HybridRetriever:
    """Create a retriever with mocked dependencies."""
    with (
        patch(
            "src.retrieval.hybrid_retriever.EmbeddingService",
            return_value=mock_services["embedding"],
        ),
        patch(
            "src.retrieval.hybrid_retriever.VectorSearch",
            return_value=mock_services["vector"],
        ),
        patch(
            "src.retrieval.hybrid_retriever.GraphExpansion",
            return_value=mock_services["graph"],
        ),
        patch(
            "src.retrieval.hybrid_retriever.GeminiService",
            return_value=mock_services["gemini"],
        ),
    ):
        return HybridRetriever(MagicMock())


def test_rejects_empty_question(
    retriever: HybridRetriever,
) -> None:
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retriever.retrieve("   ")


def test_rejects_invalid_top_k(
    retriever: HybridRetriever,
) -> None:
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.retrieve("What is the system?", top_k=0)


def test_rejects_invalid_graph_hops(
    retriever: HybridRetriever,
) -> None:
    with pytest.raises(ValueError, match="Graph expansion hops must be 1 or 2"):
        retriever.retrieve("What is the system?", expand_graph_hops=3)


def test_retrieves_answer_with_sources(
    retriever: HybridRetriever,
    mock_services: dict[str, MagicMock],
) -> None:
    embedding = [0.1, 0.2, 0.3]
    sources = [
        {
            "chunk_urn": "urn:chunk:sample:0",
            "text": "The Finance System manages customer records.",
        }
    ]
    graph_context = [
        {
            "entity_urn": "urn:system:finance",
            "entity_name": "Finance System",
        }
    ]

    mock_services["embedding"].embed_text.return_value = embedding
    mock_services["vector"].search.return_value = sources
    mock_services["graph"].expand.return_value = graph_context
    mock_services["gemini"].generate_answer.return_value = (
        "The Finance System manages customer records."
    )

    result = retriever.retrieve("What does the Finance System do?")

    assert result["answer"] == (
        "The Finance System manages customer records."
    )
    assert result["sources"] == sources
    assert result["graph_context"] == graph_context

    mock_services["embedding"].embed_text.assert_called_once_with(
        "What does the Finance System do?"
    )
    mock_services["vector"].search.assert_called_once_with(
        query_embedding=embedding,
        top_k=5,
    )
    mock_services["graph"].expand.assert_called_once_with(
        chunk_urns=["urn:chunk:sample:0"],
        hops=1,
    )
    mock_services["gemini"].generate_answer.assert_called_once()


def test_returns_fallback_when_no_sources_found(
    retriever: HybridRetriever,
    mock_services: dict[str, MagicMock],
) -> None:
    mock_services["embedding"].embed_text.return_value = [0.1, 0.2]
    mock_services["vector"].search.return_value = []
    mock_services["graph"].expand.return_value = []

    result = retriever.retrieve("Unknown information")

    assert result["answer"] == (
        "I could not find relevant information in the indexed documents."
    )
    assert result["sources"] == []
    assert result["graph_context"] == []

    mock_services["gemini"].generate_answer.assert_not_called()


def test_passes_custom_search_and_graph_parameters(
    retriever: HybridRetriever,
    mock_services: dict[str, MagicMock],
) -> None:
    mock_services["embedding"].embed_text.return_value = [0.5, 0.6]
    mock_services["vector"].search.return_value = [
        {"chunk_urn": "urn:chunk:test:0", "text": "Test content"}
    ]
    mock_services["graph"].expand.return_value = []
    mock_services["gemini"].generate_answer.return_value = "Test answer"

    retriever.retrieve(
        "Test question",
        top_k=10,
        expand_graph_hops=2,
    )

    mock_services["vector"].search.assert_called_once_with(
        query_embedding=[0.5, 0.6],
        top_k=10,
    )
    mock_services["graph"].expand.assert_called_once_with(
        chunk_urns=["urn:chunk:test:0"],
        hops=2,
    )