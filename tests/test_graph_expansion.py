"""Tests for Neo4j graph expansion."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.graph_expansion import GraphExpansion


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mocked Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    result = MagicMock()

    driver.session.return_value.__enter__.return_value = session
    session.run.return_value = result
    result.__iter__.return_value = iter([])

    return driver


def test_expand_returns_empty_for_no_chunks(
    mock_driver: MagicMock,
) -> None:
    expansion = GraphExpansion(mock_driver)

    result = expansion.expand([])

    assert result == []
    mock_driver.session.assert_not_called()


def test_expand_rejects_invalid_hops(
    mock_driver: MagicMock,
) -> None:
    expansion = GraphExpansion(mock_driver)

    with pytest.raises(ValueError, match="hops must be either 1 or 2"):
        expansion.expand(["urn:chunk:test:0"], hops=3)

    mock_driver.session.assert_not_called()


@pytest.mark.parametrize("hops", [1, 2])
def test_expand_runs_query_with_requested_hops(
    mock_driver: MagicMock,
    hops: int,
) -> None:
    expansion = GraphExpansion(mock_driver)
    chunk_urns = ["urn:chunk:test:0"]

    expansion.expand(chunk_urns, hops=hops)

    session = mock_driver.session.return_value.__enter__.return_value
    session.run.assert_called_once()

    _, kwargs = session.run.call_args

    assert kwargs["chunk_urns"] == chunk_urns
    assert kwargs["hops"] == hops


def test_expand_returns_entity_and_actor_data(
    mock_driver: MagicMock,
) -> None:
    expected = [
        {
            "entity_urn": "urn:system:finance",
            "entity_name": "Finance System",
            "entity_type": ["System"],
            "source_chunks": ["urn:chunk:test:0"],
            "related_actors": [
                {"urn": "urn:actor:admin", "name": "Admin"}
            ],
            "connection_type": "direct",
        }
    ]

    session = mock_driver.session.return_value.__enter__.return_value
    record = MagicMock()
    record.data.return_value = expected[0]
    session.run.return_value = [record]
    expansion = GraphExpansion(mock_driver)

    result = expansion.expand(["urn:chunk:test:0"])

    assert result == expected