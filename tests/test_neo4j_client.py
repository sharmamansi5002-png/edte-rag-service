"""Unit tests for Neo4jClient."""

from unittest.mock import MagicMock, patch

import pytest

from src.graph.neo4j_client import Neo4jClient


@pytest.fixture
def mock_client() -> Neo4jClient:
    """Create a Neo4jClient without a real database connection."""
    with (
        patch("src.graph.neo4j_client.load_dotenv"),
        patch("src.graph.neo4j_client.GraphDatabase.driver") as mock_driver,
    ):
        mock_driver.return_value = MagicMock()
        client = Neo4jClient(password="test-password")
        return client


def test_init_uses_default_configuration() -> None:
    with (
        patch("src.graph.neo4j_client.load_dotenv"),
        patch("src.graph.neo4j_client.GraphDatabase.driver") as mock_driver,
    ):
        Neo4jClient(password="test-password")

    mock_driver.assert_called_once_with(
        "bolt://localhost:7687",
        auth=("neo4j", "test-password"),
    )


def test_init_raises_when_password_missing() -> None:
    with (
        patch("src.graph.neo4j_client.load_dotenv"),
        patch.dict("os.environ", {"NEO4J_PASSWORD": ""}),
        pytest.raises(ValueError, match="NEO4J_PASSWORD is missing"),
    ):
        Neo4jClient()


def test_init_uses_custom_configuration() -> None:
    with (
        patch("src.graph.neo4j_client.load_dotenv"),
        patch("src.graph.neo4j_client.GraphDatabase.driver") as mock_driver,
    ):
        Neo4jClient(
            uri="bolt://localhost:7687",
            username="admin",
            password="secret",
        )

    mock_driver.assert_called_once_with(
        "bolt://localhost:7687",
        auth=("admin", "secret"),
    )


def test_driver_property(mock_client: Neo4jClient) -> None:
    assert mock_client.driver is mock_client._driver


def test_verify_connection(mock_client: Neo4jClient) -> None:
    mock_client.verify_connection()
    mock_client.driver.verify_connectivity.assert_called_once()


def test_close(mock_client: Neo4jClient) -> None:
    mock_client.close()
    mock_client.driver.close.assert_called_once()


def test_create_vector_index(mock_client: Neo4jClient) -> None:
    with patch.object(
        mock_client.vector_search,
        "create_index",
    ) as mock_create:
        mock_client.create_vector_index()

    mock_create.assert_called_once()

def test_create_constraints(mock_client: Neo4jClient) -> None:
    session = MagicMock()
    mock_client.driver.session.return_value.__enter__.return_value = session

    mock_client.create_constraints()

    assert session.run.call_count == 6


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Finance System", "finance_system"),
        ("Customer Table", "customer_table"),
        ("Admin-User", "admin_user"),
        ("API_v2", "api_v2"),
        ("  Hello World!  ", "hello_world"),
    ],
)
def test_normalize_entity_name(name: str, expected: str) -> None:
    assert Neo4jClient.normalize_entity_name(name) == expected


def test_normalize_entity_name_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Entity name cannot be empty"):
        Neo4jClient.normalize_entity_name(" !!! ")


def test_store_document(mock_client: Neo4jClient) -> None:
    session = MagicMock()
    mock_client.driver.session.return_value.__enter__.return_value = session

    chunks = [{"urn": "urn:chunk:test:0"}]
    entities = {"systems": [], "roles": [], "actors": [], "resource_details": []}

    mock_client.store_document(
        "urn:doc:test",
        "Test Document",
        "docs/test.md",
        chunks,
        entities,
    )

    session.execute_write.assert_called_once()


def test_store_document_transaction() -> None:
    tx = MagicMock()

    chunks = [
        {
            "urn": "urn:chunk:test:0",
            "text": "Finance System Customer Admin",
            "chunk_index": 0,
            "embedding": [0.1, 0.2],
        }
    ]

    entities = {
        "systems": ["Finance System"],
        "roles": ["Admin"],
        "actors": ["Bob"],
        "resource_details": [
            {
                "system": "Finance System",
                "type": "table",
                "name": "Customer",
            }
        ],
    }

    Neo4jClient._store_document_transaction(
        tx,
        "urn:doc:test",
        "Test Document",
        "docs/test.md",
        chunks,
        entities,
    )

    assert tx.run.call_count == 7