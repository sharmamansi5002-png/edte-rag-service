"""Opt-in integration tests using a real Neo4j database."""

import os
import time
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.graph.neo4j_client import Neo4jClient
from src.ingestion.urns import (
    actor_urn,
    chunk_urn,
    document_urn,
    resource_urn,
    role_urn,
    system_urn,
)
from src.retrieval.hybrid_retriever import HybridRetriever


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_NEO4J_INTEGRATION") != "1",
    reason="Set RUN_NEO4J_INTEGRATION=1 to run Neo4j integration tests.",
)


def wait_for_vector_index(client: Neo4jClient) -> None:
    """Wait until the Neo4j vector index is online."""

    query = """
    SHOW VECTOR INDEXES
    YIELD name, state
    WHERE name = $index_name
    RETURN state
    """

    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:
        with client.driver.session() as session:
            record = session.run(
                query,
                index_name="chunk_embedding",
            ).single()

        if record is not None and record["state"] == "ONLINE":
            return

        time.sleep(1)

    raise TimeoutError("Neo4j vector index did not become ONLINE.")


def test_hybrid_retrieval_with_real_neo4j() -> None:
    """Test vector search and graph expansion against real Neo4j."""

    client = Neo4jClient()

    test_id = uuid4().hex

    doc_id = f"integration_{test_id}"

    doc_urn = document_urn(doc_id)
    test_chunk_urn = chunk_urn(doc_id, 0)

    system_name = f"Integration System {test_id}"
    resource_name = f"Integration Table {test_id}"
    role_name = f"Integration Admin {test_id}"
    actor_name = f"Integration Actor {test_id}"

    system_id = system_urn(system_name)
    resource_id = resource_urn(
        system_name,
        "table",
        resource_name,
    )
    role_id = role_urn(role_name)
    actor_id = actor_urn(actor_name)

    test_urns = [
        doc_urn,
        test_chunk_urn,
        system_id,
        resource_id,
        role_id,
        actor_id,
    ]

    embedding = [1.0] + [0.0] * 383

    chunk_text = (
        f"The {system_name} contains the {resource_name}. "
        f"The {role_name} role manages this resource. "
        f"The responsible person is {actor_name}."
    )

    entities = {
        "systems": [system_name],
        "roles": [role_name],
        "actors": [actor_name],
        "resource_details": [
            {
                "system": system_name,
                "type": "table",
                "name": resource_name,
            }
        ],
    }

    try:
        client.verify_connection()

        # Ensure the vector index exists and is ready.
        client.create_vector_index()
        wait_for_vector_index(client)

        # Insert a uniquely identifiable test document.
        client.store_document(
            document_urn=doc_urn,
            title="Neo4j Integration Test",
            uri=f"integration://{test_id}",
            chunks=[
                {
                    "urn": test_chunk_urn,
                    "text": chunk_text,
                    "chunk_index": 0,
                    "embedding": embedding,
                }
            ],
            entities=entities,
        )

        # Mock embedding generation and Gemini only.
        # Vector search and graph expansion use real Neo4j.
        with (
            patch(
                "src.retrieval.hybrid_retriever.EmbeddingService"
            ) as mock_embedding,
            patch(
                "src.retrieval.hybrid_retriever.GeminiService"
            ) as mock_gemini,
        ):
            mock_embedding.return_value.embed_text.return_value = embedding

            mock_gemini.return_value.generate_answer.return_value = (
                "Integration test answer."
            )

            retriever = HybridRetriever(client.driver)

            result = retriever.retrieve(
                question="Which system contains the integration table?",
                top_k=100,
                expand_graph_hops=1,
            )

        # Verify the vector search retrieved our document chunk.
        retrieved_urns = {
            source["chunk_urn"]
            for source in result["sources"]
        }

        assert test_chunk_urn in retrieved_urns

        # Verify graph expansion returned the expected entities.
        graph_urns = {
            entity["entity_urn"]
            for entity in result["graph_context"]
        }

        assert system_id in graph_urns
        assert resource_id in graph_urns
        assert role_id in graph_urns

        # Verify the actor was linked to the graph context.
        related_actors = [
            actor
            for entity in result["graph_context"]
            for actor in entity["related_actors"]
        ]

        assert any(
            actor["urn"] == actor_id
            for actor in related_actors
        )

        # Verify Gemini answer generation was called.
        assert result["answer"] == "Integration test answer."

        mock_gemini.return_value.generate_answer.assert_called_once()

    finally:
        # Delete ONLY the uniquely identified integration-test records.
        try:
            with client.driver.session() as session:
                session.run(
                    """
                    MATCH (n)
                    WHERE n.urn IN $test_urns
                    DETACH DELETE n
                    """,
                    test_urns=test_urns,
                ).consume()
        finally:
            client.close()