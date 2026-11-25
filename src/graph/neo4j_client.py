"""Neo4j connection and document graph storage."""

import os
import re
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

from src.retrieval.vector_search import VectorSearch
from src.ingestion.urns import (
    actor_urn,
    resource_urn,
    role_urn,
    system_urn,
)


class Neo4jClient:
    """Manage Neo4j connections and store document graph data."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        load_dotenv(override=True)

        self._uri = uri if uri is not None else (
    os.getenv("NEO4J_URI") or "bolt://127.0.0.1:7687"
)
        self._username = username if username is not None else (
    os.getenv("NEO4J_USERNAME") or "neo4j"
)
        self._password = password or os.getenv(
            "NEO4J_PASSWORD", ""
        )

        if not self._password:
            raise ValueError("NEO4J_PASSWORD is missing.")

        self._driver: Driver = GraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )
        self.vector_search = VectorSearch(self._driver)

    @property
    def driver(self) -> Driver:
        """Return the Neo4j database driver."""
        return self._driver

    def verify_connection(self) -> None:
        """Check the Neo4j connection."""
        self._driver.verify_connectivity()

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    def create_vector_index(self) -> None:
        """Create the vector index for document chunks."""
        self.vector_search.create_index()

    def create_constraints(self) -> None:
        """Create uniqueness constraints for graph nodes."""

        queries = [
            "CREATE CONSTRAINT document_urn IF NOT EXISTS "
            "FOR (n:Document) REQUIRE n.urn IS UNIQUE",
            "CREATE CONSTRAINT chunk_urn IF NOT EXISTS "
            "FOR (n:DocumentChunk) REQUIRE n.urn IS UNIQUE",
            "CREATE CONSTRAINT system_urn IF NOT EXISTS "
            "FOR (n:System) REQUIRE n.urn IS UNIQUE",
            "CREATE CONSTRAINT resource_urn IF NOT EXISTS "
            "FOR (n:Resource) REQUIRE n.urn IS UNIQUE",
            "CREATE CONSTRAINT role_urn IF NOT EXISTS "
            "FOR (n:Role) REQUIRE n.urn IS UNIQUE",
            "CREATE CONSTRAINT actor_urn IF NOT EXISTS "
            "FOR (n:Actor) REQUIRE n.urn IS UNIQUE",
        ]

        with self._driver.session() as session:
            for query in queries:
                session.run(query).consume()

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        """Convert an entity name into a consistent URN identifier."""

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            name.casefold(),
        ).strip("_")

        if not normalized:
            raise ValueError("Entity name cannot be empty.")

        return normalized

    def store_document(
        self,
        document_urn: str,
        title: str,
        uri: str,
        chunks: list[dict[str, Any]],
        entities: dict[str, list[Any]],
    ) -> None:
        """Store a document, its chunks, and entity links."""

        with self._driver.session() as session:
            session.execute_write(
                self._store_document_transaction,
                document_urn,
                title,
                uri,
                chunks,
                entities,
            )

    @staticmethod
    def _store_document_transaction(
        tx: Any,
        document_urn: str,
        title: str,
        uri: str,
        chunks: list[dict[str, Any]],
        entities: dict[str, list[Any]],
    ) -> None:
        """Store graph data in one transaction."""

        tx.run(
            """
            MERGE (d:Document {urn: $urn})
            SET d.title = $title,
                d.uri = $uri
            """,
            urn=document_urn,
            title=title,
            uri=uri,
        ).consume()

        # Remove old entity links and obsolete chunks for this document.
        tx.run(
            """
            MATCH (d:Document {urn: $urn})-[rel:CONTAINS]->(c:DocumentChunk)
            OPTIONAL MATCH (c)-[ref:REFERENCES]->()
            DELETE ref, rel
            WITH c
            OPTIONAL MATCH (other:Document)-[:CONTAINS]->(c)
            WITH c, count(other) AS owners
            WHERE owners = 0
            DETACH DELETE c
            """,
            urn=document_urn,
        ).consume()
        
        chunk_urns = [chunk["urn"] for chunk in chunks]

        for chunk in chunks:
            tx.run(
                """
                MERGE (c:DocumentChunk {urn: $urn})
                SET c.text = $text,
                    c.chunk_index = $chunk_index,
                    c.embedding = $embedding
                WITH c
                MATCH (d:Document {urn: $document_urn})
                MERGE (d)-[:CONTAINS]->(c)
                """,
                urn=chunk["urn"],
                text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                embedding=chunk["embedding"],
                document_urn=document_urn,
            ).consume()

                # Store systems, roles, and actors.
        entity_specs = [
            ("systems", "System", system_urn),
            ("roles", "Role", role_urn),
            ("actors", "Actor", actor_urn),
        ]

        for entity_type, label, urn_function in entity_specs:
            for entity in entities.get(entity_type, []):
                entity_id = urn_function(entity)

                tx.run(
                    f"""
                    MERGE (e:{label} {{urn: $entity_urn}})
                    SET e.name = $name
                    WITH e
                    MATCH (c:DocumentChunk)
                    WHERE c.urn IN $chunk_urns
                      AND toLower(c.text) CONTAINS toLower($name)
                    MERGE (c)-[:REFERENCES]->(e)
                    """,
                    entity_urn=entity_id,
                    name=entity,
                    chunk_urns=chunk_urns,
                ).consume()

        # Store resources using their system and resource type.
        for resource in entities.get("resource_details", []):
            entity_id = resource_urn(
                resource["system"] or "unknown",
                resource["type"],
                resource["name"],
            )

            tx.run(
                """
                MERGE (e:Resource {urn: $entity_urn})
                SET e.name = $name,
                    e.resource_type = $resource_type
                WITH e
                MATCH (c:DocumentChunk)
                WHERE c.urn IN $chunk_urns
                  AND toLower(c.text) CONTAINS toLower($name)
                MERGE (c)-[:REFERENCES]->(e)
                """,
                entity_urn=entity_id,
                name=resource["name"],
                resource_type=resource["type"],
                chunk_urns=chunk_urns,
            ).consume()