"""Vector search for document chunks stored in Neo4j."""

from typing import Any

from neo4j import Driver


class VectorSearch:
    """Search document chunks using their stored embeddings."""

    INDEX_NAME = "chunk_embedding"

    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    def create_index(self) -> None:
        """Create a vector index for 384-dimensional embeddings."""
        query = """
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
        FOR (c:DocumentChunk) ON (c.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }
        }
        """
        with self.driver.session() as session:
            session.run(query).consume()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the most similar document chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query = """
        CALL db.index.vector.queryNodes(
            $index_name,
            $top_k,
            $query_embedding
        )
        YIELD node, score
        RETURN node.urn AS chunk_urn,
               node.text AS text,
               score
        ORDER BY score DESC
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                index_name=self.INDEX_NAME,
                top_k=top_k,
                query_embedding=query_embedding,
            )
            return [record.data() for record in result]