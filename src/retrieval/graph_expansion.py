"""Expand vector search results using Neo4j graph relationships."""

from typing import Any

from neo4j import Driver


class GraphExpansion:
    """Retrieve entities and their related actors."""

    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    def expand(
        self,
        chunk_urns: list[str],
        hops: int = 1,
    ) -> list[dict[str, Any]]:
        """Retrieve connected entities using one or two graph hops."""

        if not chunk_urns:
            return []

        if hops not in (1, 2):
            raise ValueError("hops must be either 1 or 2.")

        query = """
        CALL {
            MATCH (c:DocumentChunk)-[:REFERENCES]->(e)
            WHERE c.urn IN $chunk_urns
              AND NOT e:Actor

            RETURN
                e AS entity,
                c.urn AS source_chunk,
                'direct' AS connection_type

            UNION ALL

            MATCH (source:DocumentChunk)-[:REFERENCES]->(shared)
                  <-[:REFERENCES]-(related:DocumentChunk)
                  -[:REFERENCES]->(e)
            WHERE $hops = 2
              AND source.urn IN $chunk_urns
              AND related <> source
              AND NOT shared:Actor
              AND NOT e:Actor

            RETURN
                e AS entity,
                related.urn AS source_chunk,
                'shared_entity' AS connection_type
        }

        WITH
            entity,
            collect(DISTINCT source_chunk) AS source_chunks,
            collect(DISTINCT connection_type) AS connection_types

        UNWIND source_chunks AS chunk_urn

        OPTIONAL MATCH
            (c:DocumentChunk {urn: chunk_urn})
            -[:REFERENCES]->(a:Actor)

        WITH
            entity,
            source_chunks,
            connection_types,
            collect(DISTINCT CASE
                WHEN a IS NOT NULL THEN {
                    urn: a.urn,
                    name: a.name
                }
            END) AS actors

        RETURN
            entity.urn AS entity_urn,
            entity.name AS entity_name,
            labels(entity) AS entity_type,
            source_chunks,
            [actor IN actors WHERE actor IS NOT NULL]
                AS related_actors,
            CASE
                WHEN 'direct' IN connection_types THEN 'direct'
                ELSE 'shared_entity'
            END AS connection_type

        ORDER BY entity_urn
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                chunk_urns=chunk_urns,
                hops=hops,
            )

            return [record.data() for record in result]