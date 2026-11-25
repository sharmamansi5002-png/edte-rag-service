"""Test the Neo4j database connection."""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


def test_neo4j_connection() -> None:
    """Verify that the application can connect to Neo4j."""

    load_dotenv(override=True)

    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise ValueError("NEO4J_PASSWORD is missing from your .env file.")

    with GraphDatabase.driver(
        uri,
        auth=(username, password),
    ) as driver:
        driver.verify_connectivity()

    print("SUCCESS: Connected to Neo4j!")