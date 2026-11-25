"""Load, chunk, embed, and optionally store documents in Neo4j."""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.graph.neo4j_client import Neo4jClient
from src.ingestion.chunker import chunk_text
from src.ingestion.document_loader import load_document
from src.ingestion.entity_extractor import extract_entities
from src.ingestion.urns import document_urn, chunk_urn
from src.retrieval.embeddings import EmbeddingService

load_dotenv()


def ingest_document(
    file_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
    *,
    store: bool = False,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Load, process, embed, and optionally store a document."""
    path = Path(file_path).resolve()
    text = load_document(path)

    if not text.strip():
        return {
            "chunks": [],
            "entities": {
    "systems": [],
    "resources": [],
    "roles": [],
    "actors": [],
},
        }

    chunk_texts = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    entities = extract_entities(text)

    document_id = (
        document_id or path.stem.lower().replace(" ", "_")
    )
    doc_urn = document_urn(document_id)

    embedding_service = EmbeddingService()
    vectors = embedding_service.embed_texts(chunk_texts)

    chunks = [
        {
            "urn": chunk_urn(document_id, index),
            "text": chunk,
            "chunk_index": index,
            "embedding": vectors[index],
        }
        for index, chunk in enumerate(chunk_texts)
    ]

    if store:
        client = Neo4jClient()
        try:
            client.verify_connection()
            client.create_constraints()
            client.store_document(
                   document_urn=doc_urn,
                title=path.name,
                uri=str(path),
                chunks=chunks,
                entities=entities,
            )
        finally:
            client.close()

    return {
        "document_urn": doc_urn,
        "chunks": chunk_texts,
        "chunks_indexed": len(chunks),
        "entities": entities,
    }