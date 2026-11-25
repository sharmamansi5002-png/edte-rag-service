"""FastAPI application for the Semantic GraphRAG Engine."""

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.graph.neo4j_client import Neo4jClient
from src.ingestion.pipeline import ingest_document
from src.retrieval.hybrid_retriever import HybridRetriever


app = FastAPI(
    title="Semantic GraphRAG Engine",
    description="Hybrid vector and graph-based retrieval API.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request model for document retrieval."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)
    expand_graph_hops: int = Field(default=1, ge=1, le=2)


@app.get("/")
def root() -> dict[str, str]:
    """Return API information."""
    return {
        "message": "Semantic GraphRAG Engine API is running."
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Check API and database connectivity."""

    client: Neo4jClient | None = None

    try:
        client = Neo4jClient()
        client.verify_connection()

        return {
            "status": "healthy",
            "database": "connected",
        }

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {error}",
        ) from error

    finally:
        if client is not None:
            client.close()


@app.post("/api/v1/rag/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload and index a Markdown or PDF document."""

    filename = file.filename or "uploaded_document"

    suffix = Path(filename).suffix.lower()

    if suffix not in (".md", ".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only Markdown and PDF files are supported.",
        )

    temporary_path: Path | None = None

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        result = ingest_document(
    file_path=temporary_path,
    store=True,
    document_id=Path(filename).stem.lower().replace(" ", "_"),
)

        return {
            "message": "Document indexed successfully.",
            "document_urn": result.get("document_urn"),
            "chunks_indexed": result.get("chunks_indexed", 0),
            "entities": result.get("entities", {}),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {error}",
        ) from error

    finally:
        await file.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/api/v1/rag/query")
def query_endpoint(
    request: QueryRequest,
) -> dict[str, Any]:
    """Answer a question using hybrid vector and graph retrieval."""

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="The question cannot be empty.",
        )

    client: Neo4jClient | None = None

    try:
        client = Neo4jClient()
        client.verify_connection()

        retriever = HybridRetriever(client.driver)

        result = retriever.retrieve(
            question=request.question,
            top_k=request.top_k,
            expand_graph_hops=request.expand_graph_hops,
        )

        print("DEBUG - Graph context before API formatting:")
        print(result["graph_context"])

        sources = [
            {
                "chunk_urn": source["chunk_urn"],
                "score": source["score"],
                "text_snippet": source["text"][:300],
            }
            for source in result["sources"]
        ]

        graph_context = [
            {
                **entity,
                "related_actors": entity.get(
                    "related_actors", []
                ),
            }
            for entity in result["graph_context"]
        ]

        return {
            "answer": result["answer"],
            "sources": sources,
            "graph_context": graph_context,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {error}",
        ) from error

    finally:
        if client is not None:
            client.close()