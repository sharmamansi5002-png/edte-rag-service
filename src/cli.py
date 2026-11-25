"""Command-line interface for the Semantic GraphRAG Engine."""

import argparse
from pathlib import Path
from typing import Any

from src.graph.neo4j_client import Neo4jClient
from src.ingestion.pipeline import ingest_document
from src.retrieval.hybrid_retriever import HybridRetriever


def index_documents(
    docs_dir: Path,
    parser: argparse.ArgumentParser,
) -> None:
    """Index Markdown and PDF documents."""

    if not docs_dir.is_dir():
        parser.error(f"Directory does not exist: {docs_dir}")

    files = sorted(
        file
        for file in docs_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in {".md", ".pdf"}
    )

    if not files:
        print(f"No Markdown or PDF documents found in {docs_dir}")
        return

    for file in files:
        try:
            result = ingest_document(file, store=True)
            print(
                f"Indexed {file.name}: "
                f"{result['chunks_indexed']} chunks"
            )
        except Exception as exc:
            print(f"Failed to index {file.name}: {exc}")


def ask_question(
    question: str,
    top_k: int = 5,
    expand_graph_hops: int = 1,
) -> None:
    """Answer a question using hybrid retrieval."""

    client = Neo4jClient()

    try:
        client.verify_connection()
        retriever = HybridRetriever(client.driver)

        result: dict[str, Any] = retriever.retrieve(
            question=question,
            top_k=top_k,
            expand_graph_hops=expand_graph_hops,
        )

        print("\nANSWER")
        print("=" * 50)
        print(result["answer"])

        print("\nSOURCES")
        print("=" * 50)

        for source in result["sources"]:
            print(f"Chunk: {source['chunk_urn']}")
            print(f"Score: {source['score']:.4f}")
            print(f"Text: {source['text'][:300]}")
            print()

        print("\nGRAPH CONTEXT")
        print("=" * 50)

        for entity in result["graph_context"]:
            print(f"Entity: {entity['entity_urn']}")
            print(f"Name: {entity.get('entity_name')}")
            print(
                f"Related actors: "
                f"{entity.get('related_actors', [])}"
            )
            print()

    finally:
        client.close()


def main() -> None:
    """Run CLI commands."""

    parser = argparse.ArgumentParser(
        description="Semantic GraphRAG Engine CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index-docs",
        help="Index Markdown and PDF documents",
    )

    index_parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing documents",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question using hybrid retrieval",
    )

    ask_parser.add_argument(
        "--query",
        required=True,
        help="Question to ask",
    )

    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of source chunks to retrieve",
    )

    ask_parser.add_argument(
        "--expand-graph-hops",
        type=int,
        choices=[1, 2],
        default=1,
        help="Number of graph expansion hops",
    )

    args = parser.parse_args()

    if args.command == "index-docs":
        index_documents(Path(args.dir), parser)

    elif args.command == "ask":
        ask_question(
            question=args.query,
            top_k=args.top_k,
            expand_graph_hops=args.expand_graph_hops,
        )


if __name__ == "__main__":
    main()