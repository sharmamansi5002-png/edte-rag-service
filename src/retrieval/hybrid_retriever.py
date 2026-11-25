"""Combine vector search, graph expansion, and Gemini answer generation."""

import json
from typing import Any

from neo4j import Driver

from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_search import VectorSearch
from src.retrieval.graph_expansion import GraphExpansion
from src.llm.gemini_service import GeminiService


class HybridRetriever:
    """Retrieve relevant chunks, expand graph context, and generate answers."""

    def __init__(self, driver: Driver) -> None:
        self.embedding_service = EmbeddingService()
        self.vector_search = VectorSearch(driver)
        self.graph_expansion = GraphExpansion(driver)
        self.gemini_service = GeminiService()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        expand_graph_hops: int = 1,
    ) -> dict[str, Any]:
        """Retrieve information and generate an answer using Gemini."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if expand_graph_hops not in (1, 2):
            raise ValueError("Graph expansion hops must be 1 or 2.")

        # Step 1: Convert the question into an embedding.
        query_embedding = self.embedding_service.embed_text(question)

        # Step 2: Find relevant document chunks.
        sources = self.vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Step 3: Extract the chunk URNs.
        chunk_urns = [
            source["chunk_urn"]
            for source in sources
        ]

        # Step 4: Expand the graph around those chunks.
        graph_context = self.graph_expansion.expand(
            chunk_urns=chunk_urns,
            hops=expand_graph_hops,
        )

        # Step 5: Prepare the retrieved information for Gemini.
        if not sources:
            answer = (
                "I could not find relevant information in the indexed documents."
            )
        else:
            prompt = f"""
You are a helpful assistant answering questions about technical systems.

Answer the user's question using ONLY the retrieved document chunks
and graph context below.

If the information is not available in the provided context,
clearly say that it could not be found.

Do not invent facts.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CHUNKS:
{json.dumps(sources, indent=2, ensure_ascii=False)}

GRAPH CONTEXT:
{json.dumps(graph_context, indent=2, ensure_ascii=False)}

Provide a clear and concise answer.
"""

            answer = self.gemini_service.generate_answer(prompt)

        # Step 6: Return the answer and supporting information.
        return {
            "answer": answer,
            "sources": sources,
            "graph_context": graph_context,
        }