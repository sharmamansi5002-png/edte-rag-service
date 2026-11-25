"""Generate text embeddings for semantic retrieval."""

from typing import Any, cast


class EmbeddingService:
    """Generate vector embeddings using a sentence-transformer model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        # Import only when a real embedding model is required.
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for one text."""
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
        )
        return cast(list[float], vector.tolist())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
        )
        return cast(list[list[float]], vectors.tolist())