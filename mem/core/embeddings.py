"""OpenAI embedding wrapper."""

from typing import List, Optional

from openai import OpenAI

from mem.config import get_settings


class EmbeddingService:
    """Service for generating text embeddings via OpenAI."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.embedding_model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def embed_text(text: str) -> List[float]:
    """Convenience function to embed a single text."""
    return get_embedding_service().embed_text(text)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_service().embed_batch(texts)
