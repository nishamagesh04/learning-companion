from typing import List
from app.core.config import settings
from app.core.logging import get_logger
from app.integrations.embedding_client import GeminiEmbeddingClient
from app.models.response_models import ChunkPreview

logger = get_logger("embedding_service")

class EmbeddingService:
    """Orchestrates vector embedding generation for content chunks."""

    def __init__(self, client: GeminiEmbeddingClient = None):
        self.client = client or GeminiEmbeddingClient()

    def generate_embeddings_for_chunks(self, chunks: List[ChunkPreview]) -> List[List[float]]:
        """Generates embedding vectors for a list of ChunkPreview items."""
        if not chunks:
            return []

        texts = [c.chunk_text for c in chunks]
        logger.info(f"Generating embeddings for {len(chunks)} chunks using model '{self.client.model_name}'...")
        embeddings = self.client.generate_batch_embeddings(texts)
        
        # Annotate chunk preview objects
        for chunk, emb in zip(chunks, embeddings):
            chunk.has_embedding = (len(emb) > 0)

        logger.info(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings
