from typing import List, Optional
from sqlalchemy.orm import Session
from app.repositories.chunk_repository import ChunkRepository
from app.integrations.embedding_client import GeminiEmbeddingClient
from app.models.request_models import RetrievalRequest
from app.models.response_models import RetrievalResponse, RetrievedChunk
from app.core.logging import get_logger

logger = get_logger("retrieval_service")

class RetrievalService:
    """Orchestrates query embedding generation and vector similarity retrieval."""

    def __init__(self, db_session: Session, embedding_client: Optional[GeminiEmbeddingClient] = None):
        self.db = db_session
        self.chunk_repo = ChunkRepository(db_session)
        self.embedding_client = embedding_client or GeminiEmbeddingClient()

    def retrieve_context(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Executes end-to-end vector retrieval for a given query request.
        1. Embeds query text using Gemini Embedding Client.
        2. Computes cosine similarity scores against persisted DB chunks.
        3. Formats and returns top-k retrieved chunks.
        """
        logger.info(f"Executing retrieval query: '{request.query}' (top_k={request.top_k})")

        # Step 1: Generate embedding vector for user search query
        query_vector = self.embedding_client.generate_single_embedding_with_retry(request.query)

        # Step 2: Perform vector similarity search in database repository
        scored_chunks = self.chunk_repo.find_similar_chunks(
            query_embedding=query_vector,
            top_k=request.top_k,
            resource_id=request.resource_id,
            min_score=request.min_similarity_score
        )

        # Step 3: Build response payload with top-k matching chunks
        results: List[RetrievedChunk] = []
        for chunk, score in scored_chunks:
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    resource_id=chunk.resource_id,
                    chunk_index=chunk.chunk_index,
                    chunk_text=chunk.chunk_text,
                    similarity_score=round(score, 4),
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    token_count=chunk.token_count
                )
            )

        logger.info(f"Retrieved {len(results)} relevant chunks for query '{request.query}'.")
        return RetrievalResponse(
            query=request.query,
            total_matches=len(results),
            top_k=request.top_k,
            retrieved_chunks=results
        )

    def format_context_for_prompt(self, response: RetrievalResponse) -> str:
        """
        Formats retrieved chunks into a clean, structured context string 
        ready to be injected directly into the LLM prompt.
        """
        if not response.retrieved_chunks:
            return "No relevant learning material context found."

        context_blocks = []
        for rank, chunk in enumerate(response.retrieved_chunks, start=1):
            header = f"[Source #{rank} | Page {chunk.page_number or 'N/A'}]"
            if chunk.section_heading:
                header += f" (Section: {chunk.section_heading})"
            
            block = f"{header}\n{chunk.chunk_text}"
            context_blocks.append(block)

        formatted_context = (
            "=== RETRIEVED LEARNING MATERIAL CONTEXT ===\n"
            + "\n\n".join(context_blocks)
            + "\n==========================================="
        )
        return formatted_context

