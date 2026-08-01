import datetime
from typing import List, Optional, Callable
from sqlalchemy.orm import Session
from app.models.database_models import ContentChunk
from app.models.response_models import ChunkPreview
from app.core.config import settings
from app.core.exceptions import DatabaseStorageError
from app.core.logging import get_logger

logger = get_logger("chunk_repository")

class ChunkRepository:
    """Handles persistence operations for chunk records and vector embeddings with batching support."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_chunks(
        self,
        resource_id: int,
        chunks: List[ChunkPreview],
        embeddings: List[List[float]],
        embedding_model: str,
        batch_size: int = settings.DB_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ContentChunk]:
        """
        Bulk inserts content chunks along with vector embeddings into database using mini-batching.
        Prevents SQLite/PostgreSQL buffer overflow locks when persisting 1,000+ chunks.
        """
        total = len(chunks)
        db_chunks: List[ContentChunk] = []

        try:
            for idx, (chunk_preview, emb_vector) in enumerate(zip(chunks, embeddings), start=1):
                db_chunk = ContentChunk(
                    resource_id=resource_id,
                    chunk_index=chunk_preview.chunk_index,
                    chunk_text=chunk_preview.chunk_text,
                    token_count=chunk_preview.token_count,
                    page_number=chunk_preview.page_number,
                    section_heading=chunk_preview.section_heading,
                    embedding=emb_vector,
                    embedding_model=embedding_model,
                    created_at=datetime.datetime.utcnow()
                )
                db_chunks.append(db_chunk)

                # Commit in mini-batches
                if idx % batch_size == 0 or idx == total:
                    start_idx = idx - len(db_chunks)
                    batch_to_add = db_chunks[start_idx:idx]
                    self.db.bulk_save_objects(batch_to_add)
                    self.db.commit()

                    if progress_callback:
                        progress_callback(idx, total)
                    elif idx % 200 == 0 or idx == total:
                        pct = int((idx / total) * 100)
                        logger.info(f"Database Storage Progress: {idx}/{total} chunks persisted ({pct}%).")

            logger.info(f"Successfully persisted all {total} chunks for resource ID={resource_id}.")
            return db_chunks

        except Exception as e:
            self.db.rollback()
            raise DatabaseStorageError(f"Failed to persist chunks for resource ID={resource_id}: {str(e)}") from e

    def delete_chunks_by_resource_id(self, resource_id: int) -> int:
        """Deletes existing chunks for a resource (used during Reprocessing)."""
        try:
            deleted_count = self.db.query(ContentChunk).filter(ContentChunk.resource_id == resource_id).delete()
            self.db.commit()
            logger.info(f"Deleted {deleted_count} previous chunks for resource ID={resource_id}.")
            return deleted_count
        except Exception as e:
            self.db.rollback()
            raise DatabaseStorageError(f"Failed to delete previous chunks for resource ID={resource_id}: {str(e)}") from e

    def get_chunks_by_resource_id(self, resource_id: int) -> List[ContentChunk]:
        """Fetches chunks for a resource ordered by chunk_index."""
        return self.db.query(ContentChunk).filter(ContentChunk.resource_id == resource_id).order_by(ContentChunk.chunk_index.asc()).all()

    def find_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        resource_id: Optional[int] = None,
        min_score: float = 0.0
    ) -> List[tuple]:
        """
        Computes Cosine Similarity between query_embedding and all persisted chunk embeddings.
        Returns a list of (ContentChunk, similarity_score) tuples sorted descending by score.
        """
        import math

        query = self.db.query(ContentChunk).filter(ContentChunk.embedding.isnot(None))
        if resource_id is not None:
            query = query.filter(ContentChunk.resource_id == resource_id)

        chunks = query.all()
        if not chunks or not query_embedding:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_embedding))
        if q_norm == 0:
            return []

        scored_chunks = []
        for chunk in chunks:
            emb = chunk.embedding
            if not emb or len(emb) != len(query_embedding):
                continue

            c_norm = math.sqrt(sum(y * y for y in emb))
            if c_norm == 0:
                continue

            dot_product = sum(x * y for x, y in zip(query_embedding, emb))
            similarity = dot_product / (q_norm * c_norm)

            if similarity >= min_score:
                scored_chunks.append((chunk, float(similarity)))

        # Sort by similarity score descending
        scored_chunks.sort(key=lambda item: item[1], reverse=True)
        return scored_chunks[:top_k]
        query = self.db.query(ContentChunk).join(ContentChunk.resource).filter(ContentChunk.embedding.isnot(None))
        if resource_id is not None:
            query = query.filter(ContentChunk.resource_id == resource_id)
        if module_id is not None:
            query = query.filter(LearningResource.module_id == module_id)

