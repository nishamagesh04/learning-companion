import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import SessionLocal, init_db
from app.models.database_models import LearningResource, ResourceContent, ContentChunk
from app.repositories.resource_repository import ResourceRepository
from app.repositories.chunk_repository import ChunkRepository
from app.services.video_transcript_service import VideoTranscriptService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.models.response_models import ExtractedPage

TRANSCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "transcripts"))

def ingest_transcripts():
    init_db()
    db = SessionLocal()
    
    resource_repo = ResourceRepository(db)
    chunk_repo = ChunkRepository(db)
    transcript_service = VideoTranscriptService()
    chunking_service = ChunkingService()
    embedding_service = EmbeddingService()

    if not os.path.exists(TRANSCRIPT_DIR):
        print(f"Transcript directory not found: {TRANSCRIPT_DIR}")
        return

    files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith('.txt')]
    print(f"Found {len(files)} transcript files to ingest.")

    processed_count = 0
    skipped_count = 0

    for idx, file_name in enumerate(files, start=1):
        file_path = os.path.join(TRANSCRIPT_DIR, file_name)
        title = os.path.splitext(file_name)[0]

        # Check if already ingested
        existing = db.query(LearningResource).filter(LearningResource.file_name == file_name).first()
        if existing and existing.processing_status == "COMPLETED":
            print(f"[{idx}/{len(files)}] Skipping '{file_name}' (already indexed).")
            skipped_count += 1
            continue

        print(f"[{idx}/{len(files)}] Indexing video transcript: '{title}'...")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            pages, cleaned_text, ext_method, source_url = transcript_service.parse_transcript_text(raw_text)

            resource = LearningResource(
                title=title,
                resource_type="VIDEO",
                file_name=file_name,
                file_path=file_path,
                source_url=source_url,
                processing_status="PENDING",
                version=1,
                is_active=True,
                uploaded_at=datetime.datetime.utcnow()
            )
            db.add(resource)
            db.commit()
            db.refresh(resource)

            resource_repo.save_resource_content(
                resource.id,
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                extraction_method=ext_method
            )

            # Chunk content
            chunk_previews = chunking_service.chunk_content(
                pages=pages,
                strategy="semantic",
                max_chunk_size=800,
                chunk_overlap=150,
                min_chunk_size=100
            )

            # Generate vector embeddings
            embeddings = embedding_service.generate_embeddings_for_chunks(chunk_previews)

            # Save chunks to database
            chunk_repo.save_chunks(
                resource_id=resource.id,
                chunks=chunk_previews,
                embeddings=embeddings,
                embedding_model=embedding_service.client.model_name
            )

            resource_repo.update_processing_status(resource.id, "COMPLETED")
            processed_count += 1
            print(f"[SUCCESS] [{idx}/{len(files)}] Successfully indexed '{title}' ({len(chunk_previews)} chunks).")

        except Exception as e:
            print(f"[ERROR] Error processing '{file_name}': {type(e).__name__}: {e}")
            db.rollback()

    db.close()
    print(f"\n[DONE] Ingestion complete! Indexed: {processed_count}, Skipped: {skipped_count}, Total: {len(files)}.")

if __name__ == "__main__":
    ingest_transcripts()
