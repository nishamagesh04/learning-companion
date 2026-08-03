import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.exceptions import IngestionPipelineError
from app.core.logging import get_logger
from app.models.request_models import PDFUploadRequest, PDFReprocessRequest
from app.models.response_models import IngestionStatusResponse, ProcessingLogItem, ExtractedPage, ChunkPreview, ExtractedContentPreview
from app.repositories.resource_repository import ResourceRepository
from app.repositories.chunk_repository import ChunkRepository
from app.services.text_extraction_service import TextExtractionService
from app.services.text_cleaning_service import TextCleaningService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService

logger = get_logger("pdf_ingestion_service")

class PDFIngestionService:
    """Master orchestrator for the PDF content ingestion and reprocessing pipeline."""

    def __init__(
        self,
        db_session: Session,
        extraction_service: TextExtractionService = None,
        cleaning_service: TextCleaningService = None,
        chunking_service: ChunkingService = None,
        embedding_service: EmbeddingService = None
    ):
        self.db = db_session
        self.resource_repo = ResourceRepository(db_session)
        self.chunk_repo = ChunkRepository(db_session)
        
        self.extraction_service = extraction_service or TextExtractionService()
        self.cleaning_service = cleaning_service or TextCleaningService()
        self.chunking_service = chunking_service or ChunkingService()
        self.embedding_service = embedding_service or EmbeddingService()

    def process_pdf(self, file_bytes: bytes, file_name: str, file_path: str, req: PDFUploadRequest) -> IngestionStatusResponse:
        """
        Executes the end-to-end PDF Ingestion Pipeline:
        Validation -> Extraction -> Cleaning -> Chunking -> Embedding -> Storage -> Audit Logging.
        """
        resource = self.resource_repo.create_resource(file_name=file_name, file_path=file_path, req=req)
        resource_id = resource.id

        try:
            # Stage 1: Validation
            self.resource_repo.add_processing_log(resource_id, "VALIDATION", "INFO", f"Starting {file_name.split('.')[-1].upper()} file validation.")
            self.extraction_service.validate_document_file(file_bytes, file_name)
            self.resource_repo.add_processing_log(resource_id, "VALIDATION", "SUCCESS", f"{file_name.split('.')[-1].upper()} file structure and size validated.")

            # Stage 2: Text Extraction
            self.resource_repo.update_processing_status(resource_id, "EXTRACTING")
            self.resource_repo.add_processing_log(resource_id, "EXTRACTION", "INFO", "Extracting text page-by-page.")
            
            def extraction_progress(current, total):
                if current % 100 == 0 or current == total:
                    pct = int((current / total) * 100)
                    self.resource_repo.add_processing_log(
                        resource_id, "EXTRACTION", "INFO", f"Extraction progress: {current}/{total} pages ({pct}%)."
                    )

            extracted_pages, raw_text, ext_method = self.extraction_service.extract_text_from_document_bytes(
                file_bytes, file_name, progress_callback=extraction_progress
            )
            self.resource_repo.add_processing_log(
                resource_id, "EXTRACTION", "SUCCESS",
                f"Extracted {len(extracted_pages)} pages ({len(raw_text)} chars) using {ext_method}."
            )

            # Stage 3: Text Cleaning
            self.resource_repo.update_processing_status(resource_id, "CLEANING")
            self.resource_repo.add_processing_log(resource_id, "CLEANING", "INFO", "Cleaning headers/footers, hyphens, and whitespace.")
            cleaned_pages, full_cleaned_text = self.cleaning_service.clean_extracted_pages(extracted_pages)
            self.resource_repo.save_resource_content(resource_id, raw_text=raw_text, cleaned_text=full_cleaned_text, extraction_method=ext_method)
            self.resource_repo.add_processing_log(resource_id, "CLEANING", "SUCCESS", "Saved raw and cleaned extracted text.")

            # Stage 4: Content Chunking
            self.resource_repo.update_processing_status(resource_id, "CHUNKING")
            strategy = req.chunk_strategy or settings.DEFAULT_CHUNK_STRATEGY
            max_size = req.max_chunk_size or settings.MAX_CHUNK_SIZE
            overlap = req.chunk_overlap or settings.CHUNK_OVERLAP
            min_size = req.min_chunk_size or settings.MIN_CHUNK_SIZE

            self.resource_repo.add_processing_log(
                resource_id, "CHUNKING", "INFO",
                f"Executing {strategy} chunking (max_size={max_size}, overlap={overlap})."
            )
            chunk_previews = self.chunking_service.chunk_content(
                pages=cleaned_pages,
                strategy=strategy,
                max_chunk_size=max_size,
                chunk_overlap=overlap,
                min_chunk_size=min_size
            )
            self.resource_repo.add_processing_log(resource_id, "CHUNKING", "SUCCESS", f"Created {len(chunk_previews)} chunks.")

            # Stage 5: Embedding Generation
            self.resource_repo.update_processing_status(resource_id, "EMBEDDING")
            self.resource_repo.add_processing_log(
                resource_id, "EMBEDDING", "INFO",
                f"Generating vector embeddings for {len(chunk_previews)} chunks in mini-batches."
            )

            def embedding_progress(current, total):
                if current % 300 == 0 or current == total:
                    pct = int((current / total) * 100)
                    self.resource_repo.add_processing_log(
                        resource_id, "EMBEDDING", "INFO", f"Embedding progress: {current}/{total} chunks ({pct}%)."
                    )

            embeddings = self.embedding_service.client.generate_batch_embeddings(
                texts=[c.chunk_text for c in chunk_previews],
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                progress_callback=embedding_progress
            )
            self.resource_repo.add_processing_log(resource_id, "EMBEDDING", "SUCCESS", f"Generated {len(embeddings)} vector embeddings.")

            # Stage 6: Persistence & Complete
            def db_progress(current, total):
                if current % 500 == 0 or current == total:
                    pct = int((current / total) * 100)
                    self.resource_repo.add_processing_log(
                        resource_id, "PERSISTENCE", "INFO", f"Storage progress: {current}/{total} chunks ({pct}%)."
                    )

            self.chunk_repo.save_chunks(
                resource_id=resource_id,
                chunks=chunk_previews,
                embeddings=embeddings,
                embedding_model=self.embedding_service.client.model_name,
                batch_size=settings.DB_BATCH_SIZE,
                progress_callback=db_progress
            )
            self.resource_repo.update_processing_status(resource_id, "COMPLETED")
            self.resource_repo.add_processing_log(resource_id, "PERSISTENCE", "SUCCESS", "PDF processing pipeline completed successfully.")

            return self.get_ingestion_status(resource_id)

        except IngestionPipelineError as e:
            error_msg = str(e)
            logger.error(f"Pipeline error for resource ID={resource_id}: {error_msg}")
            self.resource_repo.update_processing_status(resource_id, "FAILED", error_message=error_msg)
            self.resource_repo.add_processing_log(resource_id, "ERROR", "ERROR", f"Pipeline failed: {error_msg}")
            return self.get_ingestion_status(resource_id)

        except Exception as e:
            error_msg = f"Unexpected pipeline failure: {str(e)}"
            logger.exception(f"Unhandled error for resource ID={resource_id}")
            self.resource_repo.update_processing_status(resource_id, "FAILED", error_message=error_msg)
            self.resource_repo.add_processing_log(resource_id, "ERROR", "ERROR", error_msg)
            return self.get_ingestion_status(resource_id)

    def reprocess_pdf(self, req: PDFReprocessRequest) -> IngestionStatusResponse:
        """
        Reprocesses an existing PDF resource by clearing previous chunks and re-executing
        cleaning, chunking, and embedding generation with updated parameters.
        """
        resource_id = req.resource_id
        resource = self.resource_repo.get_resource_by_id(resource_id)
        if not resource:
            raise IngestionPipelineError(f"Cannot reprocess resource ID={resource_id}: Resource not found.")

        if not resource.content or not resource.content.cleaned_text:
            raise IngestionPipelineError(f"Cannot reprocess resource ID={resource_id}: Cleaned text content is missing.")

        logger.info(f"Initiating reprocessing for resource ID={resource_id} (Version {resource.version + 1})...")

        try:
            self.resource_repo.update_processing_status(resource_id, "PENDING")
            self.resource_repo.increment_resource_version(resource_id)
            self.resource_repo.add_processing_log(
                resource_id, "REPROCESSING", "INFO",
                f"Starting reprocessing (Version={resource.version}). Clearing prior chunks."
            )

            # Step 1: Clear prior chunks
            self.chunk_repo.delete_chunks_by_resource_id(resource_id)

            # Step 2: Extract cleaned page representations from stored cleaned text
            # Reconstruct page bounds if markers exist or treat as single block
            raw_text = resource.content.cleaned_text
            page_blocks = raw_text.split("## Page ")
            cleaned_pages = []

            if len(page_blocks) > 1:
                for block in page_blocks[1:]:
                    lines = block.splitlines()
                    p_num = int(lines[0].strip()) if lines[0].strip().isdigit() else 1
                    p_text = "\n".join(lines[1:]).strip()
                    cleaned_pages.append(ExtractedPage(page_number=p_num, text=p_text))
            else:
                cleaned_pages.append(ExtractedPage(page_number=1, text=raw_text))

            # Step 3: Re-chunk
            self.resource_repo.update_processing_status(resource_id, "CHUNKING")
            strategy = req.chunk_strategy or settings.DEFAULT_CHUNK_STRATEGY
            max_size = req.max_chunk_size or settings.MAX_CHUNK_SIZE
            overlap = req.chunk_overlap or settings.CHUNK_OVERLAP

            self.resource_repo.add_processing_log(
                resource_id, "CHUNKING", "INFO",
                f"Reprocessing with strategy={strategy}, max_size={max_size}, overlap={overlap}."
            )
            chunk_previews = self.chunking_service.chunk_content(
                pages=cleaned_pages,
                strategy=strategy,
                max_chunk_size=max_size,
                chunk_overlap=overlap
            )

            # Step 4: Re-embed if requested
            embeddings = []
            if req.generate_embeddings:
                self.resource_repo.update_processing_status(resource_id, "EMBEDDING")
                embeddings = self.embedding_service.generate_embeddings_for_chunks(chunk_previews)
            else:
                embeddings = [[] for _ in chunk_previews]

            # Step 5: Persist updated chunks
            self.chunk_repo.save_chunks(
                resource_id=resource_id,
                chunks=chunk_previews,
                embeddings=embeddings,
                embedding_model=self.embedding_service.client.model_name if req.generate_embeddings else "None"
            )

            self.resource_repo.update_processing_status(resource_id, "COMPLETED")
            self.resource_repo.add_processing_log(resource_id, "REPROCESSING", "SUCCESS", "Reprocessing completed successfully.")

            return self.get_ingestion_status(resource_id)

        except Exception as e:
            error_msg = f"Reprocessing failed: {str(e)}"
            logger.error(f"Reprocessing error for ID={resource_id}: {error_msg}")
            self.resource_repo.update_processing_status(resource_id, "FAILED", error_message=error_msg)
            self.resource_repo.add_processing_log(resource_id, "REPROCESSING", "ERROR", error_msg)
            return self.get_ingestion_status(resource_id)

    def get_ingestion_status(self, resource_id: int) -> IngestionStatusResponse:
        """Retrieves current processing status, errors, chunk counts, and logs."""
        resource = self.resource_repo.get_resource_by_id(resource_id)
        if not resource:
            raise IngestionPipelineError(f"Resource ID={resource_id} not found.")

        chunks = self.chunk_repo.get_chunks_by_resource_id(resource_id)
        log_items = [
            ProcessingLogItem(
                processing_stage=log.processing_stage,
                status=log.status,
                message=log.message,
                created_at=log.created_at
            ) for log in resource.logs
        ]

        return IngestionStatusResponse(
            resource_id=resource.id,
            title=resource.title,
            file_name=resource.file_name,
            processing_status=resource.processing_status,
            processing_error=resource.processing_error,
            chunk_count=len(chunks),
            uploaded_at=resource.uploaded_at,
            processed_at=resource.processed_at,
            logs=log_items
        )

    def get_content_preview(self, resource_id: int) -> ExtractedContentPreview:
        """Retrieves raw and cleaned text content preview for admin inspection."""
        
        resource = self.resource_repo.get_resource_by_id(resource_id)
        if not resource:
            raise IngestionPipelineError(f"Resource ID={resource_id} not found.")

        content = self.resource_repo.get_resource_content_preview(resource_id)
        if not content:
            raise IngestionPipelineError(f"Content not found for resource ID={resource_id}.")

        return ExtractedContentPreview(
            resource_id=resource_id,
            file_name=resource.file_name,
            extraction_method=content.extraction_method,
            total_pages=len(content.cleaned_text.split("--- Page")) if "--- Page" in content.cleaned_text else 1,
            raw_text=content.raw_text[:5000] + "..." if len(content.raw_text) > 5000 else content.raw_text,
            cleaned_text=content.cleaned_text[:5000] + "..." if len(content.cleaned_text) > 5000 else content.cleaned_text
        )

    def get_chunk_previews(self, resource_id: int, limit: int = 50) -> List[ChunkPreview]:
        """Retrieves chunk previews for admin inspection."""
        chunks = self.chunk_repo.get_chunks_by_resource_id(resource_id)
        if not chunks:
            raise IngestionPipelineError(f"No chunks found for resource ID={resource_id}.")

        return [
            ChunkPreview(
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text[:1000] + "..." if len(chunk.chunk_text) > 1000 else chunk.chunk_text,
                token_count=chunk.token_count,
                page_number=chunk.page_number,
                section_heading=chunk.section_heading,
                has_embedding=chunk.embedding is not None and len(chunk.embedding) > 0
            )
            for chunk in chunks[:limit]
        ]

    def get_single_chunk_preview(self, resource_id: int, chunk_index: int) -> ChunkPreview:
        """Retrieves a single chunk preview for admin inspection."""
        chunk = self.chunk_repo.get_chunk_preview(resource_id, chunk_index)
        if not chunk:
            raise IngestionPipelineError(f"Chunk {chunk_index} not found for resource ID={resource_id}.")

        return ChunkPreview(
            chunk_index=chunk.chunk_index,
            chunk_text=chunk.chunk_text,
            token_count=chunk.token_count,
            page_number=chunk.page_number,
            section_heading=chunk.section_heading,
            has_embedding=chunk.embedding is not None and len(chunk.embedding) > 0
        )
