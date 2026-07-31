"""
Demo script to demonstrate the PDF Content Ingestion Pipeline.
Executes Extraction -> Cleaning -> Chunking -> Embedding -> Database Storage -> Reprocessing.
"""
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.database_models import Base
from app.models.request_models import PDFUploadRequest, PDFReprocessRequest
from app.services.pdf_ingestion_service import PDFIngestionService
from app.models.response_models import ExtractedPage

def run_demo():
    print("=" * 70)
    print("AI Learning Material Companion - PDF Ingestion Pipeline Demo")
    print("=" * 70)

    # Initialize SQLite database engine in memory
    engine = create_engine("sqlite:///./learning_companion.db", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    pipeline = PDFIngestionService(db)

    # Sample PDF raw byte content simulation
    sample_pdf_bytes = b"%PDF-1.5 Sample Document for Learning Companion Pipeline"
    file_name = "sample_ai_engineering.pdf"

    # Mock text extraction for demo purposes
    def mock_extract_demo(file_bytes, file_name):
        pages = [
            ExtractedPage(
                page_number=1,
                text="ACME UNIVERSITY - AI ENGINEERING\nPage 1 of 3\n\n"
                     "1. Introduction to RAG Architecture\n\n"
                     "Retrieval-Augmented Generation (RAG) connects LLMs to external knowledge sources.\n"
                     "By retrieving curated learning documents, hallucinations are significantly reduced.\n"
                     "This is an imple-\nmentation guide for document ingestion."
            ),
            ExtractedPage(
                page_number=2,
                text="ACME UNIVERSITY - AI ENGINEERING\nPage 2 of 3\n\n"
                     "2. PDF Text Processing Pipeline\n\n"
                     "The content ingestion pipeline processes PDF documents through distinct stages:\n"
                     "1. Text Extraction\n"
                     "2. Text Cleaning & Header Removal\n"
                     "3. Semantic Chunking\n"
                     "4. Embedding Generation\n"
                     "5. Vector Storage & Audit Logging"
            )
        ]
        return pages, "Raw PDF Content Stream...", "pypdf"

    pipeline.extraction_service.extract_text_from_pdf_bytes = mock_extract_demo

    upload_req = PDFUploadRequest(
        title="AI Engineering RAG Architecture & Ingestion Guide",
        module_id=1,
        programme_name="Accelerated AI Engineering",
        week="Week 2",
        section="Document Processing",
        topic="RAG Ingestion",
        chunk_strategy="semantic",
        max_chunk_size=400,
        chunk_overlap=50
    )

    print("\n[Step 1] Processing Uploaded PDF Document...")
    status = pipeline.process_pdf(
        file_bytes=sample_pdf_bytes,
        file_name=file_name,
        file_path=f"/uploads/{file_name}",
        req=upload_req
    )

    print(f"\n[+] Processing Status: {status.processing_status}")
    print(f"[+] Resource ID: {status.resource_id}")
    print(f"[+] Chunks Created: {status.chunk_count}")

    print("\n[Audit Logs]:")
    for log in status.logs:
        print(f"   [{log.created_at.strftime('%H:%M:%S')}] [{log.status}] [{log.processing_stage}] {log.message}")

    print("\n[Step 2] Reprocessing Resource with Fixed-Length Chunking...")
    reprocess_req = PDFReprocessRequest(
        resource_id=status.resource_id,
        chunk_strategy="fixed",
        max_chunk_size=200,
        chunk_overlap=30,
        generate_embeddings=True
    )
    reprocess_status = pipeline.reprocess_pdf(reprocess_req)

    print(f"\n[+] Reprocessing Status: {reprocess_status.processing_status}")
    print(f"[+] Updated Chunk Count: {reprocess_status.chunk_count}")

    print("\n[Reprocessing Audit Logs]:")
    for log in reprocess_status.logs[-3:]:
        print(f"   [{log.created_at.strftime('%H:%M:%S')}] [{log.status}] [{log.processing_stage}] {log.message}")

    print("\nDemo completed successfully!")
    db.close()

if __name__ == "__main__":
    run_demo()
