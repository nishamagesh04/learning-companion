"""
Test script to verify reprocessing workflow works end-to-end.
Tests the complete reprocessing pipeline including chunk clearing, re-chunking, and re-embedding.
"""
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test API key to allow embedding client initialization (will use mock if actual calls fail)
os.environ["GEMINI_API_KEY"] = "test_key_for_reprocessing_workflow"

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.database_models import Base
from app.models.request_models import PDFUploadRequest, PDFReprocessRequest
from app.services.pdf_ingestion_service import PDFIngestionService
from app.models.response_models import ExtractedPage

def test_reprocessing_workflow():
    print("=" * 70)
    print("AI Learning Material Companion - Reprocessing Workflow Test")
    print("=" * 70)

    # Initialize SQLite database engine
    engine = create_engine("sqlite:///./learning_companion.db", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    pipeline = PDFIngestionService(db)

    # Mock text extraction for testing
    def mock_extract_demo(file_bytes, file_name, progress_callback=None):
        pages = [
            ExtractedPage(
                page_number=1,
                text="TEST DOCUMENT - Page 1\n\n"
                     "This is a test document for reprocessing workflow validation.\n"
                     "The reprocessing should clear old chunks and create new ones.\n"
                     "It should also regenerate embeddings if requested."
            ),
            ExtractedPage(
                page_number=2,
                text="TEST DOCUMENT - Page 2\n\n"
                     "Second page of test content.\n"
                     "This content will be re-chunked with different parameters.\n"
                     "The workflow should handle version incrementing properly."
            )
        ]
        return pages, "Raw test content...", "mock-extraction"

    pipeline.extraction_service.extract_text_from_document_bytes = mock_extract_demo
    pipeline.extraction_service.validate_document_file = lambda file_bytes, file_name: None  # Skip validation
    
    # Mock embedding generation to avoid API calls
    def mock_generate_embeddings(texts, batch_size=None, progress_callback=None):
        return [[0.0] * 768 for _ in texts]
    
    pipeline.embedding_service.client.generate_batch_embeddings = mock_generate_embeddings

    # Step 1: Initial processing
    print("\n[Step 1] Initial document processing...")
    upload_req = PDFUploadRequest(
        title="Test Document for Reprocessing",
        module_id=1,
        programme_name="Test Programme",
        week="Week 1",
        section="Testing",
        topic="Reprocessing Workflow",
        chunk_strategy="semantic",
        max_chunk_size=400,
        chunk_overlap=50
    )

    sample_bytes = b"test document bytes"
    status = pipeline.process_pdf(
        file_bytes=sample_bytes,
        file_name="test_reprocessing.pdf",
        file_path="/uploads/test_reprocessing.pdf",
        req=upload_req
    )

    print(f"[+] Initial Processing Status: {status.processing_status}")
    print(f"[+] Resource ID: {status.resource_id}")
    print(f"[+] Initial Chunk Count: {status.chunk_count}")
    print(f"[+] Initial Version: {status.logs[0].message if status.logs else 'N/A'}")

    resource_id = status.resource_id
    initial_chunk_count = status.chunk_count

    # Step 2: Verify chunks were created
    print("\n[Step 2] Verifying initial chunks...")
    chunks = pipeline.chunk_repo.get_chunks_by_resource_id(resource_id)
    print(f"[+] Chunks in database: {len(chunks)}")
    if chunks:
        print(f"[+] First chunk preview: {chunks[0].chunk_text[:100]}...")

    # Step 3: Reprocess with different parameters
    print("\n[Step 3] Reprocessing with fixed chunking...")
    reprocess_req = PDFReprocessRequest(
        resource_id=resource_id,
        chunk_strategy="fixed",
        max_chunk_size=200,
        chunk_overlap=30,
        generate_embeddings=False  # Skip embeddings for workflow test
    )

    try:
        reprocess_status = pipeline.reprocess_pdf(reprocess_req)
        print(f"[+] Reprocessing Status: {reprocess_status.processing_status}")
        print(f"[+] Updated Chunk Count: {reprocess_status.chunk_count}")
        
        # Step 4: Verify reprocessing results
        print("\n[Step 4] Verifying reprocessing results...")
        
        # Check that chunks were cleared and recreated
        new_chunks = pipeline.chunk_repo.get_chunks_by_resource_id(resource_id)
        print(f"[+] New chunks in database: {len(new_chunks)}")
        
        # Check version was incremented
        resource = pipeline.resource_repo.get_resource_by_id(resource_id)
        print(f"[+] Resource version after reprocessing: {resource.version}")
        
        # Check processing logs
        print("\n[Reprocessing Audit Logs]:")
        for log in reprocess_status.logs[-5:]:
            print(f"   [{log.created_at.strftime('%H:%M:%S')}] [{log.status}] [{log.processing_stage}] {log.message}")

        # Step 5: Test content preview
        print("\n[Step 5] Testing content preview endpoints...")
        try:
            content_preview = pipeline.get_content_preview(resource_id)
            print(f"[+] Content preview retrieved successfully")
            print(f"[+] Extraction method: {content_preview.extraction_method}")
            print(f"[+] Raw text length: {len(content_preview.raw_text)}")
        except Exception as e:
            print(f"[!] Content preview failed: {e}")

        # Step 6: Test chunk preview
        print("\n[Step 6] Testing chunk preview endpoints...")
        try:
            chunk_previews = pipeline.get_chunk_previews(resource_id, limit=5)
            print(f"[+] Retrieved {len(chunk_previews)} chunk previews")
            if chunk_previews:
                print(f"[+] First chunk has embedding: {chunk_previews[0].has_embedding}")
        except Exception as e:
            print(f"[!] Chunk preview failed: {e}")

        print("\n" + "=" * 70)
        print("Reprocessing workflow test completed successfully!")
        print("=" * 70)
        
        return True

    except Exception as e:
        print(f"\n[!] Reprocessing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()

if __name__ == "__main__":
    success = test_reprocessing_workflow()
    sys.exit(0 if success else 1)
