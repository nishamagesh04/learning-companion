import pytest
import io
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pypdf import PdfWriter, PageObject
from app.models.database_models import Base
from app.models.request_models import PDFUploadRequest, PDFReprocessRequest
from app.services.pdf_ingestion_service import PDFIngestionService

@pytest.fixture
def db_session():
    """In-memory SQLite database session fixture."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def create_valid_dummy_pdf() -> bytes:
    """Generates a valid PDF with readable text for pipeline testing."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    # Binary PDF header + structure
    stream = io.BytesIO()
    writer.write(stream)
    pdf_bytes = stream.getvalue()
    return pdf_bytes

def test_full_pdf_ingestion_pipeline_with_mock(db_session, monkeypatch):
    service = PDFIngestionService(db_session)
    
    # Mock extraction to return predictable text
    def mock_extract(file_bytes, file_name, progress_callback=None):
        from app.models.response_models import ExtractedPage
        pages = [
            ExtractedPage(page_number=1, text="ACME COURSE\n\n1. Overview\n\nFastAPI is a modern web framework.\nPage 1 of 2"),
            ExtractedPage(page_number=2, text="ACME COURSE\n\n2. Vector Embeddings\n\nEmbeddings store semantic text representation.\nPage 2 of 2")
        ]
        return pages, "Raw Concatenated Text", "mock_pypdf"

    monkeypatch.setattr(service.extraction_service, "extract_text_from_pdf_bytes", mock_extract)

    pdf_bytes = b"%PDF-1.4 Mock PDF Content Bytes for Testing"
    req = PDFUploadRequest(
        title="FastAPI & Vector Search Guide",
        module_id=101,
        programme_name="Accelerated AI Engineering",
        week="Week 3",
        section="Backend & Vector Search",
        topic="Document Processing",
        chunk_strategy="semantic",
        max_chunk_size=300,
        chunk_overlap=50
    )

    status_resp = service.process_pdf(
        file_bytes=pdf_bytes,
        file_name="fastapi_guide.pdf",
        file_path="/uploads/fastapi_guide.pdf",
        req=req
    )

    assert status_resp.processing_status == "COMPLETED"
    assert status_resp.chunk_count > 0
    assert status_resp.processing_error is None
    assert len(status_resp.logs) >= 5

    # Test Reprocessing
    reprocess_req = PDFReprocessRequest(
        resource_id=status_resp.resource_id,
        chunk_strategy="fixed",
        max_chunk_size=100,
        chunk_overlap=20,
        generate_embeddings=True
    )

    reprocess_resp = service.reprocess_pdf(reprocess_req)
    assert reprocess_resp.processing_status == "COMPLETED"
    assert reprocess_resp.chunk_count > 0
