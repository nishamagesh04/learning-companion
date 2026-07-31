import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database_models import Base
from app.integrations.embedding_client import GeminiEmbeddingClient
from app.models.request_models import PDFUploadRequest
from app.services.pdf_ingestion_service import PDFIngestionService
from app.models.response_models import ExtractedPage

@pytest.fixture
def db_session_large():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def test_embedding_client_batching_and_retry(monkeypatch):
    client = GeminiEmbeddingClient(api_key="", model_name="models/text-embedding-004")
    texts = [f"Text chunk {i} content" for i in range(100)]
    
    progress_count = []
    def callback(current, total):
        progress_count.append((current, total))

    embeddings = client.generate_batch_embeddings(texts, batch_size=25, progress_callback=callback)
    
    assert len(embeddings) == 100
    assert len(embeddings[0]) == 768
    assert len(progress_count) == 4
    assert progress_count[-1] == (100, 100)

def test_large_pdf_batch_ingestion_pipeline(db_session_large, monkeypatch):
    pipeline = PDFIngestionService(db_session_large)
    
    # Generate 150 pages
    pages = [
        ExtractedPage(page_number=p, text=f"COMPANY MANUAL HEADER\nPage {p} of 150\n\nSection {p}. Content paragraph for testing pipeline batching at scale.")
        for p in range(1, 151)
    ]
    
    def mock_extract(file_bytes, file_name, progress_callback=None):
        return pages, "Concatenated Text", "mock_pypdf"

    monkeypatch.setattr(pipeline.extraction_service, "extract_text_from_pdf_bytes", mock_extract)

    req = PDFUploadRequest(
        title="150 Page Manual Scaling Test",
        module_id=99,
        chunk_strategy="semantic",
        max_chunk_size=300,
        chunk_overlap=50
    )

    status_resp = pipeline.process_pdf(
        file_bytes=b"%PDF-1.5 Scaling Test Bytes",
        file_name="scaling_test_150p.pdf",
        file_path="/uploads/scaling_test_150p.pdf",
        req=req
    )

    assert status_resp.processing_status == "COMPLETED"
    assert status_resp.chunk_count >= 150
    assert status_resp.processing_error is None
