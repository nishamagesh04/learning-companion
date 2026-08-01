"""
Demo script to test and verify the Retrieval Layer implementation.
Ingests a sample document and executes vector similarity queries against stored embeddings.
"""
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.database_models import Base
from app.models.request_models import PDFUploadRequest, RetrievalRequest
from app.models.response_models import ExtractedPage
from app.services.pdf_ingestion_service import PDFIngestionService
from app.services.retrieval_service import RetrievalService

def run_retrieval_demo():
    print("=" * 75)
    print("AI Learning Companion - Vector Retrieval Layer Verification")
    print("=" * 75)

    # 1. Setup in-memory / local SQLite database session
    engine = create_engine("sqlite:///./learning_companion.db", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # 2. Ingest mock AI Engineering textbook document
    ingestion_service = PDFIngestionService(db)

    def mock_extract_demo(file_bytes, file_name, **kwargs):
        pages = [
            ExtractedPage(
                page_number=1,
                text="ACME UNIVERSITY - AI ENGINEERING MODULE 101\n"
                     "Topic: Retrieval-Augmented Generation (RAG) Fundamentals\n\n"
                     "Retrieval-Augmented Generation (RAG) enhances Large Language Models by fetching relevant "
                     "contextual chunks from a vector database before text generation. This prevents hallucinations."
            ),
            ExtractedPage(
                page_number=2,
                text="ACME UNIVERSITY - AI ENGINEERING MODULE 101\n"
                     "Topic: Vector Embedding & Similarity Metrics\n\n"
                     "Dense vector embeddings transform raw text chunks into high-dimensional numerical vectors. "
                     "Cosine similarity measures the angle between query vectors and document chunk vectors."
            ),
            ExtractedPage(
                page_number=3,
                text="ACME UNIVERSITY - AI ENGINEERING MODULE 101\n"
                     "Topic: Python & Database Storage\n\n"
                     "SQLite stores data in lightweight database tables. Python SQLAlchemy handles ORM mapping "
                     "for persistent storage of learning resource metadata and content chunks."
            )
        ]
        return pages, "Raw text stream", "pypdf"

    ingestion_service.extraction_service.extract_text_from_pdf_bytes = mock_extract_demo

    upload_req = PDFUploadRequest(
        title="AI Engineering RAG & Vector Search Manual",
        module_id=101,
        programme_name="AI Engineering",
        week="Week 3",
        section="Retrieval",
        topic="RAG Architecture",
        chunk_strategy="fixed",
        max_chunk_size=300,
        chunk_overlap=30
    )

    print("\n[Step 1] Ingesting Sample PDF Document into Vector DB...")
    status = ingestion_service.process_pdf(
        file_bytes=b"%PDF-Sample",
        file_name="rag_retrieval_guide.pdf",
        file_path="/uploads/rag_retrieval_guide.pdf",
        req=upload_req
    )
    print(f" -> Ingestion Status: {status.processing_status}")
    print(f" -> Resource ID: {status.resource_id}")
    print(f" -> Total Chunks Persisted: {status.chunk_count}")

    # 3. Test Retrieval Service
    retrieval_service = RetrievalService(db)

    queries = [
        "How does RAG prevent hallucinations in LLMs?",
        "What is cosine similarity for vector embeddings?",
        "How does SQLAlchemy store learning resources in SQLite?"
    ]

    print("\n[Step 2] Executing Vector Retrieval Queries (top_k=2)...")
    for q_text in queries:
        print("\n" + "-" * 65)
        print(f"Query: '{q_text}'")
        
        req = RetrievalRequest(
            query=q_text,
            top_k=2,
            resource_id=status.resource_id,
            min_similarity_score=0.0
        )
        response = retrieval_service.retrieve_context(req)

        print(f"Matches Found: {response.total_matches} (Requested top_k={response.top_k})")
        for rank, chunk in enumerate(response.retrieved_chunks, start=1):
            print(f"\n  [Rank #{rank}] Similarity Score: {chunk.similarity_score:.4f}")
            print(f"  [Chunk #{chunk.chunk_index} | Page {chunk.page_number} | ID={chunk.chunk_id}]")
            print(f"  Snippet: {chunk.chunk_text[:120]}...")

    print("\n" + "=" * 75)
    print("Retrieval Layer Verification Completed Successfully!")
    db.close()

if __name__ == "__main__":
    run_retrieval_demo()
