"""
End-to-End RAG Demonstration Script:
1. Ingests sample PDF text into SQLite.
2. Uses RetrievalService to fetch top-k context.
3. Formats prompt context and generates answer using Gemini LLM.
"""
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.database_models import Base
from app.models.request_models import PDFUploadRequest, RetrievalRequest
from app.models.response_models import ExtractedPage
from app.services.pdf_ingestion_service import PDFIngestionService
from app.services.retrieval_service import RetrievalService
from app.core.config import settings

def run_rag_demo():
    print("=" * 75)
    print("AI Learning Companion - Full RAG (Retrieval + Generation) Demo")
    print("=" * 75)

    # 1. DB Session
    engine = create_engine("sqlite:///./learning_companion.db", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # 2. Ingest Sample Document
    ingestion_service = PDFIngestionService(db)

    def mock_extract_demo(file_bytes, file_name, **kwargs):
        pages = [
            ExtractedPage(
                page_number=1,
                text="ACME UNIVERSITY - AI ENGINEERING MODULE 101\n"
                     "Topic: Retrieval-Augmented Generation (RAG) Fundamentals\n\n"
                     "Retrieval-Augmented Generation (RAG) enhances Large Language Models by fetching relevant "
                     "contextual chunks from a vector database before text generation. This prevents hallucinations."
            )
        ]
        return pages, "Raw text", "pypdf"

    ingestion_service.extraction_service.extract_text_from_pdf_bytes = mock_extract_demo
    upload_req = PDFUploadRequest(title="RAG Architecture", module_id=101)
    
    status = ingestion_service.process_pdf(
        file_bytes=b"%PDF", file_name="rag.pdf", file_path="/uploads/rag.pdf", req=upload_req
    )

    # 3. Execute Retrieval
    retrieval_service = RetrievalService(db)
    user_query = "How does RAG prevent AI hallucinations?"
    print(f"\n[User Question]: '{user_query}'")

    retrieval_req = RetrievalRequest(query=user_query, top_k=2, min_similarity_score=0.0)
    retrieval_res = retrieval_service.retrieve_context(retrieval_req)

    # 4. Format Context for LLM
    context_text = retrieval_service.format_context_for_prompt(retrieval_res)
    print("\n[Formatted Context Passed to LLM]:")
    print(context_text)

    # 5. Build Augmented Prompt
    rag_prompt = f"""
You are an AI Learning Companion assistant. Answer the user's question using ONLY the provided context below.

{context_text}

Question: {user_query}
Answer:
"""

    # 6. Generate Response using Gemini (if API key available)
    if settings.GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=rag_prompt
            )
            print("\n[LLM Generated Answer]:")
            print(response.text)
        except Exception as e:
            print(f"\n[Gemini LLM Call Note]: Could not call API ({e}). Augmented prompt built successfully.")
    else:
        print("\n[Note]: No GEMINI_API_KEY configured. Augmented prompt prepared successfully!")

    db.close()

if __name__ == "__main__":
    run_rag_demo()
