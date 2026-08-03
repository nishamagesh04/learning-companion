import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import datetime
import streamlit as st

from app.core.database import SessionLocal
from app.models.database_models import ConversationSession, ChatMessage, ChatResponse, LearningResource
from app.models.request_models import PDFUploadRequest
from app.models.response_models import ExtractedPage
from app.repositories.resource_repository import ResourceRepository
from app.repositories.chunk_repository import ChunkRepository
from app.services.pdf_ingestion_service import PDFIngestionService
from app.services.text_cleaning_service import TextCleaningService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.chat_service import answer_question

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploaded_files")
os.makedirs(STORAGE_DIR, exist_ok=True)

SUPPORTED_VIA_PIPELINE = {"pdf", "docx", "md", "csv"}

st.set_page_config(page_title="AI Learning Companion", layout="centered")
st.title("AI Learning Companion")

if "session_id" not in st.session_state:
    db = SessionLocal()
    session = ConversationSession()
    db.add(session); db.commit(); db.refresh(session)
    st.session_state.session_id = session.id
    st.session_state.history = []
    db.close()


def ingest_via_pipeline(db, file_bytes, file_name, file_path, title):
    req = PDFUploadRequest(title=title)
    pipeline = PDFIngestionService(db_session=db)
    status = pipeline.process_pdf(file_bytes, file_name, file_path, req)
    return getattr(status, "resource_id", None), getattr(status, "processing_status", None)


def ingest_txt(db, file_bytes, file_name, file_path, title):
    resource_repo = ResourceRepository(db)
    chunk_repo = ChunkRepository(db)
    cleaning_service = TextCleaningService()
    chunking_service = ChunkingService()
    embedding_service = EmbeddingService()

    raw_text = file_bytes.decode("utf-8", errors="ignore")
    resource = LearningResource(
        title=title, resource_type="TXT", file_name=file_name, file_path=file_path,
        processing_status="PENDING", version=1, is_active=True,
        uploaded_at=datetime.datetime.utcnow(),
    )
    db.add(resource); db.commit(); db.refresh(resource)
    resource_id = resource.id

    pages = [ExtractedPage(page_number=1, text=raw_text)]
    cleaned_pages, full_cleaned_text = cleaning_service.clean_extracted_pages(pages)
    resource_repo.save_resource_content(resource_id, raw_text=raw_text, cleaned_text=full_cleaned_text, extraction_method="plain_text")

    chunk_previews = chunking_service.chunk_content(
        pages=cleaned_pages, strategy="semantic", max_chunk_size=800, chunk_overlap=150, min_chunk_size=100,
    )
    embeddings = embedding_service.generate_embeddings_for_chunks(chunk_previews)
    chunk_repo.save_chunks(resource_id, chunks=chunk_previews, embeddings=embeddings, embedding_model=embedding_service.client.model_name)
    resource_repo.update_processing_status(resource_id, "COMPLETED")
    return resource_id, "COMPLETED"


def handle_upload(file):
    file_bytes = file.getvalue()
    file_name = file.name
    file_ext = file_name.split(".")[-1].lower()
    title = os.path.splitext(file_name)[0]

    safe_name = f"{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file_name}"
    file_path = os.path.join(STORAGE_DIR, safe_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    db = SessionLocal()
    try:
        if file_ext in SUPPORTED_VIA_PIPELINE:
            resource_id, status = ingest_via_pipeline(db, file_bytes, file_name, file_path, title)
        elif file_ext == "txt":
            resource_id, status = ingest_txt(db, file_bytes, file_name, file_path, title)
        else:
            return f"'{file_name}' is not a supported file type (PDF, DOCX, MD, CSV, TXT only)."

        if status == "COMPLETED":
            return f"'{file_name}' uploaded and indexed. You can ask questions about it now."
        else:
            return f"'{file_name}' was uploaded but processing status is '{status}' — something went wrong during ingestion."
    except Exception as e:
        return f"Failed to process '{file_name}': {type(e).__name__}: {e}"
    finally:
        db.close()


# --- Render existing conversation ---
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

# --- Combined input: type a question, or attach a file, in the same bar ---
prompt = st.chat_input(
    "Ask a question, or attach a file to upload...",
    accept_file="multiple",
    file_type=["pdf", "docx", "md", "csv", "txt"],
)

if prompt:
    question_text = prompt.text.strip() if prompt.text else ""
    files = prompt.files if prompt.files else []

    # Handle any attached files first
    for f in files:
        with st.chat_message("user"):
            st.write(f"📎 Uploaded: {f.name}")
        st.session_state.history.append({"role": "user", "text": f"📎 Uploaded: {f.name}"})

        with st.chat_message("assistant"):
            with st.spinner(f"Processing {f.name}..."):
                result_msg = handle_upload(f)
            st.write(result_msg)
        st.session_state.history.append({"role": "assistant", "text": result_msg})

    # Then handle the typed question, if any
    if question_text:
        st.session_state.history.append({"role": "user", "text": question_text})
        with st.chat_message("user"):
            st.write(question_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                db = SessionLocal()
                user_msg = ChatMessage(session_id=st.session_state.session_id, role="user", message_text=question_text)
                db.add(user_msg); db.commit(); db.refresh(user_msg)

                answer, sources = answer_question(db, question_text)
                st.write(answer)
                if sources:
                    with st.expander("Sources used"):
                        for c in sources:
                            st.caption(f"Chunk {c.chunk_index} (resource {c.resource_id}): {c.chunk_text[:200]}...")

                resp = ChatResponse(message_id=user_msg.id, answer_text=answer, model_name="gemini-flash-lite-latest")
                db.add(resp); db.commit()
                db.close()

        st.session_state.history.append({"role": "assistant", "text": answer})