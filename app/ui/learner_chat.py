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

st.set_page_config(page_title="AI Learning Companion", layout="wide")


# ---------------------------------------------------------------------------
# Ingestion helpers (unchanged from before)
# ---------------------------------------------------------------------------
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


def handle_upload(file_bytes, file_name):
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


# ---------------------------------------------------------------------------
# KB (Knowledge Base) helpers - direct queries for now.
# TODO: swap for ResourceRepository.list_all_resources() / delete_resource()
# once Nisha adds them - keep this function signature the same so the panel
# code below doesn't need to change.
# ---------------------------------------------------------------------------
def list_kb_files(db):
    return db.query(LearningResource).order_by(LearningResource.uploaded_at.desc()).all()


def delete_kb_file(db, resource_id: int):
    chunk_repo = ChunkRepository(db)
    chunk_repo.delete_chunks_by_resource_id(resource_id)

    from app.models.database_models import ResourceContent
    db.query(ResourceContent).filter(ResourceContent.resource_id == resource_id).delete()

    db.query(LearningResource).filter(LearningResource.id == resource_id).delete()
    db.commit()


# ---------------------------------------------------------------------------
# Chat history helpers (left panel)
# ---------------------------------------------------------------------------
def list_sessions(db):
    return db.query(ConversationSession).order_by(ConversationSession.created_at.desc()).all()


def session_preview_label(db, session: ConversationSession) -> str:
    first_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .first()
    )
    if first_msg and first_msg.message_text:
        text = first_msg.message_text.strip()
        return (text[:40] + "...") if len(text) > 40 else text
    return f"New chat - {session.created_at.strftime('%b %d, %H:%M')}"


def load_history_for_session(db, session_id: int):
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    history = []
    for m in messages:
        history.append({"role": "user", "text": m.message_text})
        if m.response:
            history.append({"role": "assistant", "text": m.response.answer_text})
    return history


def start_new_session():
    db = SessionLocal()
    session = ConversationSession()
    db.add(session); db.commit(); db.refresh(session)
    st.session_state.session_id = session.id
    st.session_state.history = []
    db.close()


def switch_to_session(session_id: int):
    db = SessionLocal()
    st.session_state.session_id = session_id
    st.session_state.history = load_history_for_session(db, session_id)
    db.close()


if "session_id" not in st.session_state:
    start_new_session()


# ---------------------------------------------------------------------------
# LEFT PANEL - Chat history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Chat History")
    if st.button("+ New chat", use_container_width=True):
        start_new_session()
        st.rerun()

    st.divider()

    db = SessionLocal()
    sessions = list_sessions(db)
    for s in sessions:
        label = session_preview_label(db, s)
        is_active = s.id == st.session_state.session_id
        if st.button(("● " if is_active else "") + label, key=f"session_{s.id}", use_container_width=True):
            switch_to_session(s.id)
            st.rerun()
    db.close()


# ---------------------------------------------------------------------------
# CENTER (chat) + RIGHT (knowledge base) PANELS
# ---------------------------------------------------------------------------
chat_col, kb_col = st.columns([3, 1])

with chat_col:
    st.title("AI Learning Companion")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    prompt = st.chat_input(
        "Ask a question, or attach a file to upload...",
        accept_file="multiple",
        file_type=["pdf", "docx", "md", "csv", "txt"],
    )

    if prompt:
        question_text = prompt.text.strip() if prompt.text else ""
        files = prompt.files if prompt.files else []

        for f in files:
            with st.chat_message("user"):
                st.write(f"📎 Uploaded: {f.name}")
            st.session_state.history.append({"role": "user", "text": f"📎 Uploaded: {f.name}"})

            with st.chat_message("assistant"):
                with st.spinner(f"Processing {f.name}..."):
                    result_msg = handle_upload(f.getvalue(), f.name)
                st.write(result_msg)
            st.session_state.history.append({"role": "assistant", "text": result_msg})

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

with kb_col:
    st.subheader("Knowledge Base")

    kb_upload = st.file_uploader(
        "Add file",
        type=["pdf", "docx", "md", "csv", "txt"],
        label_visibility="collapsed",
        key="kb_panel_uploader",
    )
    if kb_upload is not None:
        upload_key = f"{kb_upload.name}_{kb_upload.size}"
        if st.session_state.get("last_kb_upload") != upload_key:
            with st.spinner(f"Processing {kb_upload.name}..."):
                result_msg = handle_upload(kb_upload.getvalue(), kb_upload.name)
            st.session_state.last_kb_upload = upload_key
            st.success(result_msg) if "uploaded and indexed" in result_msg else st.error(result_msg)
            st.rerun()

    st.divider()

    db = SessionLocal()
    files = list_kb_files(db)
    if not files:
        st.caption("No files uploaded yet.")
    for r in files:
        status_icon = "✅" if r.processing_status == "COMPLETED" else ("⏳" if r.processing_status not in ("COMPLETED", "FAILED") else "❌")
        col_name, col_del = st.columns([4, 1])
        with col_name:
            st.caption(f"{status_icon} {r.file_name}")
        with col_del:
            if st.button("🗑", key=f"delete_{r.id}", help=f"Delete {r.file_name}"):
                delete_kb_file(db, r.id)
                st.rerun()
    db.close()