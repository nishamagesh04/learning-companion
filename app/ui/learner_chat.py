import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from app.core.database import SessionLocal
from app.models.database_models import ConversationSession, ChatMessage, ChatResponse
from app.services.chat_service import answer_question

st.set_page_config(page_title="AI Learning Companion", layout="centered")
st.title("AI Learning Companion")

if "session_id" not in st.session_state:
    db = SessionLocal()
    session = ConversationSession()
    db.add(session); db.commit(); db.refresh(session)
    st.session_state.session_id = session.id
    st.session_state.history = []
    db.close()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])

question = st.chat_input("Ask a question about the course material...")

if question:
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            db = SessionLocal()
            user_msg = ChatMessage(session_id=st.session_state.session_id, role="user", message_text=question)
            db.add(user_msg); db.commit(); db.refresh(user_msg)

            answer, sources = answer_question(db, question)
            st.write(answer)
            if sources:
                with st.expander("Sources used"):
                    for c in sources:
                        st.caption(f"Chunk {c.chunk_index} (resource {c.resource_id}): {c.chunk_text[:200]}...")

            resp = ChatResponse(message_id=user_msg.id, answer_text=answer, model_name="gemini-2.0-flash")
            db.add(resp); db.commit()
            db.close()

    st.session_state.history.append({"role": "assistant", "text": answer})