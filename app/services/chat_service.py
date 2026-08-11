from sqlalchemy.orm import Session
from google import genai
from app.core.config import settings
from app.services.retrieval_service import RetrievalService
from app.models.request_models import RetrievalRequest

_gen_client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None


def answer_question(db: Session, question: str, top_k: int = 4):
    retrieval_service = RetrievalService(db_session=db)

    try:
        request = RetrievalRequest(query=question, top_k=top_k, min_similarity_score=0.3)
        retrieval_response = retrieval_service.retrieve_context(request)
    except Exception as e:
        return f"I'm temporarily unable to search the knowledge base ({type(e).__name__}). Please try again shortly.", []

    if not retrieval_response.retrieved_chunks:
        return "I couldn't find anything on that in the uploaded material yet — try rephrasing, or check if the relevant file has been uploaded.", []

    context = retrieval_service.format_context_for_prompt(retrieval_response)
    prompt = f"""You are a friendly, knowledgeable learning assistant helping a student understand their course material.
Answer naturally and conversationally, the way a helpful tutor would - not like you're reading from a manual.
Use ONLY the context below to answer. If the context doesn't cover the question, say so plainly and warmly,
don't guess or make things up.

{context}

Question: {question}
Answer:"""

    if _gen_client:
        try:
            resp = _gen_client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
            answer = resp.text
        except Exception as e:
            answer = f"I couldn't generate an answer right now ({type(e).__name__}). Please try again shortly."
    else:
        answer = "Gemini API key not configured — cannot generate an answer."

    return answer, retrieval_response.retrieved_chunks