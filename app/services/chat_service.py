import numpy as np
from sqlalchemy.orm import Session
from google import genai
from app.core.config import settings
from app.integrations.embedding_client import GeminiEmbeddingClient
from app.models.database_models import ContentChunk

_embed_client = GeminiEmbeddingClient()
_gen_client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None

def answer_question(db: Session, question: str, top_k: int = 4):
    q_vec = np.array(_embed_client.generate_single_embedding_with_retry(question))

    chunks = db.query(ContentChunk).filter(ContentChunk.embedding.isnot(None)).all()
    scored = []
    for c in chunks:
        c_vec = np.array(c.embedding)
        sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:top_k]]

    if not top_chunks:
        return "This topic isn't covered in the indexed learning material yet.", []

    context = "\n\n".join(f"[Source {i+1}] {c.chunk_text}" for i, c in enumerate(top_chunks))
    prompt = f"""Answer the learner's question using ONLY the context below. If the answer isn't in the context, say so clearly.

Context:
{context}

Question: {question}
Answer:"""

    if _gen_client:
        try:
            resp = _gen_client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
            answer = resp.text
        except Exception as e:
            answer = f"I couldn't generate an answer right now (Gemini API error: {type(e).__name__}). Please try again shortly."
    else:
        answer = "Gemini API key not configured — cannot generate an answer."

    return answer, top_chunks