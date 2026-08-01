import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database_models import Base, ContentChunk, LearningResource
from app.models.request_models import RetrievalRequest
from app.repositories.chunk_repository import ChunkRepository
from app.services.retrieval_service import RetrievalService
from app.integrations.embedding_client import GeminiEmbeddingClient

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create sample resource
    resource = LearningResource(
        title="Test RAG Resource",
        resource_type="PDF",
        file_name="test.pdf",
        file_path="/test.pdf",
        processing_status="COMPLETED"
    )
    session.add(resource)
    session.commit()

    client = GeminiEmbeddingClient()
    
    # Create sample chunks with embeddings
    c1_text = "Retrieval Augmented Generation reduces AI hallucinations."
    c2_text = "Python SQLAlchemy handles database models and sessions."
    
    emb1 = client._generate_mock_embedding(c1_text)
    emb2 = client._generate_mock_embedding(c2_text)

    chunk1 = ContentChunk(
        resource_id=resource.id,
        chunk_index=1,
        chunk_text=c1_text,
        token_count=10,
        page_number=1,
        embedding=emb1,
        embedding_model="models/text-embedding-004"
    )
    chunk2 = ContentChunk(
        resource_id=resource.id,
        chunk_index=2,
        chunk_text=c2_text,
        token_count=10,
        page_number=2,
        embedding=emb2,
        embedding_model="models/text-embedding-004"
    )
    session.add_all([chunk1, chunk2])
    session.commit()

    yield session
    session.close()

def test_chunk_repository_find_similar_chunks(db_session):
    repo = ChunkRepository(db_session)
    client = GeminiEmbeddingClient()
    
    query_text = "Tell me about RAG and AI hallucinations"
    query_emb = client._generate_mock_embedding(query_text)
    
    results = repo.find_similar_chunks(query_embedding=query_emb, top_k=2, min_score=-1.0)
    assert len(results) == 2

    top_chunk, top_score = results[0]
    assert top_score > 0.0
    assert "Retrieval" in top_chunk.chunk_text

def test_retrieval_service(db_session):
    service = RetrievalService(db_session)
    req = RetrievalRequest(query="Tell me about RAG and AI hallucinations", top_k=1)
    
    response = service.retrieve_context(req)
    assert response.total_matches == 1
    assert response.top_k == 1
    assert len(response.retrieved_chunks) == 1
    assert "Retrieval" in response.retrieved_chunks[0].chunk_text
