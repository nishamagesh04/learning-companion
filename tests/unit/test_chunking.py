from app.services.chunking_service import ChunkingService
from app.models.response_models import ExtractedPage

def test_fixed_length_chunking():
    chunker = ChunkingService()
    pages = [
        ExtractedPage(
            page_number=1,
            text="Sentence one. " * 30 + "\n\nSentence two. " * 30
        )
    ]
    chunks = chunker.chunk_content(pages, strategy="fixed", max_chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    assert chunks[0].page_number == 1
    assert chunks[0].token_count > 0

def test_semantic_structure_chunking():
    chunker = ChunkingService()
    pages = [
        ExtractedPage(
            page_number=1,
            text="1. Introduction\n\nThis is paragraph one introducing the concept.\n\n2. Architecture\n\nThis is paragraph two detailing system components."
        )
    ]
    chunks = chunker.chunk_content(pages, strategy="semantic", max_chunk_size=70, chunk_overlap=15)
    assert len(chunks) >= 2
    assert any("Introduction" in (c.section_heading or "") or "Introduction" in c.chunk_text for c in chunks)
    assert any("Architecture" in (c.section_heading or "") or "Architecture" in c.chunk_text for c in chunks)
