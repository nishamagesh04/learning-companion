import re
from typing import List, Optional
from app.core.config import settings
from app.core.logging import get_logger
from app.models.response_models import ExtractedPage, ChunkPreview

logger = get_logger("chunking_service")

class ChunkingService:
    """Provides Fixed-Length and Structure-Aware (Semantic) Chunking strategies."""

    def estimate_token_count(self, text: str) -> int:
        """Estimates token count (~4 characters per token or word count basis)."""
        words = text.split()
        return max(1, int(len(text) / 4)) if text else 0

    def extract_section_heading(self, text: str) -> Optional[str]:
        """Extracts the first heading line if present."""
        heading_pattern = r'^(#{1,6}\s+.+|\d+(\.\d+)*\s+[A-Z].+|[A-Z0-9\s_-]{3,50}:)'
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(heading_pattern, stripped):
                return stripped[:100]
        return None

    def fixed_length_chunking(
        self,
        pages: List[ExtractedPage],
        max_chunk_size: int = settings.MAX_CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE
    ) -> List[ChunkPreview]:
        """
        Splits text into fixed character/token length chunks with overlapping windows.
        Maintains page number attribution for each chunk.
        """
        chunks: List[ChunkPreview] = []
        chunk_index = 0

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            heading = self.extract_section_heading(text)
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + max_chunk_size, text_len)

                # Try to break at paragraph or sentence boundary if not at end of text
                if end < text_len:
                    last_space = text.rfind('\n\n', start, end)
                    if last_space == -1 or last_space < start + min_chunk_size:
                        last_space = text.rfind('. ', start, end)
                    if last_space != -1 and last_space > start + min_chunk_size:
                        end = last_space + 1

                chunk_str = text[start:end].strip()

                if len(chunk_str) >= min_chunk_size or start == 0:
                    chunk_index += 1
                    chunks.append(
                        ChunkPreview(
                            chunk_index=chunk_index,
                            chunk_text=chunk_str,
                            token_count=self.estimate_token_count(chunk_str),
                            page_number=page.page_number,
                            section_heading=heading
                        )
                    )

                if end >= text_len:
                    break

                # Advance window with overlap
                start = max(start + 1, end - chunk_overlap)

        logger.info(f"Fixed-length chunking created {len(chunks)} chunks.")
        return chunks

    def semantic_structure_chunking(
        self,
        pages: List[ExtractedPage],
        max_chunk_size: int = settings.MAX_CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE
    ) -> List[ChunkPreview]:
        """
        Structure-aware semantic chunking.
        Groups paragraphs under section headings while maintaining page bounds.
        """
        chunks: List[ChunkPreview] = []
        chunk_index = 0

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            # Split text by double newlines or heading markers
            paragraphs = re.split(r'\n\n+', text)
            current_chunk_paragraphs: List[str] = []
            current_len = 0
            current_heading: Optional[str] = None

            for para in paragraphs:
                para_stripped = para.strip()
                if not para_stripped:
                    continue

                # Check if paragraph is a heading
                detected_heading = self.extract_section_heading(para_stripped)
                if detected_heading:
                    current_heading = detected_heading

                para_len = len(para_stripped)

                # If adding this paragraph exceeds max chunk size and we already have accumulated text
                if current_len + para_len > max_chunk_size and current_chunk_paragraphs:
                    chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                    chunk_index += 1
                    chunks.append(
                        ChunkPreview(
                            chunk_index=chunk_index,
                            chunk_text=chunk_str,
                            token_count=self.estimate_token_count(chunk_str),
                            page_number=page.page_number,
                            section_heading=current_heading
                        )
                    )

                    # Carry over overlap if possible
                    overlap_str = chunk_str[-chunk_overlap:] if len(chunk_str) > chunk_overlap else ""
                    current_chunk_paragraphs = [overlap_str, para_stripped] if overlap_str else [para_stripped]
                    current_len = sum(len(p) for p in current_chunk_paragraphs)
                else:
                    current_chunk_paragraphs.append(para_stripped)
                    current_len += para_len

            # Flush remaining paragraph buffer for this page
            if current_chunk_paragraphs:
                chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                if chunk_str:
                    chunk_index += 1
                    chunks.append(
                        ChunkPreview(
                            chunk_index=chunk_index,
                            chunk_text=chunk_str,
                            token_count=self.estimate_token_count(chunk_str),
                            page_number=page.page_number,
                            section_heading=current_heading
                        )
                    )

        logger.info(f"Structure-aware semantic chunking created {len(chunks)} chunks.")
        return chunks

    def chunk_content(
        self,
        pages: List[ExtractedPage],
        strategy: str = "semantic",
        max_chunk_size: int = settings.MAX_CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE
    ) -> List[ChunkPreview]:
        """Entry point for executing selected chunking strategy."""
        if strategy == "fixed":
            return self.fixed_length_chunking(pages, max_chunk_size, chunk_overlap, min_chunk_size)
        else:
            return self.semantic_structure_chunking(pages, max_chunk_size, chunk_overlap, min_chunk_size)
