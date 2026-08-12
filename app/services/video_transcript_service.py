import re
from typing import List, Tuple, Optional
from app.models.response_models import ExtractedPage
from app.core.logging import get_logger

logger = get_logger("video_transcript_service")

class VideoTranscriptService:
    """Service to parse video transcripts (.txt, .vtt, .srt) into structured timestamped chunks."""

    def parse_transcript_text(self, text: str) -> Tuple[List[ExtractedPage], str, str, Optional[str]]:
        """
        Parses transcript text containing timestamps.
        Returns: (extracted_pages, cleaned_text, extraction_method, source_url)
        """
        lines = text.splitlines()
        source_url = None
        title = None
        cleaned_lines = []

        timestamp_pattern = re.compile(r'^(\d{2}:\d{2}:\d{2}(?:\.\d{3})?|\d{2}:\d{2}(?:\.\d{3})?)\s*[-–]?\s*(.*)$')
        
        blocks = []
        current_time_start = None
        current_text_block = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Header metadata parsing
            if stripped.startswith('#'):
                if 'http' in stripped and not source_url:
                    match = re.search(r'https?://[^\s]+', stripped)
                    if match:
                        source_url = match.group(0)
                elif not title and len(stripped) > 2:
                    title = stripped.lstrip('#').strip()
                continue

            # Timestamp matching
            match = timestamp_pattern.match(stripped)
            if match:
                ts = match.group(1)
                text_content = match.group(2).strip()

                if not current_time_start:
                    current_time_start = ts

                if text_content:
                    current_text_block.append(f"[{ts}] {text_content}")
                    cleaned_lines.append(f"[{ts}] {text_content}")
            else:
                current_text_block.append(stripped)
                cleaned_lines.append(stripped)

        full_cleaned_text = "\n".join(cleaned_lines)
        pages = [ExtractedPage(page_number=1, text=full_cleaned_text)]
        
        return pages, full_cleaned_text, "video_transcript_parser", source_url
