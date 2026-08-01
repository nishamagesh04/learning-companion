import re
from typing import Optional, List, Tuple, Dict, Set
from app.core.logging import get_logger
from app.models.response_models import ExtractedPage

logger = get_logger("text_cleaning_service")

class TextCleaningService:
    """Performs structured text cleaning on extracted PDF content."""

    def detect_repeated_headers_footers(self, pages: List[ExtractedPage]) -> Set[str]:
        """
        Identifies header/footer lines that repeat across multiple pages (>= 60% of pages, or on all pages if 2 pages).
        """
        if len(pages) < 2:
            return set()

        line_counts: Dict[str, int] = {}
        for page in pages:
            lines = [l.strip() for l in page.text.splitlines() if l.strip()]
            if not lines:
                continue
            
            # Inspect first 2 and last 2 lines on each page
            candidate_lines = set(lines[:2] + lines[-2:])
            for line in candidate_lines:
                # Ignore very long lines or standard headings
                if len(line) < 100 and not line.startswith('#'):
                    line_counts[line] = line_counts.get(line, 0) + 1

        threshold = 2 if len(pages) == 2 else max(2, int(len(pages) * 0.6))
        repeated = {line for line, count in line_counts.items() if count >= threshold}
        logger.info(f"Identified {len(repeated)} repeated header/footer line patterns across pages.")
        return repeated

    def remove_page_numbers_and_markers(self, text: str) -> str:
        """Removes common page number formats (e.g., 'Page 1 of 10', '- 3 -', 'Page 4')."""
        patterns = [
            r'(?i)^\s*page\s+\d+\s+of\s+\d+\s*$',
            r'(?i)^\s*page\s+\d+\s*$',
            r'^\s*-\s*\d+\s*-\s*$',
            r'^\s*\d+\s*$',  # Standalone line numbers
            r'^---\s*Page\s+\d+\s*---$'
        ]
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if any(re.match(pat, stripped) for pat in patterns):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def fix_hyphenated_line_endings(self, text: str) -> str:
        """
        Fixes words split across lines with hyphens (e.g. 'imple-\nmentation' -> 'implementation').
        """
        # Match word ending with hyphen followed by newline and lowercase continuation
        pattern = r'(\b[A-Za-z]+)-\n([a-z]+\b)'
        return re.sub(pattern, r'\1\2', text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalizes excessive blank lines and spaces while preserving paragraph boundaries.
        """
        # Replace 3 or more consecutive newlines with 2 newlines (paragraph boundary)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize trailing spaces on lines
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def clean_extracted_pages(self, pages: List[ExtractedPage]) -> Tuple[List[ExtractedPage], str]:
        """
        Cleans extracted page objects and returns cleaned pages along with full cleaned text.
        """
        repeated_lines = self.detect_repeated_headers_footers(pages)

        cleaned_pages: List[ExtractedPage] = []
        full_cleaned_parts: List[str] = []

        for page in pages:
            lines = page.text.splitlines()
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped in repeated_lines:
                    continue
                filtered_lines.append(line)

            page_text = "\n".join(filtered_lines)
            page_text = self.remove_page_numbers_and_markers(page_text)
            page_text = self.fix_hyphenated_line_endings(page_text)
            page_text = self.normalize_whitespace(page_text)

            cleaned_pages.append(ExtractedPage(page_number=page.page_number, text=page_text))
            if page_text:
                full_cleaned_parts.append(f"## Page {page.page_number}\n\n{page_text}")

        full_cleaned_text = "\n\n".join(full_cleaned_parts)
        return cleaned_pages, full_cleaned_text
