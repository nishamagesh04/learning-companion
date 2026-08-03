import io
import os
from typing import List, Tuple, Callable, Optional
from app.core.config import settings
from app.core.exceptions import PDFValidationError, ExtractionError
from app.core.logging import get_logger
from app.models.response_models import ExtractedPage

logger = get_logger("markdown_extraction_service")

class MarkdownExtractionService:
    """Handles Markdown file validation and text extraction."""

    def validate_markdown_file(self, file_bytes: bytes, file_name: str) -> None:
        """Validates Markdown file size and extension."""
        if not file_name.lower().endswith('.md'):
            raise PDFValidationError(f"Invalid file extension for '{file_name}'. Only Markdown files are supported.")

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise PDFValidationError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE_MB} MB)."
            )

    def extract_text_from_markdown_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[ExtractedPage], str, str]:
        """
        Extracts text from Markdown bytes, splitting by headers into logical sections.
        """
        self.validate_markdown_file(file_bytes, file_name)

        pages: List[ExtractedPage] = []
        raw_text_parts: List[str] = []

        try:
            text = file_bytes.decode('utf-8')
            
            if not text.strip():
                raise ExtractionError(f"Markdown '{file_name}' contains no readable text.")

            logger.info(f"Extracting text from Markdown document '{file_name}'...")

            # Split by headers (#, ##, ###) to create logical sections
            lines = text.split('\n')
            current_section = []
            section_number = 0
            
            for line in lines:
                # Check if this is a header
                if line.startswith('#'):
                    # Save previous section if it exists
                    if current_section:
                        section_text = '\n'.join(current_section).strip()
                        if section_text:
                            pages.append(ExtractedPage(page_number=section_number, text=section_text))
                            raw_text_parts.append(f"--- Section {section_number} ---\n{section_text}")
                            section_number += 1
                    current_section = [line]
                else:
                    current_section.append(line)
            
            # Don't forget the last section
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    pages.append(ExtractedPage(page_number=section_number + 1, text=section_text))
                    raw_text_parts.append(f"--- Section {section_number + 1} ---\n{section_text}")

            # If no headers found, treat entire document as one section
            if not pages:
                pages.append(ExtractedPage(page_number=1, text=text))
                raw_text_parts.append(f"--- Section 1 ---\n{text}")

            full_raw_text = "\n\n".join(raw_text_parts)
            total_clean_len = sum(len(p.text.strip()) for p in pages)

            if total_clean_len == 0:
                raise ExtractionError(f"Markdown '{file_name}' contains no readable text.")

            logger.info(f"Completed extraction for {len(pages)} sections ({total_clean_len} characters) from '{file_name}'.")
            return pages, full_raw_text, "markdown-parser"

        except PDFValidationError:
            raise
        except ExtractionError:
            raise
        except UnicodeDecodeError:
            logger.error(f"Failed to decode Markdown '{file_name}' as UTF-8")
            raise ExtractionError(f"Markdown file '{file_name}' is not valid UTF-8 text") from None
        except Exception as e:
            logger.error(f"Failed to extract text from Markdown '{file_name}': {str(e)}")
            raise ExtractionError(f"Markdown text extraction failed for '{file_name}': {str(e)}") from e

    def extract_text_from_filepath(self, file_path: str) -> Tuple[List[ExtractedPage], str, str]:
        """Helper to extract text directly from a local Markdown file path."""
        if not os.path.exists(file_path):
            raise PDFValidationError(f"Markdown file not found at path: {file_path}")

        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return self.extract_text_from_markdown_bytes(file_bytes, file_name)
