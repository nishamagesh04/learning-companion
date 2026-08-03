import io
import os
from typing import List, Tuple, Callable, Optional
from app.core.config import settings
from app.core.exceptions import PDFValidationError, ExtractionError
from app.core.logging import get_logger
from app.models.response_models import ExtractedPage

logger = get_logger("docx_extraction_service")

class DocxExtractionService:
    """Handles DOCX file validation and text extraction."""

    def validate_docx_file(self, file_bytes: bytes, file_name: str) -> None:
        """Validates DOCX file size and extension."""
        if not file_name.lower().endswith('.docx'):
            raise PDFValidationError(f"Invalid file extension for '{file_name}'. Only DOCX files are supported.")

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise PDFValidationError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE_MB} MB)."
            )

        # DOCX files are ZIP archives, check for ZIP signature
        if len(file_bytes) < 4 or not file_bytes.startswith(b'PK\x03\x04'):
            raise PDFValidationError(f"File '{file_name}' does not appear to be a valid DOCX file (missing ZIP header).")

    def extract_text_from_docx_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[ExtractedPage], str, str]:
        """
        Extracts text from DOCX bytes paragraph by paragraph with progress reporting.
        """
        self.validate_docx_file(file_bytes, file_name)

        pages: List[ExtractedPage] = []
        raw_text_parts: List[str] = []

        try:
            from docx import Document

            stream = io.BytesIO(file_bytes)
            doc = Document(stream)

            logger.info(f"Extracting text from DOCX document '{file_name}'...")

            # Extract paragraphs and group them into logical pages
            # DOCX doesn't have explicit page breaks, so we'll group by paragraph count
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            if not paragraphs:
                raise ExtractionError(f"DOCX '{file_name}' contains no readable text.")

            # Group paragraphs into "pages" (each page ~20-30 paragraphs)
            paragraphs_per_page = 25
            total_pages = (len(paragraphs) + paragraphs_per_page - 1) // paragraphs_per_page

            for page_num in range(total_pages):
                start_idx = page_num * paragraphs_per_page
                end_idx = min(start_idx + paragraphs_per_page, len(paragraphs))
                page_paragraphs = paragraphs[start_idx:end_idx]
                page_text = "\n".join(page_paragraphs)
                
                pages.append(ExtractedPage(page_number=page_num + 1, text=page_text))
                raw_text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

                # Progress callback
                if progress_callback:
                    progress_callback(page_num + 1, total_pages)

            full_raw_text = "\n\n".join(raw_text_parts)
            total_clean_len = sum(len(p.text.strip()) for p in pages)

            if total_clean_len == 0:
                raise ExtractionError(f"DOCX '{file_name}' contains no readable text.")

            logger.info(f"Completed extraction for {total_pages} logical pages ({total_clean_len} characters) from '{file_name}'.")
            return pages, full_raw_text, "python-docx"

        except PDFValidationError:
            raise
        except ExtractionError:
            raise
        except ImportError:
            logger.error("python-docx library not installed. Install with: pip install python-docx")
            raise ExtractionError("python-docx library not installed. Install with: pip install python-docx") from None
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX '{file_name}': {str(e)}")
            raise ExtractionError(f"DOCX text extraction failed for '{file_name}': {str(e)}") from e

    def extract_text_from_filepath(self, file_path: str) -> Tuple[List[ExtractedPage], str, str]:
        """Helper to extract text directly from a local DOCX file path."""
        if not os.path.exists(file_path):
            raise PDFValidationError(f"DOCX file not found at path: {file_path}")

        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return self.extract_text_from_docx_bytes(file_bytes, file_name)
