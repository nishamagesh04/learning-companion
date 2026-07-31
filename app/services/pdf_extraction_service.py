import io
import os
from typing import List, Tuple, Callable, Optional
from pypdf import PdfReader
from app.core.config import settings
from app.core.exceptions import PDFValidationError, ExtractionError
from app.core.logging import get_logger
from app.models.response_models import ExtractedPage

logger = get_logger("pdf_extraction_service")

class PDFExtractionService:
    """Handles PDF file validation and memory-efficient page-by-page text extraction."""

    def validate_pdf_file(self, file_bytes: bytes, file_name: str) -> None:
        """Validates PDF file size, extension, and header signature."""
        if not file_name.lower().endswith('.pdf'):
            raise PDFValidationError(f"Invalid file extension for '{file_name}'. Only PDF files are supported.")

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise PDFValidationError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE_MB} MB)."
            )

        if len(file_bytes) < 5 or not file_bytes.startswith(b'%PDF-'):
            raise PDFValidationError(f"File '{file_name}' does not appear to be a valid PDF binary (missing %PDF- header).")

    def extract_text_from_pdf_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[ExtractedPage], str, str]:
        """
        Extracts text from PDF bytes page-by-page with progress reporting.
        Supports 500+ page documents with low memory footprint.
        """
        self.validate_pdf_file(file_bytes, file_name)

        pages: List[ExtractedPage] = []
        raw_text_parts: List[str] = []

        try:
            stream = io.BytesIO(file_bytes)
            reader = PdfReader(stream)
            total_pages = len(reader.pages)

            if total_pages == 0:
                raise ExtractionError(f"PDF '{file_name}' contains no pages.")

            logger.info(f"Extracting text from {total_pages}-page PDF document '{file_name}'...")

            interval = settings.PROGRESS_LOG_INTERVAL_PAGES

            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                pages.append(ExtractedPage(page_number=idx, text=page_text))
                raw_text_parts.append(f"--- Page {idx} ---\n{page_text}")

                # Progress logging & callback for large documents
                if idx % interval == 0 or idx == total_pages:
                    pct = int((idx / total_pages) * 100)
                    if progress_callback:
                        progress_callback(idx, total_pages)
                    elif idx % (interval * 2) == 0 or idx == total_pages:
                        logger.info(f"Extraction Progress: {idx}/{total_pages} pages extracted ({pct}%).")

            full_raw_text = "\n\n".join(raw_text_parts)
            total_clean_len = sum(len(p.text.strip()) for p in pages)

            if total_clean_len == 0:
                raise ExtractionError(
                    f"PDF '{file_name}' contains no readable text. "
                    "It may be a scanned document or image-only PDF requiring OCR."
                )

            logger.info(f"Completed extraction for {total_pages} pages ({total_clean_len} characters) from '{file_name}'.")
            return pages, full_raw_text, "pypdf"

        except PDFValidationError:
            raise
        except ExtractionError:
            raise
        except Exception as e:
            logger.error(f"Failed to extract text from PDF '{file_name}': {str(e)}")
            raise ExtractionError(f"PDF text extraction failed for '{file_name}': {str(e)}") from e

    def extract_text_from_filepath(self, file_path: str) -> Tuple[List[ExtractedPage], str, str]:
        """Helper to extract text directly from a local PDF file path."""
        if not os.path.exists(file_path):
            raise PDFValidationError(f"PDF file not found at path: {file_path}")

        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return self.extract_text_from_pdf_bytes(file_bytes, file_name)
