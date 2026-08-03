import io
import os
import csv
from typing import List, Tuple, Callable, Optional
from app.core.config import settings
from app.core.exceptions import PDFValidationError, ExtractionError
from app.core.logging import get_logger
from app.models.response_models import ExtractedPage

logger = get_logger("csv_extraction_service")

class CSVExtractionService:
    """Handles CSV file validation and text extraction."""

    def validate_csv_file(self, file_bytes: bytes, file_name: str) -> None:
        """Validates CSV file size and extension."""
        if not file_name.lower().endswith('.csv'):
            raise PDFValidationError(f"Invalid file extension for '{file_name}'. Only CSV files are supported.")

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise PDFValidationError(
                f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({settings.MAX_FILE_SIZE_MB} MB)."
            )

    def extract_text_from_csv_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[ExtractedPage], str, str]:
        """
        Extracts text from CSV bytes, converting rows to readable text format.
        """
        self.validate_csv_file(file_bytes, file_name)

        pages: List[ExtractedPage] = []
        raw_text_parts: List[str] = []

        try:
            text = file_bytes.decode('utf-8')
            
            if not text.strip():
                raise ExtractionError(f"CSV '{file_name}' contains no readable text.")

            logger.info(f"Extracting text from CSV document '{file_name}'...")

            # Parse CSV
            stream = io.StringIO(text)
            csv_reader = csv.reader(stream)
            
            rows = list(csv_reader)
            
            if not rows:
                raise ExtractionError(f"CSV '{file_name}' contains no rows.")

            # Group rows into logical pages (e.g., 100 rows per page)
            rows_per_page = 100
            total_pages = (len(rows) + rows_per_page - 1) // rows_per_page

            for page_num in range(total_pages):
                start_idx = page_num * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(rows))
                page_rows = rows[start_idx:end_idx]
                
                # Convert rows to readable text
                page_text_lines = []
                for row in page_rows:
                    page_text_lines.append(" | ".join(str(cell) for cell in row))
                
                page_text = "\n".join(page_text_lines)
                
                pages.append(ExtractedPage(page_number=page_num + 1, text=page_text))
                raw_text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

                # Progress callback
                if progress_callback:
                    progress_callback(page_num + 1, total_pages)

            full_raw_text = "\n\n".join(raw_text_parts)
            total_clean_len = sum(len(p.text.strip()) for p in pages)

            if total_clean_len == 0:
                raise ExtractionError(f"CSV '{file_name}' contains no readable text.")

            logger.info(f"Completed extraction for {total_pages} pages ({total_clean_len} characters) from '{file_name}'.")
            return pages, full_raw_text, "csv-parser"

        except PDFValidationError:
            raise
        except ExtractionError:
            raise
        except UnicodeDecodeError:
            logger.error(f"Failed to decode CSV '{file_name}' as UTF-8")
            raise ExtractionError(f"CSV file '{file_name}' is not valid UTF-8 text") from None
        except Exception as e:
            logger.error(f"Failed to extract text from CSV '{file_name}': {str(e)}")
            raise ExtractionError(f"CSV text extraction failed for '{file_name}': {str(e)}") from e

    def extract_text_from_filepath(self, file_path: str) -> Tuple[List[ExtractedPage], str, str]:
        """Helper to extract text directly from a local CSV file path."""
        if not os.path.exists(file_path):
            raise PDFValidationError(f"CSV file not found at path: {file_path}")

        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        return self.extract_text_from_csv_bytes(file_bytes, file_name)
