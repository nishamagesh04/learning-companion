from typing import List, Tuple, Callable, Optional
from app.core.exceptions import PDFValidationError, ExtractionError
from app.core.logging import get_logger
from app.models.response_models import ExtractedPage
from app.services.pdf_extraction_service import PDFExtractionService
from app.services.docx_extraction_service import DocxExtractionService
from app.services.markdown_extraction_service import MarkdownExtractionService
from app.services.csv_extraction_service import CSVExtractionService

logger = get_logger("text_extraction_service")

class TextExtractionService:
    """Generic text extraction service that handles multiple document formats (PDF, DOCX, Markdown, CSV)."""

    def __init__(self):
        self.pdf_service = PDFExtractionService()
        self.docx_service = DocxExtractionService()
        self.markdown_service = MarkdownExtractionService()
        self.csv_service = CSVExtractionService()

    def validate_document_file(self, file_bytes: bytes, file_name: str) -> None:
        """Validates document file based on its extension."""
        if file_name.lower().endswith('.pdf'):
            self.pdf_service.validate_pdf_file(file_bytes, file_name)
        elif file_name.lower().endswith('.docx'):
            self.docx_service.validate_docx_file(file_bytes, file_name)
        elif file_name.lower().endswith('.md'):
            self.markdown_service.validate_markdown_file(file_bytes, file_name)
        elif file_name.lower().endswith('.csv'):
            self.csv_service.validate_csv_file(file_bytes, file_name)
        else:
            raise PDFValidationError(f"Unsupported file extension for '{file_name}'. Only PDF, DOCX, Markdown, and CSV files are supported.")

    def extract_text_from_document_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[ExtractedPage], str, str]:
        """
        Extracts text from document bytes based on file type.
        Returns: (pages, full_raw_text, extraction_method)
        """
        if file_name.lower().endswith('.pdf'):
            return self.pdf_service.extract_text_from_pdf_bytes(file_bytes, file_name, progress_callback)
        elif file_name.lower().endswith('.docx'):
            return self.docx_service.extract_text_from_docx_bytes(file_bytes, file_name, progress_callback)
        elif file_name.lower().endswith('.md'):
            return self.markdown_service.extract_text_from_markdown_bytes(file_bytes, file_name, progress_callback)
        elif file_name.lower().endswith('.csv'):
            return self.csv_service.extract_text_from_csv_bytes(file_bytes, file_name, progress_callback)
        else:
            raise ExtractionError(f"Unsupported file type for '{file_name}'. Only PDF, DOCX, Markdown, and CSV files are supported.")

    def extract_text_from_filepath(self, file_path: str) -> Tuple[List[ExtractedPage], str, str]:
        """Helper to extract text directly from a local document file path."""
        if file_path.lower().endswith('.pdf'):
            return self.pdf_service.extract_text_from_filepath(file_path)
        elif file_path.lower().endswith('.docx'):
            return self.docx_service.extract_text_from_filepath(file_path)
        elif file_path.lower().endswith('.md'):
            return self.markdown_service.extract_text_from_filepath(file_path)
        elif file_path.lower().endswith('.csv'):
            return self.csv_service.extract_text_from_filepath(file_path)
        else:
            raise ExtractionError(f"Unsupported file type for '{file_path}'. Only PDF, DOCX, Markdown, and CSV files are supported.")
