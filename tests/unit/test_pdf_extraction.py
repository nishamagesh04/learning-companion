import pytest
import io
from pypdf import PdfWriter
from app.services.pdf_extraction_service import PDFExtractionService
from app.core.exceptions import PDFValidationError, ExtractionError

def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to generate valid PDF binary bytes in memory for testing."""
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=612, height=792)
        # Note: blank pages in pypdf might not have text stream unless added,
        # but pypdf Writer allows page creation.
    
    # Simple binary PDF output
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()

def test_validate_pdf_invalid_extension():
    service = PDFExtractionService()
    with pytest.raises(PDFValidationError) as exc:
        service.validate_pdf_file(b"%PDF-1.4 test", "sample.txt")
    assert "Only PDF files are supported" in str(exc.value)

def test_validate_pdf_invalid_header():
    service = PDFExtractionService()
    with pytest.raises(PDFValidationError) as exc:
        service.validate_pdf_file(b"INVALID_HEADER_BYTES", "sample.pdf")
    assert "missing %PDF- header" in str(exc.value)

def test_validate_pdf_size_exceeded():
    service = PDFExtractionService()
    large_bytes = b"%PDF-1.5 " + b"X" * (300 * 1024 * 1024)  # 300 MB
    with pytest.raises(PDFValidationError) as exc:
        service.validate_pdf_file(large_bytes, "large.pdf")
    assert "exceeds maximum allowed size" in str(exc.value)
