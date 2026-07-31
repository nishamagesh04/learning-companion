from app.services.text_cleaning_service import TextCleaningService
from app.models.response_models import ExtractedPage

def test_remove_page_numbers():
    cleaner = TextCleaningService()
    text = "Introduction to AI\nPage 1 of 10\nSome content here.\n- 1 -\nPage 2"
    cleaned = cleaner.remove_page_numbers_and_markers(text)
    assert "Page 1 of 10" not in cleaned
    assert "- 1 -" not in cleaned
    assert "Some content here." in cleaned

def test_fix_hyphenated_line_endings():
    cleaner = TextCleaningService()
    text = "This is an imple-\nmentation of the algorithm."
    cleaned = cleaner.fix_hyphenated_line_endings(text)
    assert "implementation" in cleaned
    assert "imple-\nmentation" not in cleaned

def test_detect_repeated_headers():
    cleaner = TextCleaningService()
    pages = [
        ExtractedPage(page_number=1, text="HEADER LOGO\nPage content 1\nFooter info"),
        ExtractedPage(page_number=2, text="HEADER LOGO\nPage content 2\nFooter info"),
        ExtractedPage(page_number=3, text="HEADER LOGO\nPage content 3\nFooter info")
    ]
    repeated = cleaner.detect_repeated_headers_footers(pages)
    assert "HEADER LOGO" in repeated
    assert "Footer info" in repeated

def test_clean_extracted_pages():
    cleaner = TextCleaningService()
    pages = [
        ExtractedPage(page_number=1, text="ACME CORP CONFIDENTIAL\nIntroduction to FastAPI.\nThis is an exam-\nple.\nPage 1 of 5"),
        ExtractedPage(page_number=2, text="ACME CORP CONFIDENTIAL\nFastAPI handles async routes efficiently.\nPage 2 of 5")
    ]
    cleaned_pages, full_text = cleaner.clean_extracted_pages(pages)
    assert len(cleaned_pages) == 2
    assert "ACME CORP CONFIDENTIAL" not in full_text
    assert "example." in full_text
    assert "Page 1 of 5" not in full_text
